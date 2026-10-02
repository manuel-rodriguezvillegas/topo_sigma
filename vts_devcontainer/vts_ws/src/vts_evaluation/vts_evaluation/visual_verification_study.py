"""Fixed-node study of local image verification and fixed-budget view selection."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from vts_core.artifacts import atomic_json, file_digest, identity
from vts_core.mapper import TopologicalMapper
from vts_core.topo_graph import TopoGraph
from vts_evaluation.closure_comparison import aggregate, load_verified, replay, score
from vts_evaluation.distinctive_memory import (
    LocalMatcher, MatchEvidence, decide_extra, replace_views, source_images,
)
from vts_evaluation.frame_recovery_comparison import load_frames
from vts_evaluation.summarize_experiments import ENVIRONMENTS
from vts_players.cid_sims_data import load_ground_truth, load_wheel_odometry, timestamped_color_images
from vts_players.cold_odometry import load_cold_odometry

RADII = (.5, 1., 2., 3.)


def frame_paths(directory: Path, dataset: Path, count: int) -> list[str]:
    manifest = json.loads((directory / 'run_manifest.json').read_text())
    player = manifest['parameters']['player']
    root = dataset / Path(manifest['inputs']['root']).relative_to('/workspace/encoder')
    if 'frame_stride' in player:
        odom = load_wheel_odometry(str(root / player['odometry_file']))
        gt = load_ground_truth(str(root / player['ground_truth_file']))
        raw = timestamped_color_images(str(root / player['color_subdir']))[::player['frame_stride']]
    else:
        odom = load_cold_odometry(str(root / player['odometry_subpath']))
        gt = None
        raw = sorted((float(p.name.split('_')[0][1:]), str(p))
                     for p in (root / player['images_subdir']).glob('*.jpeg'))
    paths = []
    for timestamp, filename in raw:
        try:
            odom.at(timestamp, player['max_odometry_gap_s'])
            if gt is not None:
                gt.at(timestamp, player['max_ground_truth_gap_s'])
        except ValueError:
            continue
        paths.append(filename)
    if len(paths) != count:
        raise ValueError('Camera order and cached descriptors disagree')
    return paths


def verify(graph: TopoGraph, images: dict[int, list[np.ndarray]],
           visual_outlier_k: float, competition: bool) -> tuple[TopoGraph, list[dict]]:
    mapped = replay(graph, 'cosine', sequence_policy='self_only', recovery_mode='off',
                    visual_outlier_k=visual_outlier_k)
    matcher = LocalMatcher()
    diagnostics = []
    for event in mapped.loop_events:
        q = event.query_node_id
        if event.accepted or q < 2:
            continue
        ids = range(q - 1)
        similarities = {i: TopologicalMapper._node_similarity(graph.nodes[q], graph.nodes[i])
                        for i in ids}
        proposed = sorted(ids, key=lambda i: (-similarities[i], i))[:5]
        for i in [q, *proposed]:
            for view, image in enumerate(images[i]):
                matcher.prepare(f'{i}:{view}', image)
        evidence = {}
        for i in proposed:
            best = MatchEvidence()
            for a, query_image in enumerate(images[q]):
                query_key = f'{q}:{a}'
                for b, candidate_image in enumerate(images[i]):
                    candidate_key = f'{i}:{b}'
                    if competition:
                        competitors = [f'{other}:{view}' for other in proposed if other != i
                                       for view in range(len(images[other]))]
                        result = matcher.compare_competition(query_key, candidate_key,
                                                             competitors)
                    else:
                        result = matcher.compare(query_key, candidate_key)
                    if result.score > best.score:
                        best = result
            evidence[i] = best
        chosen = decide_extra(evidence)
        diagnostics.append({'query': q, 'candidates': proposed,
                            'scores': {str(i): asdict(x) for i, x in evidence.items()},
                            'selected': chosen})
        if chosen is not None:
            event.accepted = True
            event.matched_node_id = chosen
            event.similarity = similarities[chosen]
            event.reason = 'accepted_local_image_evidence'
            mapped.add_edge(q, chosen, edge_type='loop')
    return mapped, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('output/dinov3_revisits'))
    parser.add_argument('--dataset', type=Path, default=Path('../../encoder'))
    parser.add_argument('--cache', type=Path, default=Path('output/cache/descriptors'))
    parser.add_argument('--output', type=Path, default=Path('output/discriminative_memory'))
    parser.add_argument('--stage', choices=('local', 'memory'), required=True)
    parser.add_argument('--verification', choices=('plain', 'competitive'), default='competitive')
    parser.add_argument('--environments', nargs='+', choices=ENVIRONMENTS, default=ENVIRONMENTS)
    args = parser.parse_args()
    protocol = {
        'study': 'distinctive_memory_v3', 'stage': args.stage,
        'verification': args.verification,
        'parameters': {'sift_features': 1200, 'ratio': .75, 'ransac_px': 3.,
                       'min_inliers': 12, 'min_coverage': .05, 'winner_ratio': 1.5,
                       'top_k': 5, 'memory_budget': 3, 'discrimination_weight': .2},
        'source': str(args.source.resolve()),
        'inputs': {n: load_verified(args.source / n)[1] for n in args.environments},
        'full_graph_hashes': {n: json.loads((args.source / n / 'completion.json').read_text())
                              ['artifacts']['graph_0.pkl'] for n in args.environments},
        'code': {Path(__file__).name: file_digest(__file__),
                 'distinctive_memory.py': file_digest(Path(__file__).with_name('distinctive_memory.py'))},
        'opencv': cv2.__version__, 'tolerances': RADII,
    }
    fingerprint = identity(protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'study_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Different study already exists at output path')
    atomic_json(manifest_path, {'fingerprint': fingerprint, **protocol})
    results = {}
    for name in args.environments:
        source, meta = load_verified(args.source / name)
        full = source_images(args.source / name)
        if any(len(full.nodes[i].views) != len(source.nodes[i].view_features)
               for i in source.nodes):
            raise ValueError('Full and compact node views disagree')
        variants = {'original': (source, {i: full.nodes[i].views for i in source.nodes})}
        selected = {}
        if args.stage == 'memory':
            features, _, anchors, starts, ends, _ = load_frames(
                args.source / name, args.dataset, args.cache, source)
            paths = frame_paths(args.source / name, args.dataset, len(features))
            for policy in ('diverse', 'discriminative'):
                graph, ids = replace_views(source, features, starts, ends, anchors, policy)
                images = {i: [cv2.imread(paths[j]) for j in ids[i]] for i in ids}
                if any(x is None for values in images.values() for x in values):
                    raise ValueError('Failed to read a selected image')
                variants[policy] = graph, images
                selected[policy] = ids
        results[name] = {'variants': {}, 'selected_frames': selected}
        for policy, (graph, images) in variants.items():
            baseline = replay(graph, 'cosine', sequence_policy='self_only', recovery_mode='off',
                              visual_outlier_k=meta['parameters']['visual_outlier_k'])
            local, diagnostics = verify(graph, images,
                                        meta['parameters']['visual_outlier_k'],
                                        args.verification == 'competitive')
            for method, mapped in (('baseline', baseline), ('local', local)):
                label = f'{policy}_{method}'
                results[name]['variants'][label] = {
                    'metrics': [score(mapped, source, radius) for radius in RADII],
                    'decisions': [asdict(e) for e in mapped.loop_events],
                    'diagnostics': diagnostics if method == 'local' else []}
        atomic_json(args.output / 'progress.json', results)
        print(f'{name}: compared {len(variants)} memories', flush=True)
    labels = list(next(iter(results.values()))['variants'])
    totals = {label: [aggregate([r['variants'][label]['metrics'][i] for r in results.values()])
                      for i in range(len(RADII))] for label in labels}
    atomic_json(args.output / 'summary.json', {'fingerprint': fingerprint,
                                              'protocol': protocol, 'aggregate': totals,
                                              'results': results})
    for label, rows in totals.items():
        print(label, rows[2], flush=True)


if __name__ == '__main__':
    main()
