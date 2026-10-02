"""Cache-only screening of B on the same representative-frame queries as A."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

from vts_core import artifacts, descriptor_cache, directional, local_recovery, mapper, matching, metrics, topo_graph
from vts_core.artifacts import atomic_json, file_digest, identity
from vts_core.descriptor_cache import DescriptorCache
from vts_core.topo_graph import TopoGraph
from vts_evaluation import closure_comparison, frame_recovery
from vts_evaluation.closure_comparison import load_verified, replay, score, aggregate
from vts_evaluation.frame_recovery import FrameConfig, recover, select_endpoint
from vts_evaluation.summarize_experiments import ENVIRONMENTS
from vts_players.cid_sims_data import load_wheel_odometry, timestamped_color_images, load_ground_truth
from vts_players.cold_odometry import load_cold_odometry
from vts_players import cold_odometry, cid_sims_data


class CacheOnly:
    def extract(self, image: np.ndarray) -> np.ndarray:
        raise RuntimeError('Missing cached descriptor; inference is prohibited in this experiment')


def load_frames(directory: Path, dataset: Path, cache_root: Path, graph: TopoGraph) -> tuple:
    manifest = json.loads((directory / 'run_manifest.json').read_text())
    player = manifest['parameters']['player']
    root = dataset / Path(manifest['inputs']['root']).relative_to('/workspace/encoder')
    cache = DescriptorCache(CacheOnly(), str(cache_root), manifest['encoder'])
    if 'frame_stride' in player:
        odom_path = root / player['odometry_file']
        odom = load_wheel_odometry(str(odom_path))
        gt = load_ground_truth(str(root / player['ground_truth_file']))
        frames = timestamped_color_images(str(root / player['color_subdir']))[::player['frame_stride']]
    else:
        odom_path = root / player['odometry_subpath']
        odom = load_cold_odometry(str(odom_path))
        frames = sorted((float(p.name.split('_')[0][1:]), str(p))
                        for p in (root / player['images_subdir']).glob('*.jpeg'))
        gt = None
    expected = manifest['inputs']['files']
    for path in [odom_path] + ([root / player['ground_truth_file']] if gt is not None else []):
        if file_digest(path) != expected[str(path.relative_to(root))]:
            raise ValueError(f'Input changed: {path}')
    features, poses = [], []
    digest = hashlib.sha256()
    for timestamp, filename in frames:
        try:
            pose, _ = odom.at(timestamp, player['max_odometry_gap_s'])
            # Reproduce dataset coverage filtering only; GT never enters matching.
            if gt is not None:
                gt.at(timestamp, player['max_ground_truth_gap_s'])
        except ValueError:
            continue
        path = Path(filename)
        if file_digest(path) != expected[str(path.relative_to(root))]:
            raise ValueError(f'Input changed: {path}')
        image = cv2.imread(filename)
        if image is None:
            raise ValueError(f'Unreadable image: {path}')
        vector = cache.extract(image)
        digest.update(vector.tobytes())
        features.append(vector)
        poses.append(pose)
    features, poses = np.asarray(features), np.asarray(poses)
    counts = [n.visual_sample_count for n in graph.nodes.values()]
    if sum(counts) != len(features):
        raise ValueError(f'Segmentation/frame count mismatch: {sum(counts)} != {len(features)}')
    ends = np.cumsum(counts)
    starts = np.r_[0, ends[:-1]]
    anchors = []
    for node, start, end in zip(graph.nodes.values(), starts, ends):
        matches = np.flatnonzero(np.all(np.isclose(features[start:end], node.view_features[0],
                                                  atol=1e-6, rtol=1e-6), axis=1)) + start
        matches = [int(i) for i in matches if np.allclose(poses[i], node.pose, atol=1e-6, rtol=1e-6)]
        if len(matches) != 1:
            raise ValueError(f'Cannot uniquely identify representative frame: {node.node_id}')
        anchors.append(matches[0])
    return features, poses[:, :2], np.asarray(anchors), starts, ends, digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('output/dinov3_revisits'))
    parser.add_argument('--dataset', type=Path, default=Path('../../encoder'))
    parser.add_argument('--cache', type=Path, default=Path('output/cache/descriptors'))
    parser.add_argument('--output', type=Path, default=Path('output/recovery_b'))
    parser.add_argument('--environments', nargs='+', choices=ENVIRONMENTS, default=ENVIRONMENTS)
    args = parser.parse_args()
    config = FrameConfig()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {'study': 'frame_recovery_b_v1', 'config': asdict(config),
                'source': str(args.source.resolve()),
                'runner_sha256': file_digest(__file__),
                'code': {Path(m.__file__).name: file_digest(m.__file__) for m in (
                    frame_recovery, closure_comparison, artifacts, descriptor_cache, directional,
                    local_recovery, mapper, matching, metrics, topo_graph, cold_odometry, cid_sims_data)},
                'packages': {**{p: version(p) for p in ('numpy', 'scipy')}, 'opencv': cv2.__version__},
                'tolerances_m': [.5, 1., 2., 3.],
                'policy': 'Preserve base accepted closures; causal trailing frames; fixed queries; no GT selection',
                'variants': ['current', 'fixed_frames', 'flexible_frames'],
                'inputs': {name: load_verified(args.source / name)[1] for name in args.environments}}
    fingerprint = identity(protocol)
    target = args.output / 'study_manifest.json'
    if target.exists() and json.loads(target.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Different study exists; choose another --output')
    atomic_json(target, {'fingerprint': fingerprint, **protocol})
    results = {}
    for name in args.environments:
        source, metadata = load_verified(args.source / name)
        features, xy, anchors, starts, ends, digest = load_frames(args.source / name, args.dataset,
                                                                args.cache, source)
        print(f'{name}: {len(features)} verified cached frames, {len(anchors)} fixed queries', flush=True)
        results[name] = {'frame_count': len(features), 'descriptor_sha256': digest, 'variants': {}}
        for variant in protocol['variants']:
            started = time.perf_counter()
            graph = replay(source, 'cosine', sequence_policy='self_only', recovery_mode='off',
                           visual_outlier_k=metadata['parameters']['visual_outlier_k'])
            diagnostics = []
            if variant != 'current':
                for event in graph.loop_events:
                    q = event.query_node_id
                    if event.accepted or q < 2:
                        continue
                    rows = recover(features, xy, int(anchors[q]), int(starts[q - 1]), config,
                                   flexible=variant == 'flexible_frames')
                    for row in rows:
                        row['query_node_id'] = q
                        if 'endpoint' in row:
                            row['matched_node_id'] = int(np.searchsorted(ends, row['endpoint'], side='right'))
                    endpoint = select_endpoint(rows)
                    if endpoint is not None:
                        matched = int(np.searchsorted(ends, endpoint, side='right'))
                        event.accepted, event.matched_node_id = True, matched
                        event.reason = 'accepted_' + variant
                        event.similarity = float(features[anchors[q]] @ features[endpoint])
                        graph.add_edge(q, matched, edge_type='loop')
                    diagnostics.extend(rows)
            graph.save(str(args.output / f'{name}_{variant}.pkl'))
            results[name]['variants'][variant] = {
                'association_seconds': time.perf_counter() - started,
                'metrics': [score(graph, source, radius) for radius in (.5, 1., 2., 3.)],
                'decisions': [asdict(e) for e in graph.loop_events], 'diagnostics': diagnostics}
        atomic_json(args.output / 'progress.json', results)
    totals = {variant: [aggregate([r['variants'][variant]['metrics'][i] for r in results.values()])
                       for i in range(4)] for variant in protocol['variants']}
    atomic_json(args.output / 'summary.json', {'protocol': protocol, 'fingerprint': fingerprint,
                                              'aggregate': totals, 'results': results})
    for variant, values in totals.items():
        print(variant, values[2], flush=True)


if __name__ == '__main__':
    main()
