"""Evaluate B's actual matched frames, separately from node representative poses."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from vts_core.artifacts import atomic_json, file_digest
from vts_evaluation.closure_comparison import aggregate, load_verified
from vts_evaluation.frame_recovery import select_endpoint
from vts_players.cid_sims_data import load_ground_truth, load_wheel_odometry, timestamped_color_images
from vts_players.cold_odometry import load_cold_odometry


def frame_ground_truth(manifest: dict, dataset: Path) -> np.ndarray:
    """Reconstruct the evaluated frame order, using the original coverage rules."""
    player = manifest['parameters']['player']
    root = dataset / Path(manifest['inputs']['root']).relative_to('/workspace/encoder')
    if 'frame_stride' in player:
        odometry = load_wheel_odometry(str(root / player['odometry_file']))
        ground_truth = load_ground_truth(str(root / player['ground_truth_file']))
        frames = timestamped_color_images(str(root / player['color_subdir']))[::player['frame_stride']]
        path = root / player['ground_truth_file']
        if file_digest(path) != manifest['inputs']['files'][player['ground_truth_file']]:
            raise ValueError('Ground truth changed since the source run')
    else:
        odometry = load_cold_odometry(str(root / player['odometry_subpath']))
        ground_truth = None
        frames = sorted((float(p.name.split('_')[0][1:]), str(p))
                        for p in (root / player['images_subdir']).glob('*.jpeg'))
    positions = []
    for timestamp, filename in frames:
        try:
            odometry.at(timestamp, player['max_odometry_gap_s'])
            xy = (ground_truth.at(timestamp, player['max_ground_truth_gap_s'])[0][:2]
                  if ground_truth is not None else
                  [float(re.search('_x([^_]+)', filename)[1]),
                   float(re.search('_y([^_]+)', filename)[1])])
        except ValueError:
            continue
        positions.append(xy)
    return np.asarray(positions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=Path('output/recovery_b'))
    parser.add_argument('--source', type=Path, default=Path('output/dinov3_revisits'))
    parser.add_argument('--dataset', type=Path, default=Path('../../encoder'))
    args = parser.parse_args()
    summary_path = args.study / 'summary.json'
    summary = json.loads(summary_path.read_text())
    metrics = {v: [] for v in summary['aggregate']}
    results = {}
    for name, result in summary['results'].items():
        graph, metadata = load_verified(args.source / name)
        if metadata != summary['protocol']['inputs'][name]:
            raise ValueError('Source differs from the B study')
        manifest = json.loads((args.source / name / 'run_manifest.json').read_text())
        positions = frame_ground_truth(manifest, args.dataset)
        if len(positions) != result['frame_count']:
            raise ValueError('Frame order/count differs from the B study')
        ends = np.cumsum([n.visual_sample_count for n in graph.nodes.values()])
        starts = np.r_[0, ends[:-1]]
        results[name] = {}
        for variant, values in result['variants'].items():
            tp = fp = fn = 0
            new_closures = []
            for event in values['decisions']:
                q = event['query_node_id']
                query_xy = np.asarray(graph.nodes[q].gt_pose[:2])
                limit = starts[q - 1] if q >= 2 else 0
                opportunity = bool(limit and np.any(np.linalg.norm(positions[:limit] - query_xy, axis=1) <= 2))
                correct = False
                if event['accepted']:
                    if event['reason'] == 'accepted_' + variant:
                        rows = [r for r in values['diagnostics'] if r['query_node_id'] == q]
                        endpoint = select_endpoint(rows)
                        if endpoint is None:
                            raise ValueError('Accepted closure lacks an unambiguous frame endpoint')
                        distance = float(np.linalg.norm(positions[endpoint] - query_xy))
                        new_closures.append({'query': q, 'node': event['matched_node_id'],
                                             'frame': endpoint, 'distance_m': distance})
                    else:
                        distance = float(np.linalg.norm(np.asarray(
                            graph.nodes[event['matched_node_id']].gt_pose[:2]) - query_xy))
                    correct = distance <= 2
                    tp += correct
                    fp += not correct
                fn += opportunity and not correct
            row = {'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}
            metrics[variant].append(row)
            results[name][variant] = {'metrics': row, 'new_closures': new_closures}
    totals = {v: aggregate(rows) for v, rows in metrics.items()}
    atomic_json(args.study / 'frame_endpoint_audit.json', {
        'summary_sha256': file_digest(summary_path), 'audit_sha256': file_digest(__file__),
        'evaluation': 'Same queries; 2m GT to matched frame (baseline uses representative). '
                      'Opportunities include all historical frames before the previous node. '
                      'This has a different recall denominator from the node evaluation.',
        'results': results, 'aggregate': totals})
    print(totals)


if __name__ == '__main__':
    main()
