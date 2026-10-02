"""Frozen-node screening of drift recovery A, with diagnostics relevant to B."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from vts_core import directional, local_recovery, mapper, matching, metrics, topo_graph
from vts_core.artifacts import atomic_json, file_digest, identity
from vts_evaluation import closure_comparison
from vts_evaluation.closure_comparison import aggregate, load_verified, replay, score
from vts_evaluation.summarize_experiments import ENVIRONMENTS

VARIANTS = {
    'current': {},
    'expanded_gate': {'geometric_gate_scale': 4.0},
    'visual_union': {'recovery_mode': 'visual_union'},
    'soft_odometry': {'recovery_mode': 'soft_odometry'},
    'local_recovery': {'recovery_mode': 'local'},
}
RADII = (0.5, 1.0, 2.0, 3.0)


def diagnosis(graph: topo_graph.TopoGraph, source: topo_graph.TopoGraph) -> dict:
    """Offline labels for diagnostic evidence only; never used by the verifier."""
    missed = set()
    for event in graph.loop_events:
        correct = {j for j in source.nodes if j < event.query_node_id - 1 and np.linalg.norm(
            np.asarray(source.nodes[j].gt_pose[:2]) - source.nodes[event.query_node_id].gt_pose[:2]
        ) <= 2.0}
        if correct and not (event.accepted and event.matched_node_id in correct):
            missed.add(event.query_node_id)
    rows = []
    for row in graph.recovery_diagnostics:
        if 'matched_node_id' not in row:
            continue
        q, m = row['query_node_id'], row['matched_node_id']
        if q in missed and np.linalg.norm(
            np.asarray(source.nodes[q].gt_pose[:2]) - source.nodes[m].gt_pose[:2]
        ) <= 2.0:
            rows.append(row)
    by_reason = {}
    for row in rows:
        by_reason.setdefault(row['reason'], set()).add(row['query_node_id'])
    visual_geometry_conflict = {r['query_node_id'] for r in rows
                               if r.get('visual_passed') and r['reason'] == 'local_geometry_mismatch'}
    return {
        'missed_queries_2m': sorted(missed),
        'correct_hypothesis_rejection_queries': {k: sorted(v) for k, v in by_reason.items()},
        'visual_support_but_rigid_alignment_fails': sorted(visual_geometry_conflict),
        'interpretation': 'A mismatch motivates investigating segmentation, but does not prove B: '
                          'local odometry error and wrong 2 m proximity labels are alternatives.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('output/dinov3_revisits'))
    parser.add_argument('--output', type=Path, default=Path('output/recovery_a'))
    parser.add_argument('--environments', nargs='+', choices=ENVIRONMENTS, default=ENVIRONMENTS)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--horizon', type=int, default=2)
    parser.add_argument('--residual-m', type=float, default=.75)
    parser.add_argument('--min-span-m', type=float, default=1.)
    parser.add_argument('--min-turn-deg', type=float, default=15.)
    parser.add_argument('--visual-z', type=float, default=2.)
    args = parser.parse_args()
    config = local_recovery.RecoveryConfig(args.top_k, args.horizon, args.residual_m,
                                          args.min_span_m, args.min_turn_deg, args.visual_z)
    inputs = {name: load_verified(args.source / name) for name in args.environments}
    parameters = {f'recovery_{k}': v for k, v in asdict(config).items()}
    parameters.update(visual_model='cosine', sequence_policy='self_only',
                      require_visual_verification=False)
    protocol = {
        'study': 'local_drift_recovery_a_v1', 'parameters': parameters, 'variants': VARIANTS,
        'source': str(args.source.resolve()),
        'inputs': {name: metadata for name, (_, metadata) in inputs.items()},
        'sources': {Path(module.__file__).name: file_digest(module.__file__) for module in
                    (directional, local_recovery, mapper, matching, metrics, topo_graph, closure_comparison)},
        'runner_sha256': file_digest(__file__), 'tolerances_m': RADII,
        'packages': {p: version(p) for p in ('numpy', 'scipy')},
        'scope': 'chronological node replay; no inference, ground truth in mapping or parameter tuning',
    }
    fingerprint = identity(protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / 'study_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Output contains a different study; choose another --output')
    atomic_json(manifest, {'fingerprint': fingerprint, **protocol})
    results = {}
    for name, (source, metadata) in inputs.items():
        if metadata['parameters']['gate_mode'] != 'both':
            raise ValueError('Expected full-method source segmentation')
        results[name] = {}
        for variant, settings in VARIANTS.items():
            started = time.perf_counter()
            graph = replay(source, visual_outlier_k=metadata['parameters']['visual_outlier_k'],
                           **parameters, **settings)
            duration = time.perf_counter() - started
            graph.save(str(args.output / f'{name}_{variant}.pkl'))
            results[name][variant] = {
                'metrics': [score(graph, source, radius) for radius in RADII],
                'association_seconds': duration,
                'reasons': dict(Counter(e.reason for e in graph.loop_events)),
                'decisions': [asdict(e) for e in graph.loop_events],
                'diagnostics': graph.recovery_diagnostics,
                'b_diagnosis': diagnosis(graph, source) if variant == 'local_recovery' else None,
                'confirmation_delays': [e.decision_node_id - e.query_node_id for e in graph.loop_events
                                        if e.reason == 'accepted_local_recovery'],
            }
        print(f'Compared five recovery policies: {name}', flush=True)
    totals = {variant: [
        {'tolerance_m': radius, **aggregate([values[variant]['metrics'][index]
                                           for values in results.values()])}
        for index, radius in enumerate(RADII)] for variant in VARIANTS}
    atomic_json(args.output / 'summary.json', {
        'fingerprint': fingerprint, 'protocol': protocol, 'aggregate': totals, 'results': results,
    })
    print('\n2 m: policy                 TP  FP  FN     P      R     F1')
    for variant, values in totals.items():
        row = values[2]
        print(f"{variant:26s} {row['tp']:3d} {row['fp']:3d} {row['fn']:3d} "
              f"{row['precision']:.3f}  {row['recall']:.3f}  {row['f1']:.3f}")
    print(f'\nResults: {args.output / "summary.json"}')


if __name__ == '__main__':
    main()
