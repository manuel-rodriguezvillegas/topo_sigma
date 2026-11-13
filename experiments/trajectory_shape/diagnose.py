"""Exploratory diagnostic (development data only): odometry trajectory-shape loop hypotheses.

A forward camera cannot confirm a revisit made in the opposite direction, but
the odometry path can. When the robot retraces its route, the path it just
drove (walked back from the query, by arc length) coincides with the path it
drove right after the candidate place:

- ``reverse``: compared forward in time from the candidate (opposed revisit);
- ``forward``: compared backward in time from the candidate (aligned revisit).

Both curves are resampled every 0.25 m over a window that grows until it holds
``--turn-deg`` of absolute heading change, up to ``--max-window`` metres and at
most half the arc length between the two places (the windows must not overlap).
With ``--turn-deg 0`` (default) the window is the whole fold up to ``--max-window``.
A rigid 2D fit (Kabsch) aligns them. The RMS residual measures shape agreement;
drift accumulated before the window only moves or rotates the whole curve.

The per-query statistics are stored, and a sweep simulates "baseline OR shape
hypothesis" decisions. Thresholds are chosen in-sample (optimistic) and also
transferred between COLD and CID-SIMS. Ground truth is used only for scoring.

    python diagnose.py --out OUT
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "vts_devcontainer" / "vts_ws" / "src" / "vts_core"))

from vts_core import evaluation as ev  # noqa: E402
from vts_core.association import (  # noqa: E402
    PLACE_SIGMA,
    gaussian_position_statistics,
    interval_position_covariance,
)
from vts_core.graph import TopoGraph  # noqa: E402
from vts_core.provenance import code_identity, json_safe, load_config, resolve  # noqa: E402
from vts_core.session import open_session, stored_sessions  # noqa: E402

STEP_M = 0.25
CHI2_95 = 5.991464547107979


def arc_track(poses: np.ndarray):
    """Cumulative arc length (strictly increasing) and unwrapped heading per frame."""
    steps = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    arc = arc + np.arange(len(arc)) * 1e-9  # stationary frames keep a strict order
    return arc, np.unwrap(poses[:, 2])


def sample(poses, arc, s_values):
    return np.column_stack([np.interp(s_values, arc, poses[:, 0]), np.interp(s_values, arc, poses[:, 1])])


def window_length(heading, arc, start, direction, turn_rad, max_window, limit):
    """Arc length from ``start`` (in ``direction``) until ``turn_rad`` of heading change is accumulated."""
    s = np.arange(0.0, min(max_window, limit) + 1e-9, STEP_M)
    if len(s) < 2:
        return 0.0, 0.0
    h = np.interp(start + direction * s, arc, heading)
    turned = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(h)))])
    if turn_rad <= 0:  # whole available window (the fold between both places, capped)
        return float(s[-1]), float(turned[-1])
    reached = np.flatnonzero(turned >= turn_rad)
    length = s[reached[0]] if len(reached) else s[-1]
    return float(length), float(turned[min(len(turned) - 1, int(round(length / STEP_M)))])


def kabsch_2d(source, target):
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - cs).T @ (target - ct))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, d]) @ u.T
    residual = source @ rotation.T + (ct - rotation @ cs) - target
    return float(np.sqrt((residual**2).sum(axis=1).mean())), float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))


def session_rows(config, name, turn_rad, max_window, min_window):
    graph = TopoGraph.load(resolve(config, "output_root") / "maps" / f"{name}__gate__dinov3.npz")
    places = graph.places[name]
    decisions = {d.query: d for d in graph.decisions[name]}
    session = open_session(config, name)
    poses, _ = session.odometry
    arc, heading = arc_track(poses)
    gt = ev.place_gt(places, session.gt)
    eligible_sets = ev.in_session_eligible(len(places))
    kinds = ev.revisit_kinds(gt, gt, 2.0, config["evaluation"]["aligned_deg"], eligible_sets)
    rows = []
    for q, place in enumerate(places):
        decision, eligible = decisions.get(q), eligible_sets[q]
        if decision is None:
            continue
        row = {"session": name, "query": q, "kind": kinds[q], "accepted": decision.accepted,
               "matched_gt_dist": None if decision.matched is None else
               float(np.linalg.norm(gt[decision.matched, :2] - gt[q, :2])), "candidates": []}
        rows.append(row)
        if not len(eligible):
            continue
        covariance = interval_position_covariance(place, [places[j] for j in eligible], PLACE_SIGMA, check=False)
        delta = place.pose[:2] - np.asarray([places[j].pose[:2] for j in eligible])
        d2, _ = gaussian_position_statistics(delta, covariance)
        aq = arc[place.medoid]
        for j in eligible:
            aj = arc[places[j].medoid]
            limit = 0.5 * (aq - aj)
            # The query window (walked back from q) fixes the length; the turn must be in it.
            length, turned = window_length(heading, arc, aq, -1, turn_rad, max_window, limit)
            if length < min_window:
                continue
            s = np.arange(0.0, length + 1e-9, STEP_M)
            recent = sample(poses, arc, aq - s)
            entry = {"j": int(j), "d2": float(d2[j]), "gt_dist": float(np.linalg.norm(gt[j, :2] - gt[q, :2])),
                     "window": length, "turned_deg": float(np.degrees(turned))}
            for hypothesis, direction in (("reverse", 1), ("forward", -1)):
                if direction < 0 and aj - length < 0:
                    continue
                residual, rotation = kabsch_2d(sample(poses, arc, aj + direction * s), recent)
                entry[hypothesis] = {"rms": residual, "rot_deg": rotation}
            row["candidates"].append(entry)
    return rows


def decide(row, hypothesis, rms_max, ratio_max, gate_scale, rot_max, min_turn_deg):
    """Best shape candidate if it is good, distinctive and plausible; else None."""
    scored = [(c[hypothesis]["rms"], c) for c in row["candidates"]
              if hypothesis in c and c["d2"] <= CHI2_95 * gate_scale and abs(c[hypothesis]["rot_deg"]) <= rot_max
              and c["turned_deg"] >= min_turn_deg]
    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    best_rms, best = scored[0]
    if best_rms > rms_max:
        return None
    # Distinctiveness: second best among candidates not adjacent to the best place.
    others = [r for r, c in scored[1:] if abs(c["j"] - best["j"]) > 1]
    if others and best_rms > ratio_max * others[0]:
        return None
    return best_rms, best


def count(rows, rule=None):
    tp = fp = fn = 0
    split = defaultdict(int)
    for row in rows:
        dist = row["matched_gt_dist"] if row["accepted"] else None
        if dist is None and rule is not None:
            chosen = [c for c in (decide(row, h, *rule) for h in ("reverse", "forward")) if c is not None]
            if chosen:
                dist = min(chosen, key=lambda item: item[0])[1]["gt_dist"]
        true = dist is not None and dist <= 2.0
        tp += true
        fp += dist is not None and not true
        if row["kind"] is not None:
            fn += not true
            split[f"{'tp' if true else 'fn'}_{row['kind']}"] += 1
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": f1, **split}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--turn-deg", type=float, default=0.0, help="0: use the whole fold")
    parser.add_argument("--max-window", type=float, default=30.0)
    parser.add_argument("--min-window", type=float, default=0.5)
    args = parser.parse_args()
    config = load_config()
    names = [n for n in stored_sessions(config)
             if (resolve(config, "output_root") / "maps" / f"{n}__gate__dinov3.npz").is_file()]
    rows = []
    for name in names:
        rows += session_rows(config, name, np.radians(args.turn_deg), args.max_window, args.min_window)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rows.json").write_text(json.dumps(json_safe(rows)))

    groups = {"all": rows, "cold": [r for r in rows if r["session"].startswith("cold")],
              "cid_sims": [r for r in rows if r["session"].startswith("cid")]}
    grid = list(itertools.product((0.1, 0.2, 0.3, 0.5, 0.8), (0.3, 0.5, 0.7, 1.0), (1.0, 4.0, 16.0, np.inf),
                                  (15.0, 30.0), (45.0, 90.0)))
    lines = ["# Hipótesis por forma de trayectoria (exploratorio, desarrollo)", "",
             f"{len(names)} sesiones, {len(rows)} consultas. Ventana: giro {args.turn_deg:g}°, "
             f"máx. {args.max_window:g} m, mín. {args.min_window:g} m. Política: baseline OR forma (2 m).", "",
             "| Regla (rms, ratio, escala puerta, rot. máx., giro mín.) | Selección | Evaluado | TP | FP | FN | P | R | F1 | "
             "TP opuestas | FN opuestas |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    report = {"code_sha256": code_identity(), "args": vars(args) | {"out": str(args.out)}, "results": []}
    for group in ("all", "cold", "cid_sims"):
        c = count(groups[group])
        report["results"].append({"rule": None, "evaluated": group, **c})
        lines.append(f"| baseline | — | {group} | {c['tp']} | {c['fp']} | {c['fn']} | {c['precision']:.3f} | "
                     f"{c['recall']:.3f} | {c['f1']:.3f} | {c.get('tp_opposed', 0)} | {c.get('fn_opposed', 0)} |")
    best = {g: max(grid, key=lambda rule: count(groups[g], rule)["f1"]) for g in ("all", "cold", "cid_sims")}
    for chosen, evaluated in (("all", "all"), ("cold", "cold"), ("cid_sims", "cid_sims"),
                              ("cold", "cid_sims"), ("cid_sims", "cold")):
        rule = best[chosen]
        c = count(groups[evaluated], rule)
        report["results"].append({"rule": rule, "selected_on": chosen, "evaluated": evaluated, **c})
        lines.append(f"| {', '.join(f'{v:g}' for v in rule)} | {chosen} | {evaluated} | {c['tp']} | {c['fp']} | "
                     f"{c['fn']} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | "
                     f"{c.get('tp_opposed', 0)} | {c.get('fn_opposed', 0)} |")
    # Precision-constrained operating point: most TP with no more than +5 FP over the baseline.
    base = count(groups["all"])
    safe = [r for r in grid if count(groups["all"], r)["fp"] <= base["fp"] + 5]
    if safe:
        rule = max(safe, key=lambda r: count(groups["all"], r)["tp"])
        c = count(groups["all"], rule)
        report["results"].append({"rule": rule, "selected_on": "all_fp_budget5", "evaluated": "all", **c})
        lines.append(f"| {', '.join(f'{v:g}' for v in rule)} | all, ≤ +5 FP | all | {c['tp']} | {c['fp']} | {c['fn']} | "
                     f"{c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | {c.get('tp_opposed', 0)} | "
                     f"{c.get('fn_opposed', 0)} |")
    (args.out / "summary.json").write_text(json.dumps(json_safe(report), indent=1))
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
