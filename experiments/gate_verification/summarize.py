"""Summary of diagnose.py outputs (exploratory, development data).

Sections:
1. Baseline (recorded gate decisions) and comparison with the Mac reference maps.
2. Retrieval ceiling: is a correct place among the top-K by appearance?
3. Verifier separability: AUROC of each score on top-K pairs, split aligned/opposed.
4. Policies "appearance proposes, geometry verifies": per-query simulation
   without drift reset. Thresholds are chosen in-sample (optimistic) and
   transferred across datasets (COLD -> CID-SIMS and back).
5. Relative pose: MASt3R versus odometry error on true revisits.

    python summarize.py OUT
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CHI2_95 = 5.991464547107979
FIELDS = ("lg_f_inliers", "lg_matches", "m_f_inliers", "m_matches")
GATE_SCALES = (1.0, 4.0, 16.0, np.inf)
ALIGNED_RAD = np.radians(45.0)
TOLERANCES = ("2.0", "1.0")


def load(out: Path):
    records, queries = [], {}
    for path in sorted((out / "records").glob("*.jsonl")):
        records += [json.loads(line) for line in path.read_text().splitlines() if line]
        queries[path.stem] = {int(q): row for q, row in json.loads((out / "queries" / f"{path.stem}.json").read_text()).items()}
    return records, queries


def score(record: dict, field: str) -> float:
    values = [v[field] for v in record["views"] if v.get(field) is not None]
    return float(max(values)) if values else np.nan


def auroc(positive: np.ndarray, values: np.ndarray) -> float | None:
    ok = np.isfinite(values)
    positive, values = positive[ok], values[ok]
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if not n1 or not n0:
        return None
    ranks = np.empty(len(values))
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks[order] = np.arange(1, len(values) + 1)
    for value in np.unique(sorted_values):  # average ranks of ties
        tie = values == value
        ranks[tie] = ranks[tie].mean()
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def counts(sessions, queries, by_query, tolerance, field=None, theta=None, scale=None, mode="baseline"):
    tp = fp = fn = 0
    split = defaultdict(int)
    t = float(tolerance)
    for name in sessions:
        for q, row in queries[name].items():
            opportunity = row["kind"][tolerance] is not None
            accepted_dist = None
            if mode in ("baseline", "union") and row["accepted"]:
                accepted_dist = row["matched_gt_dist"]
            if mode != "baseline" and accepted_dist is None:
                best = None
                for r in by_query[(name, q)]:
                    if "topk" not in r["roles"] or r["d2"] > CHI2_95 * scale:
                        continue
                    s = r["_score"][field]
                    if np.isfinite(s) and s >= theta and (best is None or s > best[0]):
                        best = (s, r["gt_dist"])
                if best is not None:
                    accepted_dist = best[1]
            true = accepted_dist is not None and accepted_dist <= t
            tp += int(true)
            fp += int(accepted_dist is not None and not true)
            fn += int(opportunity and not true)
            if opportunity:
                split[f"{'tp' if true else 'fn'}_{row['kind'][tolerance]}"] += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, **split}


def main() -> None:
    out = Path(sys.argv[1])
    records, queries = load(out)
    for r in records:
        r["_score"] = {f: score(r, f) for f in FIELDS}
    fields = [f for f in FIELDS if any(np.isfinite(r["_score"][f]) for r in records)]
    by_query = defaultdict(list)
    for r in records:
        by_query[(r["session"], r["query"])].append(r)
    sessions = sorted(queries)
    datasets = {name: ("cold" if name.startswith("cold") else "cid_sims") for name in sessions}
    groups = {"all": sessions, "cold": [s for s in sessions if datasets[s] == "cold"],
              "cid_sims": [s for s in sessions if datasets[s] == "cid_sims"]}
    report, lines = {"sessions": len(sessions), "pairs": len(records), "fields": fields}, []
    lines += ["# Verificación geométrica de candidatos (exploratorio, desarrollo)", "",
              f"{len(sessions)} sesiones, {sum(len(v) for v in queries.values())} consultas, {len(records)} pares.", ""]

    # 1. Baseline
    lines += ["## 1. Baseline (decisiones grabadas)", "", "| Grupo | Tol | TP | FP | FN | P | R | F1 | FN alin. | FN opuestas |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    report["baseline"] = {}
    for group, members in groups.items():
        for tolerance in TOLERANCES:
            c = counts(members, queries, by_query, tolerance)
            report["baseline"][f"{group}@{tolerance}"] = c
            lines.append(f"| {group} | {tolerance} | {c['tp']} | {c['fp']} | {c['fn']} | {c['precision']:.3f} | "
                         f"{c['recall']:.3f} | {c['f1']:.3f} | {c.get('fn_aligned', 0)} | {c.get('fn_opposed', 0)} |")
    reference = HERE / "reference_mac.json"
    if reference.is_file():
        ref = json.loads(reference.read_text())
        mine = {s: counts([s], queries, by_query, "2.0") for s in sessions}
        differ = [s for s in sessions if s in ref and any(mine[s][k] != ref[s][k] for k in ("tp", "fp", "fn"))]
        total = {k: sum(ref[s][k] for s in sessions if s in ref) for k in ("tp", "fp", "fn")}
        lines += ["", f"Referencia Mac (CPU, mismas sesiones) a 2 m: TP {total['tp']}, FP {total['fp']}, FN {total['fn']}. "
                  f"Sesiones con recuentos distintos: {len(differ)} de {len(sessions)}"
                  + (f" ({', '.join(differ[:8])}{'…' if len(differ) > 8 else ''})" if differ else "") + "."]
        report["mac_reference_differs"] = differ

    # 2. Retrieval ceiling
    lines += ["", "## 2. Techo de recuperación por apariencia (top-K sin puerta)", "",
              "| Grupo | Tol | Oportunidades | con correcto en top-K | FN baseline | FN con correcto en top-K | alin. | opuestas |",
              "|---|---|---|---|---|---|---|---|"]
    report["ceiling"] = {}
    for group, members in groups.items():
        for tolerance in TOLERANCES:
            t = float(tolerance)
            opp = hit = fn = fn_hit = 0
            kind_hit = defaultdict(int)
            for name in members:
                for q, row in queries[name].items():
                    if row["kind"][tolerance] is None:
                        continue
                    opp += 1
                    found = any("topk" in r["roles"] and r["gt_dist"] <= t for r in by_query[(name, q)])
                    hit += found
                    baseline_true = row["accepted"] and row["matched_gt_dist"] <= t
                    if not baseline_true:
                        fn += 1
                        fn_hit += found
                        kind_hit[row["kind"][tolerance]] += found
            report["ceiling"][f"{group}@{tolerance}"] = {"opportunities": opp, "topk_hit": hit, "fn": fn,
                                                         "fn_topk_hit": fn_hit, **kind_hit}
            lines.append(f"| {group} | {tolerance} | {opp} | {hit} | {fn} | {fn_hit} | {kind_hit['aligned']} | "
                         f"{kind_hit['opposed']} |")

    # 3. Separability
    lines += ["", "## 3. Separabilidad (AUROC, pares top-K, positivo = distancia GT ≤ 2 m)", "",
              "| Grupo | Puntuación | todos | alineados | opuestos | n+ alin. | n+ opuestos |", "|---|---|---|---|---|---|---|"]
    report["auroc"] = {}
    for group, members in groups.items():
        member_set = set(members)
        top = [r for r in records if "topk" in r["roles"] and r["session"] in member_set]
        positive = np.asarray([r["gt_dist"] <= 2.0 for r in top])
        aligned = np.asarray([r["gt_dyaw_abs"] < ALIGNED_RAD for r in top])
        for field in [*fields, "visual_sim"]:
            values = np.asarray([r["_score"][field] if field in r["_score"] else r[field] for r in top], dtype=float)
            # Aligned/opposed AUROC: positives of that kind against all negatives.
            row = {"all": auroc(positive, values),
                   "aligned": auroc(positive[aligned | ~positive], values[aligned | ~positive]),
                   "opposed": auroc(positive[~aligned | ~positive], values[~aligned | ~positive])}
            report["auroc"][f"{group}:{field}"] = row
            fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
            lines.append(f"| {group} | {field} | {fmt(row['all'])} | {fmt(row['aligned'])} | {fmt(row['opposed'])} | "
                         f"{int((positive & aligned).sum())} | {int((positive & ~aligned).sum())} |")

    # 4. Policies
    lines += ["", "## 4. Políticas «la apariencia propone, la geometría verifica» (2 m)", "",
              "θ elegido dentro de la muestra (optimista) y transferido entre datasets. "
              "`union`: se conserva la aceptación del baseline y el verificador solo añade.", "",
              "| Puntuación | Escala puerta | Modo | Selección θ | Evaluado en | θ | TP | FP | FN | P | R | F1 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    report["policies"] = []
    for field in fields:
        values = np.asarray([r["_score"][field] for r in records if "topk" in r["roles"]], dtype=float)
        grid = np.unique(np.quantile(values[np.isfinite(values)], np.linspace(0.5, 0.995, 30)))
        for scale in GATE_SCALES:
            for mode in ("replace", "union"):
                best_by_group = {}
                for group in ("all", "cold", "cid_sims"):
                    rows = [(theta, counts(groups[group], queries, by_query, "2.0", field, theta, scale, mode))
                            for theta in grid]
                    best_by_group[group] = max(rows, key=lambda item: item[1]["f1"])
                plan = [("all", "all", best_by_group["all"][0]), ("cold", "cid_sims", best_by_group["cold"][0]),
                        ("cid_sims", "cold", best_by_group["cid_sims"][0])]
                for chosen_on, evaluated_on, theta in plan:
                    c = counts(groups[evaluated_on], queries, by_query, "2.0", field, theta, scale, mode)
                    report["policies"].append({"field": field, "gate_scale": scale, "mode": mode, "theta_from": chosen_on,
                                               "evaluated_on": evaluated_on, "theta": float(theta), **c})
                    lines.append(f"| {field} | {scale:g} | {mode} | {chosen_on} | {evaluated_on} | {theta:g} | {c['tp']} | "
                                 f"{c['fp']} | {c['fn']} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} |")

    # 5. Relative pose
    lines += ["", "## 5. Pose relativa en revisitas verdaderas (vistas con distancia GT ≤ 2 m)", "",
              "Error absoluto mediano frente a GT. La odometría acumula la deriva entre ambos pasos.", "",
              "| Grupo | n | MASt3R distancia (m) | odometría distancia (m) | MASt3R |giro| (°) | odometría |giro| (°) |",
              "|---|---|---|---|---|---|"]
    report["pose"] = {}
    for group, members in groups.items():
        member_set = set(members)
        views = [v for r in records if r["session"] in member_set for v in r["views"]
                 if v.get("m_dist") is not None and v["gt_dist"] <= 2.0]
        if not views:
            continue
        err = lambda key, ref: np.asarray([abs(v[key] - v[ref]) for v in views])  # noqa: E731
        row = {"n": len(views), "m_dist": float(np.median(err("m_dist", "gt_dist"))),
               "odom_dist": float(np.median(err("odom_dist", "gt_dist"))),
               "m_yaw_deg": float(np.degrees(np.median(err("m_yaw_abs", "gt_dyaw_abs")))),
               "odom_yaw_deg": float(np.degrees(np.median(err("odom_dyaw_abs", "gt_dyaw_abs"))))}
        report["pose"][group] = row
        lines.append(f"| {group} | {row['n']} | {row['m_dist']:.2f} | {row['odom_dist']:.2f} | {row['m_yaw_deg']:.1f} | "
                     f"{row['odom_yaw_deg']:.1f} |")
    seconds = [v["m_seconds"] for r in records for v in r["views"] if v.get("m_seconds")]
    if seconds:
        lines += ["", f"MASt3R: {np.median(seconds):.3f} s por par de imágenes (mediana, ambos sentidos)."]
    errors = sum(1 for r in records for v in r["views"] if v.get("m_error"))
    lines += [f"Errores de MASt3R: {errors}."] if errors else []

    (out / "summary.json").write_text(json.dumps(report, indent=1, default=float))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    main()
