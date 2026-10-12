"""Stage-wise development study of cross-session association (plan B, step 4).

Every stage is measured against the appearance-only baseline (mean-direction
cosine, top-1, margin as confidence) on all ordered development pairs:

E0  baseline: appearance top-1, confidence = margin over the second.
E1  calibrated appearance posterior with a new-place hypothesis (no geometry).
E2  anchoring: SE(2) hypotheses from appearance top-k (reported, not a decision rule).
E3  posterior with appearance and, after the anchor, geometry under the local transform.

Appearance calibration variants: ``supervised`` (ground truth of the *other*
environments: aligned matches within 1 m vs pairs beyond 3 m) and
``label_free`` (map A's own in-session pairs that odometry confirms with a
shared view vs pairs it rules out). Ground truth of the evaluated pair is used
only for scoring. Development data only.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from vts_core import evaluation as ev
from vts_core.association import calibration_samples, robust_gaussian
from vts_core.multisession import AppearanceModel, CrossConfig, CrossSessionAssociator, fit_se2, similarity_matrix


def supervised_samples(pairs_data: list[dict], kind: str) -> dict[str, dict[str, list[float]]]:
    """Per environment: GT-labelled score samples (aligned within 1 m vs beyond 3 m)."""
    out = defaultdict(lambda: {"same": [], "diff": []})
    for d in pairs_data:
        scores = similarity_matrix(d["query_places"], d["map_places"], kind)
        qg, mg = d["query_gt"], d["map_gt"]
        distance = np.linalg.norm(qg[:, None, :2] - mg[None, :, :2], axis=2)
        heading = np.abs((mg[None, :, 2] - qg[:, None, 2] + np.pi) % (2 * np.pi) - np.pi)
        out[d["environment"]]["same"] += scores[(distance <= 1.0) & (heading < np.radians(45))].tolist()
        out[d["environment"]]["diff"] += scores[distance > 3.0].tolist()
    return out


def supervised_model(samples: dict, environment: str) -> AppearanceModel:
    same = [s for env, v in samples.items() if env != environment for s in v["same"]]
    diff = [s for env, v in samples.items() if env != environment for s in v["diff"]]
    return AppearanceModel(robust_gaussian(same), robust_gaussian(diff))


def label_free_model(map_places) -> AppearanceModel | None:
    calibration = calibration_samples(map_places)
    if len(calibration.view) < 3 or len(calibration.diff) < 10:
        return None
    return AppearanceModel(robust_gaussian(calibration.view), robust_gaussian(calibration.diff))


def odometry_to_gt(places, gt) -> tuple[np.ndarray, np.ndarray]:
    return fit_se2(np.asarray([p.pose[:2] for p in places]), gt[:, :2])


def evaluate_pair(d: dict, model: AppearanceModel, config: CrossConfig, tolerance: float, aligned_deg: float) -> dict:
    qg, mg = d["query_gt"], d["map_gt"]
    distance = np.linalg.norm(qg[:, None, :2] - mg[None, :, :2], axis=2)
    positive = distance.min(axis=1) <= tolerance
    kinds = np.asarray(ev.revisit_kinds(qg, mg, tolerance, aligned_deg), dtype=object)
    scores = similarity_matrix(d["query_places"], d["map_places"])
    out = {}

    def record(name, matched, confidence):
        correct = distance[np.arange(len(matched)), matched] <= tolerance
        curve = ev.pr_curve(correct, positive, np.asarray(confidence))
        for kind in ("aligned", "opposed"):
            mask = kinds == kind
            curve[f"r1_{kind}"] = float(correct[mask].mean()) if mask.any() else None
            curve[f"n_{kind}"] = int(mask.sum())
        out[name] = curve

    top, margin = ev.top1_statistic(scores, "margin")
    record("E0_margin", top, margin)
    for name, use_geometry in (("E1_appearance_posterior", False), ("E3_posterior_geometry", True)):
        associator = CrossSessionAssociator(d["map_places"], model, CrossConfig(**{**vars(config),
                                                                                 "use_geometry": use_geometry}))
        decisions = [associator.observe(p) for p in d["query_places"]]
        record(name, np.asarray([x.matched for x in decisions]), [x.posterior for x in decisions])
        if use_geometry:
            anchor = associator.tracker.anchor
            info = {"anchored": anchor is not None, "anchor_query": associator.anchor_query,
                    "queries": len(d["query_places"])}
            if anchor is not None:
                support_ok = [distance[i, j] <= 2.0 for i, j in anchor.support]
                info["support_correct"] = float(np.mean(support_ok))
                # Localization error: B places mapped to A's odometry frame by the anchor,
                # then to the GT frame by A's odometry->GT fit, against B's own GT.
                ra, ta = odometry_to_gt(d["map_places"], mg)
                later = range(associator.anchor_query, len(d["query_places"]))
                errors = [np.linalg.norm(((d["query_places"][i].pose[:2] @ anchor.rotation.T + anchor.translation)
                                          @ ra.T + ta) - qg[i, :2]) for i in later]
                info["error_median_m"] = float(np.median(errors)) if errors else None
                info["error_next10_m"] = float(np.median(errors[:10])) if errors else None
            out["anchor"] = info
    return out


def pair_data(config: dict, pairs: list[tuple[str, str]], loader) -> list[dict]:
    """Places and ground truth of every session in ``pairs`` (``loader(name) -> (session, places)``)."""
    cache = {}
    for name in sorted({n for pair in pairs for n in pair}):
        session, places = loader(name)
        cache[name] = (session.recording.environment, places, ev.place_gt(places, session.gt))
    return [{"map": a, "query": b, "environment": cache[a][0], "map_places": cache[a][1], "map_gt": cache[a][2],
             "query_places": cache[b][1], "query_gt": cache[b][2]} for a, b in pairs]


def _run(task) -> dict:
    data, model, config, tolerance, aligned_deg = task
    return evaluate_pair(data, model, config, tolerance, aligned_deg)


def environment_equal_interval(differences: list[float], environments: list[str], queries: list[str],
                               resamples: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of per-environment means; stratified bootstrap of query sessions within each environment."""
    by_env: dict = defaultdict(lambda: defaultdict(list))
    for value, environment, query in zip(differences, environments, queries, strict=True):
        by_env[environment][query].append(value)

    def statistic(groups: dict) -> float:
        return float(np.mean([np.mean([v for q in sessions for v in q]) for sessions in groups.values()]))

    point = statistic({e: list(sessions.values()) for e, sessions in by_env.items()})
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(resamples):
        sample = {}
        for environment, sessions in by_env.items():
            keys = list(sessions)
            sample[environment] = [sessions[keys[k]] for k in rng.integers(0, len(keys), len(keys))]
        boots.append(statistic(sample))
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def save_model(model: AppearanceModel, path) -> None:
    from vts_core.provenance import atomic_json

    atomic_json(path, {"same": model.same, "diff": model.diff})


def load_model(path) -> AppearanceModel:
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text())
    return AppearanceModel(tuple(data["same"]), tuple(data["diff"]))


def study(data: list[dict], config: CrossConfig, tolerance: float, aligned_deg: float, calibration: str,
          pool_map=map, resamples: int = 10000, frozen: AppearanceModel | None = None) -> dict:
    """Run every stage on every pair; summary with bootstrap intervals over pairs.

    ``calibration``: "supervised" (leave-one-environment-out on ``data``), "frozen"
    (``frozen`` model, e.g. fitted on all development data for a test) or "self".
    """
    samples = supervised_samples(data, "mean_cosine") if calibration == "supervised" else None

    def model_for(d):
        if calibration == "frozen":
            return frozen
        return supervised_model(samples, d["environment"]) if samples else None

    tasks = [(d, model_for(d), config, tolerance, aligned_deg) for d in data]
    rows = list(pool_map(_run, tasks))
    summary = {}
    for method in ("E0_margin", "E1_appearance_posterior", "E3_posterior_geometry"):
        ap = [r[method]["average_precision"] for r in rows]
        summary[method] = {
            "ap": ev.bootstrap_interval(ap, resamples=resamples),
            "recall_at_p90": float(np.mean([r[method]["recall_at_p90"] for r in rows])),
            **{f"r1_{k}": float(np.mean([r[method][f"r1_{k}"] for r in rows if r[method][f"r1_{k}"] is not None]))
               for k in ("aligned", "opposed")}}
    summary["E3_minus_E0_ap"] = ev.bootstrap_interval(
        [r["E3_posterior_geometry"]["average_precision"] - r["E0_margin"]["average_precision"] for r in rows],
        resamples=resamples)
    for metric in ("average_precision", "recall_at_p95"):
        key = "ap" if metric == "average_precision" else "r_at_p95"
        summary[f"E3_minus_E0_{key}_environment_equal"] = environment_equal_interval(
            [r["E3_posterior_geometry"][metric] - r["E0_margin"][metric] for r in rows],
            [d["environment"] for d in data], [d["query"] for d in data], resamples)
    anchors = [r["anchor"] for r in rows if r["anchor"]["anchored"]]
    summary["anchor"] = {"anchored": len(anchors), "pairs": len(rows),
                         "median_anchor_query": float(np.median([a["anchor_query"] for a in anchors])) if anchors else None,
                         "bad_anchors": sum(a["support_correct"] < 0.5 for a in anchors),
                         "median_error_next10_m": float(np.median([a["error_next10_m"] for a in anchors])) if anchors else None}
    return {"summary": summary, "pairs": {f"{d['map']}:{d['query']}": r for d, r in zip(data, rows)}}
