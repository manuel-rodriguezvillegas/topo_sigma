"""Likelihood ratio of visual node scores, fitted on the design graphs (development and validation) of the geometric runs.

Positive pairs are earlier nodes within the place radius and seen with a ground-truth heading
within 45 degrees; negative pairs are earlier nodes farther than the place radius. Opposed
revisits are left out because the mapper treats their appearance as uninformative. A logistic
regression of the class on the score gives the log ratio after removing the class prior.
Each design traversal gets the fit that left it out; test traversals use the fit on all of them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from mapper_runs import PLACE_RADIUS_M, VISUAL_FILE, VISUAL_SCORES, geometric_source
from stream import DESIGN
from vts_core.mapper import TopologicalMapper
from vts_core.motion import normalize_angle
from vts_core.topo_graph import TopoGraph

ALIGNED_HEADING = np.pi / 4
SCORES = {"vmf": TopologicalMapper._node_vmf_evidence, "cosine": TopologicalMapper._node_similarity}


def labelled_scores(run: Path, score: str) -> tuple[np.ndarray, np.ndarray]:
    graph = TopoGraph.load(str(run / "graph.pkl"))
    truth = {int(k): np.asarray(v) for k, v in json.loads((run / "node_gt.json").read_text()).items()}
    ids = sorted(graph.nodes)
    values, labels = [], []
    for position, query in enumerate(ids):
        for earlier in ids[: max(0, position - graph.loop_exclusion_nodes)]:
            distance = np.hypot(*(truth[query][:2] - truth[earlier][:2]))
            aligned = abs(normalize_angle(truth[query][2] - truth[earlier][2])) < ALIGNED_HEADING
            if distance <= PLACE_RADIUS_M and not aligned:
                continue
            values.append(SCORES[score](graph.nodes[query], graph.nodes[earlier]))
            labels.append(distance <= PLACE_RADIUS_M)
    return np.asarray(values, dtype=np.float64), np.asarray(labels, dtype=bool)


def fit_log_ratio(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    center, scale = float(values.mean()), float(values.std() or 1.0)
    x = (values - center) / scale

    def loss(theta: np.ndarray) -> float:
        logit = theta[0] + theta[1] * x
        return float(np.mean(np.logaddexp(0.0, logit) - labels * logit))

    a, b = minimize(loss, np.zeros(2), method="BFGS").x
    prior = np.log(labels.mean() / (1.0 - labels.mean()))
    return {"intercept": float(a - prior - b * center / scale), "slope": float(b / scale),
            "positives": int(labels.sum()), "negatives": int((~labels).sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--encoders", nargs="*", default=list(VISUAL_SCORES))
    args = parser.parse_args()
    report = json.loads(VISUAL_FILE.read_text()) if VISUAL_FILE.exists() else {}
    for encoder in args.encoders:
        for score in VISUAL_SCORES[encoder]:
            data = {s.name: labelled_scores(args.out / "replay" / s.name / geometric_source(encoder), score)
                    for s in DESIGN}

            def pooled(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
                return tuple(np.concatenate([data[n][i] for n in names]) for i in (0, 1))

            names = list(data)
            report[f"{encoder}:{score}"] = {
                "all": fit_log_ratio(*pooled(names)),
                "leave_out": {n: fit_log_ratio(*pooled([m for m in names if m != n])) for n in names},
            }
            print(encoder, score, report[f"{encoder}:{score}"]["all"])
    VISUAL_FILE.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
