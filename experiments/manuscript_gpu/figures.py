"""Manuscript figures drawn from the development replays: lambda_2 trace, maps and text retrieval examples."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from stream import DEV, HERE, RESULTS
from vts_core.topo_graph import TopoGraph
from vts_evaluation.ground_truth import load_ground_truth_xy
from vts_evaluation.map_viz import render_map

HISTORICAL_SRC = HERE.parent / "anyloc_published" / "historical_src"
RETRIEVAL_EXAMPLES = [
    ("freiburg_a", "Take me to the kitchen", "KT"),
    ("freiburg_a", "Go to the large office", "LO"),
    ("freiburg_ext", "Take me to the stairs", "ST"),
    ("saarbruecken_a", "Take me to the terminal room", "TR"),
]


def retrieval_examples(out: Path, figures: Path) -> None:
    """Best-matching view of the top-ranked node for each example request."""
    from vts_core.retrieval import PlaceRetriever, SemanticEncoder

    encoder = SemanticEncoder()
    summary = []
    for index, (sequence, request, label) in enumerate(RETRIEVAL_EXAMPLES, start=1):
        graph = TopoGraph.load(str(out / "replay" / sequence / "topo" / "graph.pkl"))
        ranked, _ = PlaceRetriever(encoder, graph).query(request, top_k=1)
        node = ranked[0][0]
        view = int(np.argmax(node.view_embeddings @ encoder.embed_text(request)))
        path = figures / f"retrieval_{index}_{sequence}.png"
        cv2.imwrite(str(path), node.views[view])
        summary.append({"sequence": sequence, "request": request, "expected": label,
                        "top1": node.room_label, "correct": node.room_label == label, "image": path.name})
    (figures / "retrieval_examples.json").write_text(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    figures = RESULTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "vts_evaluation.plot_lambda2", "--lambda2",
         str(args.out / "replay" / "freiburg_a" / "topo" / "lambda2.npy"), "--valley-k", "2.0",
         "--out", str(figures / "lambda2_freiburg_a.pdf")],
        check=True, cwd=HISTORICAL_SRC,
    )
    for seq in DEV:
        run = args.out / "replay" / seq.name / "topo"
        node_gt = {int(k): np.asarray(v[:2]) for k, v in json.loads((run / "node_gt.json").read_text()).items()}
        render_map(TopoGraph.load(str(run / "graph.pkl")), str(figures / f"map_{seq.name}.pdf"),
                   gt_xy=load_ground_truth_xy(str(seq.trajectory_gt)), node_gt=node_gt)
    retrieval_examples(args.out, figures)


if __name__ == "__main__":
    main()
