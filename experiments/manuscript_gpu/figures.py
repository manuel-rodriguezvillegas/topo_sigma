"""Manuscript figures drawn from the development replays: lambda_2 trace, maps and text retrieval examples.

Maps are drawn at their printed size without a legend; ``map_legend.tex`` is the shared TikZ legend,
drawn with the same colors and typeset in the document font (``\\input`` it below the subfigures).
"""

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
from vts_evaluation.map_viz import (LOOP_EDGE_COLOR, SEQUENTIAL_EDGE_COLOR, TRAJECTORY_COLOR, MapStyle, render_map,
                                   room_color)

HISTORICAL_SRC = HERE.parent / "anyloc_published" / "historical_src"
TEXT_WIDTH_IN = 522.0 / 72.27
MAP_WIDTH_IN = 0.245 * TEXT_WIDTH_IN
PAPER_MAP_STYLE = MapStyle(
    size_in=(MAP_WIDTH_IN, MAP_WIDTH_IN), node_area=9.0, node_edge_width=0.25, sequential_width=0.5,
    loop_width=0.8, trajectory_width=0.5, legend=False,
    rc={"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True, "font.size": 7, "axes.linewidth": 0.4,
        "xtick.major.width": 0.4, "ytick.major.width": 0.4, "xtick.major.size": 2, "ytick.major.size": 2},
)
NO_ROOM_LABEL = "No room label"
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


def _tikz_color(hex_color: str) -> str:
    red, green, blue = (int(hex_color[k:k + 2], 16) for k in (1, 3, 5))
    return f"{{rgb,255:red,{red};green,{green};blue,{blue}}}"


def _line_entry(color: str, dashed: bool, text: str) -> str:
    dash = ",dashed" if dashed else ""
    return (rf"\tikz[baseline=-0.6ex]\draw[draw={_tikz_color(color)},line width=0.8pt{dash}] (0,0) -- (1.4em,0);"
            rf"\,{text}")


def map_legend(labels: set[str | None]) -> str:
    """TikZ legend for the maps: rooms present in ``labels`` (``None`` for unlabelled nodes) and edge types."""
    rooms = [
        rf"\tikz[baseline=-0.6ex]\node[circle,draw=black,line width=0.25pt,fill={_tikz_color(room_color(label))},"
        rf"inner sep=0pt,minimum size=4.5pt] {{}};\,{label or NO_ROOM_LABEL}"
        for label in sorted(labels, key=lambda label: (label is None, label or ""))
    ]
    lines = [_line_entry(TRAJECTORY_COLOR, False, "Ground-truth trajectory"),
             _line_entry(SEQUENTIAL_EDGE_COLOR, False, "Sequential edge"),
             _line_entry(LOOP_EDGE_COLOR, True, "Loop closure")]
    entries = [rf"\mbox{{{entry}}}" for entry in rooms + lines]
    return "\\par\\smallskip{\\centering\\footnotesize\n" + "\\quad\n".join(entries) + "\\par}\n"


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
    labels: set[str | None] = set()
    for seq in DEV:
        run = args.out / "replay" / seq.name / "topo"
        graph = TopoGraph.load(str(run / "graph.pkl"))
        node_gt = {int(k): np.asarray(v[:2]) for k, v in json.loads((run / "node_gt.json").read_text()).items()}
        render_map(graph, str(figures / f"map_{seq.name}.pdf"), gt_xy=load_ground_truth_xy(str(seq.trajectory_gt)),
                   node_gt=node_gt, style=PAPER_MAP_STYLE)
        labels |= {node.room_label for node in graph.nodes.values()}
    (figures / "map_legend.tex").write_text(map_legend(labels))
    retrieval_examples(args.out, figures)


if __name__ == "__main__":
    main()
