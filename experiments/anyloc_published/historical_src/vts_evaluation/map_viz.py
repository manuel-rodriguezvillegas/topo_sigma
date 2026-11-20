"""Topological-map visualization for the offline evaluator.

Renders the generated map to a PNG (plus a ``.pgf`` twin for LaTeX) so map
quality is inspectable at a glance:

- Nodes are coloured by room class (abbreviated class labels in the legend;
  the report gives the abbreviation-to-full-name key). Floorplan overlays
  rigidly align the estimated node layout to per-node ground truth with one
  best-fit SE(2) transform. The relative distortion of the estimated map is
  therefore preserved and remains visible in the figure.
- Sequential traversal edges and loop closures use distinct line styles.
- The GT trajectory is drawn underneath for spatial context.

Every figure is additionally exported as ``.pgf`` (when the local TeX
toolchain allows) so it can be ``\\input`` directly in LaTeX/Overleaf, where
fonts and text sizes follow the document instead of being baked into a PNG.

matplotlib is imported lazily and the renderer degrades gracefully (returns
False with a warning) if it is unavailable, so it never breaks the metrics run.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from vts_core.topo_graph import TopoGraph

_LEGEND_FONTSIZE: int = 16
_LEGEND_TITLE_FONTSIZE: int = 18
_LEGEND_MARKERSIZE: int = 11
_LEGEND_COLUMNS: int = 4

ROOM_COLORS: dict[str, str] = {
    "1PO": "#1f77b4", "2PO": "#ff7f0e", "CR": "#2ca02c", "CNR": "#9467bd", "KT": "#8c564b",
    "LO": "#e377c2", "PA": "#bcbd22", "RL": "#17becf", "ST": "#aec7e8", "TL": "#ffbb78",
    "TR": "#98df8a", "LAB": "#c5b0d5",
}
UNLABELLED_COLOR: str = "#8da0b5"
TRAJECTORY_COLOR: str = "#d1d1d1"
SEQUENTIAL_EDGE_COLOR: str = "#737373"
LOOP_EDGE_COLOR: str = "#dc143c"


@dataclass(frozen=True)
class MapStyle:
    """Figure size in inches, node marker area in points squared and line widths in points."""

    size_in: tuple[float, float]
    node_area: float
    node_edge_width: float
    sequential_width: float
    loop_width: float
    trajectory_width: float
    legend: bool
    rc: dict = field(default_factory=dict)


REPORT_STYLE = MapStyle(size_in=(11.0, 8.0), node_area=90.0, node_edge_width=0.5, sequential_width=1.1,
                        loop_width=2.0, trajectory_width=1.0, legend=True)


def room_color(label: str | None) -> str:
    return ROOM_COLORS.get(label, UNLABELLED_COLOR) if label else UNLABELLED_COLOR


def _save_latex_figure(fig, out_path: str) -> str | None:
    """Save a LaTeX-editable twin of the figure next to ``out_path``.

    Preferred format is ``.pgf`` (LaTeX source: ``\\input`` it in Overleaf and
    all text is typeset in the document's font, so size and placement stay
    editable after export), but writing PGF needs a local TeX toolchain for
    text metrics. Without one, fall back to ``.svg``: with Overleaf's ``svg``
    package (``\\usepackage{svg}`` + ``\\includesvg{...}``) the text layer is
    likewise re-typeset by LaTeX in the document's font.
    """
    pgf_path: str = os.path.splitext(out_path)[0] + ".pgf"
    try:
        fig.savefig(pgf_path, bbox_inches="tight")
        print(f"[viz] LaTeX (PGF) version saved to {pgf_path}")
        return pgf_path
    except Exception:  # no local TeX; SVG keeps the text editable in Overleaf
        svg_path: str = os.path.splitext(out_path)[0] + ".svg"
        try:
            fig.savefig(svg_path, bbox_inches="tight")
        except Exception as error:  # pragma: no cover - environment dependent
            print(f"[viz] LaTeX-editable export skipped ({error})")
            return None
        print(
            f"[viz] LaTeX-editable (SVG) version saved to {svg_path} "
            "(use \\usepackage{svg} + \\includesvg in Overleaf)"
        )
        return svg_path


def render_map(
    graph: TopoGraph,
    out_path: str,
    gt_xy: np.ndarray | None = None,
    node_gt: dict[int, np.ndarray] | None = None,
    title: str | None = None,
    style: MapStyle = REPORT_STYLE,
) -> bool:
    """Render the map to ``out_path`` and export an editable twin. Returns ``True`` on success.

    When ``node_gt`` covers every node, nodes are painted at their ground-truth coordinates
    (paint-time only: the mapper placed them from odometry); otherwise at the mapper's positions.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - environment dependent
        print(f"[viz] matplotlib unavailable, skipping map image: {error}")
        return False

    ids: list[int] = sorted(graph.nodes)
    if not ids:
        print("[viz] empty graph, nothing to draw")
        return False
    index: dict[int, int] = {nid: k for k, nid in enumerate(ids)}
    map_pos: np.ndarray = graph.positions()

    gt_pos: np.ndarray | None = None
    if node_gt is not None and all(i in node_gt for i in ids):
        gt_pos = np.array([node_gt[i] for i in ids], dtype=np.float64)
    drawn: np.ndarray = gt_pos if gt_pos is not None else map_pos

    with plt.rc_context(style.rc):
        fig, ax = plt.subplots(figsize=style.size_in)
        if gt_xy is not None and gt_xy.size:
            ax.plot(gt_xy[:, 0], gt_xy[:, 1], color=TRAJECTORY_COLOR, lw=style.trajectory_width, zorder=1)

        for id_a, id_b in graph.edges():
            ka, kb = index[id_a], index[id_b]
            is_loop = graph.edge_types.get((id_a, id_b), "odometry") == "loop"
            ax.plot(
                [drawn[ka, 0], drawn[kb, 0]],
                [drawn[ka, 1], drawn[kb, 1]],
                color=LOOP_EDGE_COLOR if is_loop else SEQUENTIAL_EDGE_COLOR,
                linestyle="--" if is_loop else "-",
                lw=style.loop_width if is_loop else style.sequential_width,
                zorder=4 if is_loop else 2,
            )

        labels: list[str | None] = [graph.nodes[i].room_label for i in ids]
        ax.scatter(drawn[:, 0], drawn[:, 1], c=[room_color(label) for label in labels],
                   s=style.node_area, edgecolors="black", linewidths=style.node_edge_width, zorder=5)

        if style.legend:
            ax.legend(
                handles=_legend_handles(plt, labels), loc="upper center", bbox_to_anchor=(0.5, -0.1),
                fontsize=_LEGEND_FONTSIZE, title="Room" if any(labels) else None,
                title_fontsize=_LEGEND_TITLE_FONTSIZE, framealpha=0.9, ncol=_LEGEND_COLUMNS,
            )
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if title:
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight" if style.legend else None)
        _save_latex_figure(fig, out_path)
    plt.close(fig)
    print(f"[viz] map image saved to {out_path}")
    return True


def _legend_handles(plt, labels: list[str | None]) -> list:
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=room_color(label),
                   markeredgecolor="black", markersize=_LEGEND_MARKERSIZE, label=label or "Node")
        for label in sorted(set(labels), key=lambda label: label or "")
    ]
    return handles + [
        plt.Line2D([0], [0], color=SEQUENTIAL_EDGE_COLOR, lw=2, label="Sequential edge"),
        plt.Line2D([0], [0], color=LOOP_EDGE_COLOR, lw=2, linestyle="--", label="Loop closure"),
    ]


def render_confusion(
    confusion: dict[str, dict[str, int]], out_path: str, title: str | None = None
) -> bool:
    """Render a row-normalized retrieval confusion matrix (true vs top-1)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover
        print(f"[viz] matplotlib unavailable, skipping confusion matrix: {error}")
        return False

    true_labels: list[str] = sorted(confusion)
    pred_labels: list[str] = sorted(
        {p for row in confusion.values() for p in row} | set(true_labels)
    )
    matrix: np.ndarray = np.zeros((len(true_labels), len(pred_labels)), dtype=float)
    for i, t in enumerate(true_labels):
        total = sum(confusion[t].values()) or 1
        for j, p in enumerate(pred_labels):
            matrix[i, j] = confusion[t].get(p, 0) / total

    fig, ax = plt.subplots(
        figsize=(max(6, len(pred_labels)), max(5, len(true_labels)))
    )
    im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(pred_labels)), pred_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(true_labels)), true_labels)
    ax.set_xlabel("predicted (top-1) room")
    ax.set_ylabel("true room")
    ax.set_title(title or "Retrieval confusion (row-normalized)")
    for i in range(len(true_labels)):
        for j in range(len(pred_labels)):
            if matrix[i, j] > 0:
                ax.text(
                    j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] > 0.5 else "black", fontsize=8,
                )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] confusion matrix saved to {out_path}")
    return True


def render_rejection_curve(
    curve: list[dict[str, float]], out_path: str, title: str | None = None
) -> bool:
    """Plot the coverage-vs-precision curve of margin-based rejection."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover
        print(f"[viz] matplotlib unavailable, skipping rejection curve: {error}")
        return False
    if not curve:
        return False
    coverage = [p["coverage"] for p in curve]
    precision = [p["precision_at_1"] for p in curve]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverage, precision, "-o", color="tab:blue", markersize=3)
    ax.set_xlabel("coverage (fraction of queries answered)")
    ax.set_ylabel("precision@1 on answered queries")
    ax.set_title(title or "Calibrated rejection: coverage vs precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(min(0.5, min(precision) - 0.05), 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] rejection curve saved to {out_path}")
    return True


def _load_calibration(floorplan_path: str) -> dict[str, float] | None:
    """Load the ``<floorplan>.calib.json`` world-metres -> pixel affine."""
    stem: str = os.path.splitext(floorplan_path)[0]
    calib_path: str = stem + ".calib.json"
    if not os.path.exists(calib_path):
        print(f"[viz] no calibration sidecar at {calib_path}; skipping overlay")
        return None
    with open(calib_path) as f:
        calib: dict = json.load(f)
    return calib


def render_on_floorplan(
    graph: TopoGraph,
    floorplan_path: str,
    out_path: str,
    node_gt: dict[int, np.ndarray],
    gt_xy: np.ndarray | None = None,
    title: str | None = None,
) -> bool:
    """Overlay the map on a metric floorplan image.

    Estimated node poses are rigidly aligned to their ground-truth
    counterparts with one best-fit SE(2) transform before being projected
    onto the floorplan. This removes only the arbitrary global frame; scale
    and internal map distortion remain visible. Per-node ground truth and a
    ``<floorplan>.calib.json`` sidecar are therefore required. Degrades
    gracefully (returns False) if matplotlib, the floorplan, the calibration,
    or the GT are unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except Exception as error:  # pragma: no cover - environment dependent
        print(f"[viz] matplotlib/PIL unavailable, skipping floorplan: {error}")
        return False

    if not os.path.exists(floorplan_path):
        print(f"[viz] floorplan not found: {floorplan_path}")
        return False
    calib: dict[str, float] | None = _load_calibration(floorplan_path)
    if calib is None:
        return False
    ids: list[int] = sorted(graph.nodes)
    if not ids or not all(i in node_gt for i in ids):
        print("[viz] floorplan overlay needs per-node ground truth; skipping")
        return False

    from vts_evaluation.floorplan_transform import build_transform

    try:
        transform = build_transform(calib)
    except (KeyError, ValueError) as error:
        print(f"[viz] bad floorplan calibration: {error}")
        return False

    # Align the mapper's odometry-frame layout to the floorplan frame without
    # changing scale or internal geometry. Unlike painting nodes at GT, this
    # keeps drift, stretched paths, and wall crossings visible.
    gt_pos: np.ndarray = np.array([node_gt[i] for i in ids], dtype=np.float64)
    from vts_core.matching import fit_se2

    fit = fit_se2(graph.positions(), gt_pos)
    if fit is None:
        print("[viz] floorplan overlay needs at least two nodes; skipping")
        return False
    rotation, translation = fit
    drawn_pos: np.ndarray = graph.positions() @ rotation.T + translation

    floor = np.array(Image.open(floorplan_path).convert("RGB"))
    height, width = floor.shape[:2]
    fig, ax = plt.subplots(figsize=(9, 13))
    ax.imshow(floor)

    if gt_xy is not None and gt_xy.size:
        gx, gy = transform(gt_xy[:, 0], gt_xy[:, 1])
        ax.plot(
            gx, gy, color="tab:green", lw=1.2, alpha=0.7,
            label="Ground-truth trajectory",
        )

    cols, rows = transform(drawn_pos[:, 0], drawn_pos[:, 1])
    index: dict[int, int] = {nid: k for k, nid in enumerate(ids)}
    for id_a, id_b in graph.edges():
        ka, kb = index[id_a], index[id_b]
        edge_type = graph.edge_types.get((id_a, id_b), "odometry")
        is_loop = edge_type == "loop"
        ax.plot(
            [cols[ka], cols[kb]], [rows[ka], rows[kb]],
            color=LOOP_EDGE_COLOR if is_loop else SEQUENTIAL_EDGE_COLOR,
            linestyle="--" if is_loop else "-",
            lw=2.0 if is_loop else 1.1,
            zorder=4 if is_loop else 2,
        )

    labels: list[str | None] = [graph.nodes[i].room_label for i in ids]
    ax.scatter(
        cols, rows, c=[room_color(label) for label in labels],
        s=85, edgecolors="black", linewidths=0.5, zorder=5,
    )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.legend(
        handles=_legend_handles(plt, labels), loc="lower left", fontsize=_LEGEND_FONTSIZE,
        title="Room" if any(labels) else None,
        title_fontsize=_LEGEND_TITLE_FONTSIZE,
    )
    if title:
        ax.set_title(title)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    _save_latex_figure(fig, out_path)
    plt.close(fig)
    print(f"[viz] floorplan overlay saved to {out_path}")
    return True
