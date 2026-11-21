"""Video of the mapper on design traversals: the camera, lambda_2 and the topological map as it is built.

The map is drawn at reference positions, like the paper maps. Each traversal is replayed with the
paper's method and must reproduce the nodes and loop closures recorded in
``$OUT/replay/<sequence>/<variant>/result.json`` before anything is drawn. Each profile writes
``<videos>/<profile>/<sequence>.mp4`` (H.264) and, as it requires, an animated ``.webp`` or a ``.png``
poster with the final map.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from PIL import Image  # noqa: E402

from figures import MAP_VARIANT  # noqa: E402
from mapper_runs import Replay, prepare_replay, replay_frames, run_inputs, visual_variants  # noqa: E402
from stream import ALL, DEV, Stream, load_stream  # noqa: E402
from vts_core.topo_graph import TopoGraph  # noqa: E402
from vts_evaluation.map_viz import LOOP_EDGE_COLOR, SEQUENTIAL_EDGE_COLOR, room_color  # noqa: E402

CAMERA_RATE_HZ = 5.0
FIGURE_SIZE_IN = (12.8, 7.2)
PREVIEW_WIDTH_PX = 720
PREVIEW_FPS = 8.0
HIGHLIGHT_VIDEO_S = 0.8
LOOP_HIGHLIGHT_FACTOR = 3
HOLD_VIDEO_S = 3.0
LAMBDA2_HISTORY_S = 60.0
TRAJECTORY_COLOR = "#9a9a9a"
ROBOT_COLOR = "#111111"
ROOM_NAMES = {
    "1PO": "One person office", "2PO": "Two person office", "CR": "Corridor", "CNR": "Conference room",
    "KT": "Kitchen", "LO": "Large office", "PA": "Printer area", "RL": "Robotics lab", "ST": "Stairs",
    "TL": "Toilet", "TR": "Terminal room",
}
TITLES = {
    "freiburg_a": "COLD Freiburg A", "freiburg_ext": "COLD Freiburg Extended", "saarbruecken_a": "COLD Saarbrücken A",
    "saarbruecken_ext": "COLD Saarbrücken Extended", "cid_sims_apartment1_1": "CID-SIMS Apartment 1",
    "cid_sims_apartment2_1": "CID-SIMS Apartment 2", "cid_sims_apartment3_1": "CID-SIMS Apartment 3",
}


@dataclass(frozen=True)
class Profile:
    """Video settings for one destination. ``speed`` is relative to the recording."""

    width_px: int
    fps: int
    speed: int
    crf: int
    webp: bool
    poster: bool

    def __post_init__(self) -> None:
        if (self.speed * CAMERA_RATE_HZ) % self.fps:
            raise ValueError("speed must advance a whole number of camera frames per video frame")

    @property
    def frame_step(self) -> int:
        return int(self.speed * CAMERA_RATE_HZ) // self.fps

    @property
    def size_px(self) -> tuple[int, int]:
        return self.width_px, round(self.width_px * FIGURE_SIZE_IN[1] / FIGURE_SIZE_IN[0])


PROFILES = {
    "readme": Profile(width_px=1280, fps=30, speed=12, crf=28, webp=True, poster=False),
    "journal": Profile(width_px=1920, fps=30, speed=6, crf=20, webp=False, poster=True),
    "web": Profile(width_px=1280, fps=30, speed=18, crf=28, webp=False, poster=True),
}


@dataclass(frozen=True)
class Timeline:
    """When each node was emitted, the last frame of each node closed at a lambda_2 valley, and the lambda_2
    samples, all indexed by camera frame."""

    node_frame: dict[int, int]
    boundary_frame: dict[int, int]
    lambda2_frames: np.ndarray
    lambda2: np.ndarray


def record(replay: Replay) -> Timeline:
    graph, monitor = replay.mapper.graph, replay.mapper._monitor
    node_frame: dict[int, int] = {}
    boundary_frame: dict[int, int] = {}
    lambda2_frames: list[int] = []
    for k in replay_frames(replay, images=False):
        for node_id in graph.nodes.keys() - node_frame.keys():
            node_frame[node_id] = k
            boundary_frame[node_id] = replay.mapper._pending_frames[0].frame_index - 1
        if len(monitor.eigenvalues) > len(lambda2_frames):
            lambda2_frames.append(k)
    last = len(replay.stream.image_paths) - 1
    for node_id in graph.nodes.keys() - node_frame.keys():
        node_frame[node_id] = last
    return Timeline(node_frame, boundary_frame, np.asarray(lambda2_frames), np.asarray(monitor.eigenvalues))


def check_reproduction(seq_name: str, graph: TopoGraph, inputs: dict, recorded: dict) -> None:
    loops = [[e.query_node_id, e.matched_node_id] for e in graph.loop_events if e.accepted]
    if inputs != recorded["inputs"] or len(graph.nodes) != recorded["nodes"] or loops != recorded["loops"]:
        raise SystemExit(f"{seq_name}: the replay does not reproduce its recorded result "
                         f"({len(graph.nodes)} nodes and {len(loops)} loops against "
                         f"{recorded['nodes']} and {len(recorded['loops'])}, inputs equal: {inputs == recorded['inputs']})")


class MapAnimation:
    """One matplotlib figure whose artists are updated frame by frame."""

    def __init__(self, title: str, profile: Profile, graph: TopoGraph, timeline: Timeline,
                 reference: np.ndarray) -> None:
        self.graph, self.timeline, self.reference = graph, timeline, reference
        self.highlight_frames = round(HIGHLIGHT_VIDEO_S * profile.fps * profile.frame_step)
        self.ids = sorted(graph.nodes)
        self.xy = {i: np.asarray(graph.nodes[i].gt_pose[:2]) for i in self.ids}
        self.color = {i: room_color(graph.nodes[i].room_label) for i in self.ids}
        self.edges = [(a, b, graph.edge_types[(a, b)] == "loop") for a, b in graph.edges()]
        self.edge_frame = {(a, b): max(timeline.node_frame[a], timeline.node_frame[b]) for a, b, _ in self.edges}

        self.fig = plt.figure(figsize=FIGURE_SIZE_IN, dpi=profile.width_px / FIGURE_SIZE_IN[0], facecolor="white")
        self.fig.text(0.02, 0.955, "S2PRO2", fontsize=20, weight="bold", va="center")
        self.fig.text(0.135, 0.955, f"{title}  ·  {profile.speed}× speed", fontsize=13,
                      va="center", color="#444444")
        self.counters = self.fig.text(0.98, 0.955, "", fontsize=13, ha="right", va="center")
        self._camera_axes()
        self._lambda2_axes()
        self._map_axes()

    def _camera_axes(self) -> None:
        ax = self.fig.add_axes((0.02, 0.33, 0.47, 0.57))
        ax.set_axis_off()
        self.camera = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        self.clock = ax.text(0.015, 0.975, "", transform=ax.transAxes, color="white", fontsize=11, va="top",
                             bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=3))
        ax.set_title("Robot camera", fontsize=12, loc="left")

    def _lambda2_axes(self) -> None:
        ax = self.fig.add_axes((0.06, 0.07, 0.42, 0.19))
        low, high = self.timeline.lambda2.min(), np.percentile(self.timeline.lambda2, 99.5)
        ax.set_ylim(low - 0.05 * (high - low), high + 0.05 * (high - low))
        ax.set_xlim(-LAMBDA2_HISTORY_S, 0)
        ax.set_xlabel("time before now [s]", fontsize=9)
        ax.set_title(r"Spectral connectivity $\lambda_2$ of the recent frames", fontsize=11, loc="left")
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        self.lambda2_axes = ax
        (self.lambda2_line,) = ax.plot([], [], color="#1f4e8c", lw=1.4)
        self.lambda2_nodes = ax.scatter([], [], s=36, edgecolors="black", linewidths=0.5, zorder=3)

    def _map_axes(self) -> None:
        ax = self.fig.add_axes((0.53, 0.14, 0.45, 0.76))
        lo, hi = self.reference.min(axis=0), self.reference.max(axis=0)
        centre, half = (lo + hi) / 2, (hi - lo).max() / 2 + 1.5
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)
        ax.set_xlabel("x [m]", fontsize=9)
        ax.set_ylabel("y [m]", fontsize=9)
        ax.grid(alpha=0.25)
        self.map_axes = ax
        (self.trajectory,) = ax.plot([], [], color=TRAJECTORY_COLOR, lw=1.2, zorder=1)
        self.sequential = LineCollection([], colors=SEQUENTIAL_EDGE_COLOR, linewidths=1.3, zorder=2)
        self.loops = LineCollection([], colors=LOOP_EDGE_COLOR, linewidths=3.0, linestyles="--", zorder=4)
        ax.add_collection(self.sequential)
        ax.add_collection(self.loops)
        self.nodes = ax.scatter([], [], s=70, edgecolors="black", linewidths=0.6, zorder=5)
        self.halo = ax.scatter([], [], s=420, facecolors="none", edgecolors="#f2b705", linewidths=2.5, zorder=6)
        self.loop_halo = ax.scatter([], [], s=420, facecolors="none", edgecolors=LOOP_EDGE_COLOR, linewidths=2.5,
                                    zorder=6)
        (self.robot,) = ax.plot([], [], marker=(3, 0, 0), markersize=13, color=ROBOT_COLOR, zorder=7)
        self.event = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=12, va="top", color=LOOP_EDGE_COLOR,
                             weight="bold")
        self._legend()

    def _legend(self) -> None:
        labels = sorted({self.graph.nodes[i].room_label for i in self.ids} - {None})
        handles = [plt.Line2D([0], [0], marker="o", linestyle="none", markersize=7, markerfacecolor=room_color(lab),
                              markeredgecolor="black", markeredgewidth=0.5, label=ROOM_NAMES.get(lab, lab))
                   for lab in labels]
        if not labels:
            handles.append(plt.Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                                      markerfacecolor=room_color(None), markeredgecolor="black", label="Node"))
        handles += [plt.Line2D([0], [0], color=TRAJECTORY_COLOR, lw=1.5, label="Reference trajectory"),
                    plt.Line2D([0], [0], color=SEQUENTIAL_EDGE_COLOR, lw=1.5, label="Sequential edge"),
                    plt.Line2D([0], [0], color=LOOP_EDGE_COLOR, lw=2, linestyle="--", label="Loop closure")]
        self.fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.755, 0.0), ncol=4, fontsize=8,
                        frameon=False, columnspacing=1.0, handletextpad=0.3)

    def draw(self, k: int, image_path: str, pose: np.ndarray) -> np.ndarray:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.camera.set_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.camera.set_extent((0, image.shape[1], image.shape[0], 0))
        seconds = k / CAMERA_RATE_HZ
        self.clock.set_text(f"{int(seconds // 60):02d}:{int(seconds % 60):02d}")

        frame_of = self.timeline.node_frame
        present = [i for i in self.ids if frame_of[i] <= k]
        recent = [i for i in present if k - frame_of[i] < self.highlight_frames]
        self.nodes.set_offsets(np.array([self.xy[i] for i in present]).reshape(-1, 2))
        self.nodes.set_facecolors([self.color[i] for i in present])
        self.halo.set_offsets(np.array([self.xy[i] for i in recent]).reshape(-1, 2))
        visible = [(a, b, is_loop) for a, b, is_loop in self.edges if self.edge_frame[(a, b)] <= k]
        self.sequential.set_segments([(self.xy[a], self.xy[b]) for a, b, is_loop in visible if not is_loop])
        self.loops.set_segments([(self.xy[a], self.xy[b]) for a, b, is_loop in visible if is_loop])
        new_loops = [(a, b) for a, b, is_loop in visible
                     if is_loop and k - self.edge_frame[(a, b)] < LOOP_HIGHLIGHT_FACTOR * self.highlight_frames]
        self.loop_halo.set_offsets(np.array([self.xy[i] for edge in new_loops for i in edge]).reshape(-1, 2))
        self.event.set_text("Loop closure accepted" if new_loops else "")
        n_loops = sum(is_loop for *_, is_loop in visible)
        self.counters.set_text(f"nodes {len(present)}   ·   loop closures {n_loops}")

        self.trajectory.set_data(self.reference[:k + 1, 0], self.reference[:k + 1, 1])
        self.robot.set_data([pose[0]], [pose[1]])
        self.robot.set_marker((3, 0, np.degrees(pose[2]) - 90.0))

        history = LAMBDA2_HISTORY_S * CAMERA_RATE_HZ
        shown = (self.timeline.lambda2_frames <= k) & (self.timeline.lambda2_frames > k - history)
        times = (self.timeline.lambda2_frames[shown] - k) / CAMERA_RATE_HZ
        self.lambda2_line.set_data(times, self.timeline.lambda2[shown])
        boundary_of = self.timeline.boundary_frame
        closed = [i for i in present if i in boundary_of and k - boundary_of[i] < history]
        self.lambda2_nodes.set_offsets(np.array([[(boundary_of[i] - k) / CAMERA_RATE_HZ,
                                                  self._lambda2_at(boundary_of[i])] for i in closed]).reshape(-1, 2))
        self.lambda2_nodes.set_facecolors([self.color[i] for i in closed])

        self.fig.canvas.draw()
        return np.ascontiguousarray(np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3])

    def _lambda2_at(self, frame: int) -> float:
        index = np.searchsorted(self.timeline.lambda2_frames, frame, side="right") - 1
        return float(self.timeline.lambda2[max(index, 0)])


def write_profile(name: str, profile: Profile, seq_name: str, stream: Stream, graph: TopoGraph,
                  timeline: Timeline, videos: Path) -> None:
    animation = MapAnimation(TITLES.get(seq_name, seq_name), profile, graph, timeline, stream.ground_truth[:, :2])
    last = len(stream.image_paths) - 1
    frames = list(range(0, last + 1, profile.frame_step))
    if frames[-1] != last:
        frames.append(last)
    hold = round(HOLD_VIDEO_S * profile.fps)
    preview_every = max(1, round(profile.fps / PREVIEW_FPS))
    preview_size = (PREVIEW_WIDTH_PX, round(PREVIEW_WIDTH_PX * FIGURE_SIZE_IN[1] / FIGURE_SIZE_IN[0]))
    preview: list[Image.Image] = []

    target = videos / name
    target.mkdir(parents=True, exist_ok=True)
    mp4 = target / f"{seq_name}.mp4"
    writer = imageio_ffmpeg.write_frames(
        str(mp4), profile.size_px, fps=profile.fps, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        macro_block_size=8,
        output_params=["-crf", str(profile.crf), "-preset", "slow", "-movflags", "+faststart"],
        ffmpeg_log_level="error")
    writer.send(None)
    for n, k in enumerate(frames):
        rgb = animation.draw(k, stream.image_paths[k], stream.ground_truth[k])
        for _ in range(hold if k == last else 1):
            writer.send(rgb.tobytes())
        if profile.webp and (n % preview_every == 0 or k == last):
            preview.append(Image.fromarray(cv2.resize(rgb, preview_size, interpolation=cv2.INTER_AREA)))
    writer.close()
    plt.close(animation.fig)

    written = [mp4]
    if profile.webp:
        webp = target / f"{seq_name}.webp"
        frame_ms = round(1000 * preview_every / profile.fps)
        preview[0].save(webp, save_all=True, append_images=preview[1:], loop=0, quality=55, method=4,
                        duration=[frame_ms] * (len(preview) - 1) + [round(1000 * HOLD_VIDEO_S)])
        written.append(webp)
    if profile.poster:
        poster = target / f"{seq_name}.png"
        Image.fromarray(rgb).save(poster)
        written.append(poster)
    sizes = "  ".join(f"{path.suffix[1:]} {path.stat().st_size / 1e6:4.1f} MB" for path in written)
    print(f"  {name:8s} {profile.speed:2d}x  {(len(frames) + hold - 1) / profile.fps:5.1f} s  {sizes}", flush=True)


def render(seq_name: str, out: Path, descriptors: Path, videos: Path, profiles: list[str]) -> None:
    seq = ALL[seq_name]
    spec = visual_variants("dev")[MAP_VARIANT]
    stream = load_stream(seq)
    inputs = run_inputs(seq, spec, stream)
    replay = prepare_replay(seq, spec, inputs, stream, descriptors)
    timeline = record(replay)
    graph = replay.mapper.graph
    recorded = json.loads((out / "replay" / seq_name / MAP_VARIANT / "result.json").read_text())
    check_reproduction(seq_name, graph, inputs, recorded)
    replay.mapper.close()
    n_loops = sum(1 for e in graph.loop_events if e.accepted)
    print(f"{seq_name}: {len(graph.nodes)} nodes, {n_loops} loop closures, reproduced", flush=True)
    for name in profiles:
        write_profile(name, PROFILES[name], seq_name, stream, graph, timeline, videos)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output of run_all.sh, with the recorded replays")
    parser.add_argument("--descriptors", type=Path, required=True, help="AnyLoc runner output directory")
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--sequences", nargs="*", default=[seq.name for seq in DEV])
    parser.add_argument("--profiles", nargs="*", choices=sorted(PROFILES), default=list(PROFILES))
    args = parser.parse_args()
    for seq_name in args.sequences:
        render(seq_name, args.out, args.descriptors, args.videos, args.profiles)


if __name__ == "__main__":
    main()
