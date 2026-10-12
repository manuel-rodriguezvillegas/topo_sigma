# TopoSIGMA — workspace packages

ROS 2 Jazzy, Ubuntu 24.04, CPython 3.12. Third-party Python dependencies are
locked by uv in `../../pyproject.toml` and `../../uv.lock`.

## `vts_core`: the engine (pure Python, no ROS)

| Module | Responsibility |
|---|---|
| `datasets` | COLD and CID-SIMS adapters (frame selection, raw odometry, ground truth), catalogue and development/test roles |
| `session` | Canonical per-session store: `frames.npz`, `<encoder>.npy`, `manifest.json` with SHA-256 hashes; legacy C1 import |
| `descriptors` | Frozen encoders (DINOv3; MegaLoc and AnyLoc-GeM as baselines), batched extraction |
| `odometry` | Additive uncertainty budget (`path_length_budget_v1`), interpolation, SE(2) Jacobian propagation |
| `segmentation` | Adaptive valleys of windowed algebraic connectivity (batched offline, streaming online, identical) |
| `places` | Segment summary: mean direction, three views with their poses, medoid pose and budget |
| `association` | In-session association: chi-square gate baseline and posterior with a new-place hypothesis |
| `multisession` | Cross-session scores, distance-aligned sequences, SE(2) anchor hypotheses |
| `graph` | Multi-session `TopoGraph` (odometry, loop and merge edges), `.npz` storage without pickle |
| `mapping` | `OnlineMapper` (frame by frame) and `map_session` (offline): the one engine used everywhere |
| `evaluation` | Opportunities, aligned/opposed revisit types, loop counts, PR curves, bootstrap intervals |
| `provenance` | Configuration, hashes, atomic outputs, run records |
| `cli` | `toposigma frames|map|associate|evaluate` |

Configuration: `vts_core/config/toposigma.yaml`. Ground truth enters only
`evaluation` (and the stores, which keep it apart from mapping inputs).

## ROS packages (thin wrappers)

- `vts_players`: `session_player` replays catalogue sessions (`sessions:=a,b`) through `vts_core.datasets`.
- `vts_mapping`: `graph_builder` encodes frames with DINOv3 and pushes them to `OnlineMapper`; writes
  `graph_<i>.npz`, `graph_<i>_views/` and `graph_<i>.json` per sequence.
- `vts_language`: `commands` ranks places of a stored graph with CLIP; destinations need confirmation.
- `vts_bringup`: `pipeline.launch.py` (`mode:=building|command`).

## Topic contract

| Topic | Type | Notes |
|---|---|---|
| `/camera/image` | sensor_msgs/Image (bgr8) | |
| `/odom` | nav_msgs/Odometry | cumulative budget in the (x, y, yaw) covariance blocks |
| `/ground_truth_pose` | geometry_msgs/PoseStamped | optional, evaluation only |
| `/dataset/sequence_done` | std_msgs/String (JSON) | end-of-run signal with `frame_count` |
| `/mapping/frame_processed` | std_msgs/String | acknowledgement `sec:nanosec` (flow control) |
