# Multi-session mapping environment

This directory holds the ongoing multi-session work on top of S2PRO2: a pure-Python engine
(`vts_core`) with the `toposigma` command line, and ROS 2 packages that wrap it. It is a later
rewrite and does not produce the numbers of the paper; those come from
[`experiments/`](../experiments) (see the [main README](../README.md)).

## Layout

```text
vts_devcontainer/
├── .devcontainer/              Ubuntu 24.04 / ROS 2 Jazzy environment
├── pyproject.toml, uv.lock     Locked Python dependencies
└── vts_ws/src/
    ├── vts_core/               Pure-Python engine and the `toposigma` CLI (no ROS)
    ├── vts_players/            ROS dataset player over the vts_core adapters
    ├── vts_mapping/            ROS graph builder: thin wrapper of the engine
    ├── vts_language/           Optional SigLIP2 text-to-place suggestions
    └── vts_bringup/            Launch file
```

See [`vts_ws/src/README.md`](vts_ws/src/README.md) for the module map and the ROS interface.

## Requirements

The supported environment is the included development container:

- Docker Desktop or Docker Engine;
- a development-container client such as Visual Studio Code with the
  Dev Containers extension;
- enough disk space for the selected datasets and model cache.

The image uses Ubuntu 24.04, ROS 2 Jazzy, CPython 3.12, and dependencies locked
with uv. ROS 2 Jazzy's binary Python extensions require the Ubuntu 24.04 Python
ABI, so the container intentionally does not install a separate Python 3.13
interpreter.

## Dataset setup

The datasets are not redistributed. Download them under `encoder/`, which is
mounted at `/workspace/encoder` inside the container.

### COLD

Download the sequences from the
[COsy Localization Database](https://www.cas.kth.se/COLD/) into
`encoder/seq_data/<recording>/` (e.g. `cold-freiburg_part_a_seq2_night1/`).
Every recording found there enters the catalogue.

Each extracted traversal must retain the original `std_cam/`, `odom_scans/`,
and `localization/` directories.

### CID-SIMS

Download CID-SIMS from
[Science Data Bank](https://doi.org/10.57760/sciencedb.ai.00003) and extract
each traversal as follows:

```text
encoder/
├── apartment1_1/
│   ├── color/
│   ├── groundtruth.txt
│   └── odom.txt
├── apartment2_1/
└── apartment3_1/
```

## Build

Open `vts_devcontainer/` in the development container, then run:

```bash
cd /workspaces/topo_sigma/vts_devcontainer/vts_ws
colcon build --symlink-install --base-paths src   # output/ holds historical package copies
source install/setup.bash
```

The offline tools need only NumPy, SciPy and PyYAML (plus PyTorch and
Transformers for descriptor extraction); they also run outside ROS with
`PYTHONPATH=src/vts_core python3 -m vts_core.cli ...`.

## Offline workflow (`toposigma`)

Every recording is a *session* with a canonical store under
`output/sessions/<session>/` (frames, raw odometry, ground truth, descriptors
and a manifest of SHA-256 hashes). Session names follow the recordings, e.g.
`cold_freiburg_a_seq2_night1` or `cid_apartment1_2`. Recordings reserved as
final test data (`test_sessions` in `vts_core/config/toposigma.yaml`) are
refused unless `--allow-test` is given, which requires a written protocol.

```bash
toposigma frames list --check            # catalogue, roles, stores, odometry quality
toposigma frames build --all-missing     # DINOv3 stores for every clean development recording
toposigma map --all --jobs 6             # in-session baseline on every stored session
toposigma evaluate loops                 # loop P/R/F1, split into aligned / opposed revisits
toposigma evaluate revisits              # cross-session revisit types per session pair
toposigma associate --kind mean_cosine --jobs 6
toposigma evaluate cross --label mean_cosine__dinov3
```

Configuration lives in one YAML file, `vts_core/config/toposigma.yaml`;
`--config my.yaml` is deep-merged over it. Outputs go to `output/toposigma/`,
each with a run record (code hash, parameters, input manifests). Outside the
container, point `TOPOSIGMA_DATA_ROOT` at the local `encoder/` directory.

## ROS pipeline

```bash
ros2 launch vts_bringup pipeline.launch.py sessions:=cold_freiburg_a_seq2_night1
ros2 launch vts_bringup pipeline.launch.py mode:=command \
  graph:=output/ros/cold_freiburg_a_seq2_night1/graph_0.npz query:="Go to the kitchen."
```

The player replays a recording through the same adapters as the stores; the
graph builder feeds the same engine as the offline replays. A robot or another
dataset must publish:

| Topic | Message | Purpose |
|---|---|---|
| `/camera/image` | `sensor_msgs/Image` | BGR camera frame |
| `/odom` | `nav_msgs/Odometry` | Planar pose; covariance fields carry the cumulative budget |
| `/dataset/sequence_done` | `std_msgs/String` | End-of-run JSON event |
| `/ground_truth_pose` | `geometry_msgs/PoseStamped` | Optional, evaluation only |

Camera and odometry share a clock. The covariance must be the monotonic
additive budget `path_length_budget_v1` (`vts_core.odometry.BudgetTracker`);
ordinary pose marginals cannot be subtracted and are rejected. The player waits
for an acknowledgement on `/mapping/frame_processed` before each frame.
Language queries only suggest places; in topic mode a destination needs
explicit confirmation (`{"query_id": ..., "confirm": <place>}` on
`/language/command`).

## Tests

```bash
cd /workspaces/topo_sigma/vts_devcontainer/vts_ws
python3 -m pytest -q src
cd .. && uvx ruff check vts_ws/src
```

`vts_core/test/test_regression.py` checks that segments, decisions and loop
counts on the seven manuscript sessions match the tagged code exactly (141
places, 38 accepted revisits); it needs the development session stores.
