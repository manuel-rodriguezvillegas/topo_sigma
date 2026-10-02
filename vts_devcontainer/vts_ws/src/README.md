# TopoSIGMA — ROS 2 workspace

The workspace targets ROS 2 Jazzy on Ubuntu 24.04 and CPython 3.12. Its
third-party Python environment is managed centrally by uv from
`../../pyproject.toml` and `../../uv.lock`.

TopoSIGMA consumes standard ROS topics and never accesses a dataset
directly. Each dataset has a small player that implements the same interface.
The COLD player synchronizes camera frames to the dataset's recorded wheel
odometry. Because COLD provides no covariance, one fixed motion model
accumulates a path-length heuristic budget without perturbing the measured trajectory.

## Packages

```
vts_core/        Pure-Python algorithm library (no ROS imports):
                 topo_graph, motion, features, directional, node_detection,
                 matching, mapper, frame_buffer, retrieval, metrics.
vts_players/     COLD player — the ONLY COLD-aware code in the project.
vts_mapping/     graph_builder node (image + odom -> final_graph.pkl).
vts_language/    commands node (natural-language query -> node target).
vts_evaluation/  Offline metrics, retrieval evaluation and map plots.
vts_bringup/     pipeline.launch.py + per-environment YAML configs.
```

## Topic contract

Any dataset player or real robot must provide:

| Topic                    | Type                      | Notes                     |
|--------------------------|---------------------------|---------------------------|
| `/camera/image`          | sensor_msgs/Image (bgr8)  |                           |
| `/odom`                  | nav_msgs/Odometry         | additive budget in covariance fields    |
| `/ground_truth_pose`     | geometry_msgs/PoseStamped | optional, evaluation only |
| `/dataset/room_label`    | std_msgs/String           | optional, evaluation only |
| `/dataset/sequence_done` | std_msgs/String (JSON)    | end-of-run signal         |

The revisit mechanism uses covariance reported on `/odom` to generate and
rank geometrically compatible candidates. A unique candidate is accepted
directly; when several places remain plausible, bidirectional three-node
sequences compare training-free vMF distributions fitted to each node's
normalized DINO descriptors. Adapters must explicitly declare
`uncertainty_contract=path_length_budget_v1` and publish the monotonic budget
from `vts_core.motion`; ordinary pose marginals are unsupported. See
`docs/SUBMISSION_CHANGES.md` in the repository root for the units and limitations.

## Quick start

```bash
cd /workspaces/topo_sigma/vts_devcontainer/vts_ws

colcon build --symlink-install
source install/setup.bash

# Build the map (single sequence -> output/submission/<env>/final_graph.pkl):
ros2 launch vts_bringup pipeline.launch.py config:=cold_freiburg_a.yaml mode:=building

# Answer a natural-language query against the stored map:
ros2 launch vts_bringup pipeline.launch.py config:=cold_freiburg_a.yaml mode:=command

# Offline evaluation:
python3 -m vts_evaluation.evaluate_run \
    --graph output/submission/freiburg_a/final_graph.pkl \
    --node-gt output/submission/freiburg_a/graph_0_node_gt.json \
    --gt-trajectory /workspace/encoder/seq_data/cold-freiburg_part_a_seq2_night1/std_cam
```

Python dependencies beyond ROS: numpy, scipy, opencv-python, torch,
torchvision, transformers, pillow, pyyaml.

## Playback and short revisits

Both dataset players default to a 50 Hz playback cap. They wait for the mapper's
acknowledgement before sending the next frame, so actual speed is limited by
processing time. This changes wall-clock replay speed, not the selected images
or recorded odometry. Override the cap with `player_rate_hz:=5.0` in ROS launch
or `PLAYER_RATE_HZ=5` in the experiment scripts.

Loop search excludes only the immediately preceding node. A short return
`A -> B -> A` is eligible from the third emitted node, with no map-size warm-up.
The visual-only ablation still needs a comparison score to establish an outlier;
a singleton is not automatically visually distinctive. The evaluation uses the
same exclusion, so recompute metrics before comparing historical runs.

## DINO inference device

The graph builder defaults to `dino_device:=auto`, which selects CUDA, then
Apple MPS, then CPU according to what the installed PyTorch runtime exposes.
The selected device is printed once when the graph builder starts. It can be
overridden for diagnosis or reproducibility:

```bash
ros2 launch vts_bringup pipeline.launch.py \
    config:=cold_freiburg_a.yaml dino_device:=cpu
```

Valid values are `auto`, `cuda`, `mps`, and `cpu`; explicitly requesting an
unavailable accelerator fails immediately. Apple Metal/MPS is not exposed to
Linux containers by Docker Desktop, so this project's devcontainer uses CPU
on a Mac. MPS is available only when running the Python pipeline natively on
macOS with an MPS-enabled PyTorch installation. On a CUDA workstation, Docker
must be configured with the NVIDIA Container Toolkit, the container must be
started with GPU access, and PyTorch must be a CUDA-enabled build. Device
selection cannot provide those host/container capabilities by itself.
