#!/usr/bin/env bash
# Build, test and regenerate submission artifacts inside the Jazzy devcontainer.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select \
  vts_core vts_players vts_mapping vts_evaluation vts_bringup vts_language
source install/setup.bash
set -u
python3 -m pytest -q src
if [ "${1:-}" = "--test-only" ]; then
  exit 0
fi
if [ "$#" -gt 0 ]; then
  echo "Usage: $0 [--test-only]" >&2
  exit 2
fi
# Replay pacing affects wall time, not recorded sampling or odometry integration.
export PLAYER_RATE_HZ="${PLAYER_RATE_HZ:-100.0}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-output/submission}"
RUN_ABLATIONS=1 ./run_experiments.sh
RUN_ABLATIONS=1 ./run_cid_sims_experiments.sh
VISUAL_MODEL=cosine VARIANT_SUFFIX=_cosine RUN_ABLATIONS=0 ./run_experiments.sh
VISUAL_MODEL=cosine VARIANT_SUFFIX=_cosine RUN_ABLATIONS=0 ./run_cid_sims_experiments.sh
./run_anyloc_baseline.sh all
./evaluate_node_segmentation.sh
./run_node_sensitivity_experiments.sh
INCLUDE_ABLATIONS=1 INCLUDE_COSINE=1 INCLUDE_NODE_SENSITIVITY=1 ./evaluate_tolerance_sweep.sh
python3 -m vts_evaluation.summarize_experiments --root "${OUTPUT_ROOT}"
python3 -m vts_evaluation.language_benchmark \
  --benchmark language_queries_100.json --maps-root "${OUTPUT_ROOT}" \
  --out "${OUTPUT_ROOT}/language_benchmark.json"
echo "Verified experiment artifacts: ${OUTPUT_ROOT}"
echo "Refresh manuscript tables, figures and conclusions from these new results before submission."
