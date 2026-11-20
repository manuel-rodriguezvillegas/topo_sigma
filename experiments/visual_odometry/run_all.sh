#!/usr/bin/env bash
# Visual heading diagnostic on the 19 design traversals, on the university GPU. No test traversal is read.
#
#   nohup bash experiments/visual_odometry/run_all.sh > visual_odometry.log 2>&1 &
#
# Resumable: traversals whose matches exist are skipped. When it ends, commit and push
# experiments/visual_odometry/results/.
# Environment: DATA_ROOT (<repo>/encoder), OUT.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE=$REPO/experiments/visual_odometry
export DATA_ROOT=${DATA_ROOT:-$REPO/encoder}
OUT=${OUT:-$REPO/vts_devcontainer/vts_ws/output/visual_odometry}
LIGHTGLUE_COMMIT=5a650c444718caffc2669dc0094cde908f42dd73
export PATH=$HOME/.local/bin:$PATH
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$OUT/matches" "$HERE/results"
step() { echo; echo "=== $(date '+%F %T') $*"; }

step "1. Python environment"
source "$REPO/experiments/manuscript_gpu/env.sh"
uv pip install --python "$PY" -q "lightglue @ git+https://github.com/cvg/LightGlue.git@$LIGHTGLUE_COMMIT"

step "2. Tests"
cd "$HERE"
$PY -m pytest -q tests

step "3. SuperPoint + LightGlue matches between consecutive frames (GPU)"
$PY matches.py --out "$OUT/matches"

step "4. Heading and position drift: wheels, vision, fusion"
git -C "$REPO" rev-parse HEAD > results/commit.txt
$PY diagnostic.py --matches "$OUT/matches"
step "Done. Commit and push experiments/visual_odometry/results/"
