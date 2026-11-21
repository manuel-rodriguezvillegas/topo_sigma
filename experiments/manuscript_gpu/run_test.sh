#!/usr/bin/env bash
# Evaluation of the frozen methods on the readable test traversals, on the university GPU, with the
# loop-closure rule of docs/TEST_PROTOCOL_PREDECESSOR_EXCLUSION_2026-11-21.md: the hypotheses of the
# first protocol (topo) and the second (topo_debiased).
#
#   nohup bash experiments/manuscript_gpu/run_test.sh > manuscript_test.log 2>&1 &
#
# It stops before mapping anything if the method code or its calibration files differ from the
# frozen ones. Re-running resumes after a crash: finished replays are skipped. When it ends,
# commit and push experiments/manuscript_gpu/results/.
# Environment: DATA_ROOT (<repo>/encoder), ANYLOC_OUT, OUT, JOBS (6).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE=$REPO/experiments/manuscript_gpu
export DATA_ROOT=${DATA_ROOT:-$REPO/encoder}
ANYLOC_OUT=${ANYLOC_OUT:-$REPO/vts_devcontainer/vts_ws/output/anyloc_published}
OUT=${OUT:-$REPO/vts_devcontainer/vts_ws/output/manuscript_gpu}
JOBS=${JOBS:-6}
FROZEN_CODE=5aa69d4e57f9dba6271fcf4717e1d3177c19cee5
FROZEN_FILES=(
  "results/calibration.json c5bf4295b52435c148a108244e8236ad142fec9029925cc2f0ab3776e383e375"
  "results/visual_likelihood.json de070bffe16bffbf0b6b261dd0f638fc6e172078afe4925a46ef300db75d5f63"
  "results/odometry_bias.json 899eaf6dfa024d5106d09a8c25b3fd8fe43c14ae01613d15a31f82341b05af05"
)
METHOD_CODE=(
  experiments/anyloc_published/historical_src/vts_core
  experiments/anyloc_published/historical_src/vts_players
  experiments/anyloc_published/common.py
  experiments/manuscript_gpu/stream.py
)
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_DISABLE_TELEMETRY=1
step() { echo; echo "=== $(date '+%F %T') $*"; }

step "0. Frozen method"
cd "$HERE"
for entry in "${FROZEN_FILES[@]}"; do
  read -r file expected <<< "$entry"
  [ "$(sha256sum "$file" | cut -d' ' -f1)" = "$expected" ] || { echo "$file differs from the frozen calibration"; exit 1; }
done
git -C "$REPO" diff --quiet "$FROZEN_CODE" HEAD -- "${METHOD_CODE[@]}" \
  || { echo "method code differs from the frozen commit $FROZEN_CODE"; exit 1; }
echo "calibration files and method code match the frozen design"

step "1. Python environment"
source "$HERE/env.sh"

step "2. Tests"
$PY -m pytest -q tests

step "3. Replays of the method, its comparators and its ablations on the readable test traversals"
$PY mapper_runs.py --out "$OUT" --descriptors "$ANYLOC_OUT" --split test --jobs "$JOBS"

step "4. Decisions and summary"
git -C "$REPO" rev-parse HEAD > "$HERE/results/test_commit.txt"
$PY test_report.py --out "$OUT" > /dev/null
cat "$HERE/results/test_summary.md"
step "Done. Commit and push experiments/manuscript_gpu/results/"
