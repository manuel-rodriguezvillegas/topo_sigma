#!/usr/bin/env bash
# Every number and figure of the manuscript, on the university GPU.
#
#   nohup bash experiments/manuscript_gpu/run_all.sh > manuscript_gpu.log 2>&1 &
#   tail -f manuscript_gpu.log
#
# Re-running resumes: finished replays are skipped. When it ends, commit and push
# experiments/manuscript_gpu/results/.
# Environment: DATA_ROOT (<repo>/encoder), ANYLOC_OUT, OUT, JOBS (6).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE=$REPO/experiments/manuscript_gpu
export DATA_ROOT=${DATA_ROOT:-$REPO/encoder}
ANYLOC_OUT=${ANYLOC_OUT:-$REPO/vts_devcontainer/vts_ws/output/anyloc_published}
OUT=${OUT:-$REPO/vts_devcontainer/vts_ws/output/manuscript_gpu}
JOBS=${JOBS:-6}
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$OUT" "$HERE/results"
step() { echo; echo "=== $(date '+%F %T') $*"; }

step "1. Data and GPU descriptors of all 19 traversals (AnyLoc runner; resumes)"
RUN_HELDOUT=1 OUT=$ANYLOC_OUT bash "$REPO/experiments/anyloc_published/run_all.sh"

step "2. Python environment"
source "$HERE/env.sh"

step "2b. DINOv3 descriptors (gated weights: request access on Hugging Face and run hf auth login first)"
(cd "$REPO/experiments/anyloc_published" && $PY extract.py --out "$ANYLOC_OUT" --encoders dinov3_s --batch 16)
cd "$HERE"

step "3. Tests"
$PY -m pytest -q tests

step "4. Odometry calibration (development ground truth only)"
$PY calibrate.py

step "5. Geometric replays"
$PY mapper_runs.py --out "$OUT" --descriptors "$ANYLOC_OUT" --split dev --exclude valley_fixed --jobs "$JOBS"
$PY mapper_runs.py --out "$OUT" --descriptors "$ANYLOC_OUT" --split heldout --jobs "$JOBS"

step "6. Node-creation sensitivity, then the fixed-prominence detector"
$PY segmentation.py --descriptors "$ANYLOC_OUT"
$PY mapper_runs.py --out "$OUT" --descriptors "$ANYLOC_OUT" --split dev --variants valley_fixed --jobs "$JOBS"

step "7. Visual likelihood ratios, then the visual replays"
$PY visual_calibration.py --out "$OUT"
$PY mapper_runs.py --out "$OUT" --descriptors "$ANYLOC_OUT" --split dev --visual --jobs "$JOBS"
$PY mapper_runs.py --out "$OUT" --descriptors "$ANYLOC_OUT" --split heldout --visual --jobs "$JOBS"

step "8. Text retrieval (SigLIP 2)"
(cd "$REPO/experiments/anyloc_published/historical_src" && $PY -m vts_evaluation.language_benchmark \
  --benchmark "$HERE/language_queries_100.json" --maps-root "$OUT/replay" --graph-name topo/graph.pkl \
  --out "$HERE/results/language.json")

step "9. Figures"
$PY figures.py --out "$OUT"

step "10. Summary"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$HERE/results/gpu.txt"
git -C "$REPO" rev-parse HEAD > "$HERE/results/commit.txt"
$PY report.py --out "$OUT" > /dev/null
cat "$HERE/results/summary.md"
step "Done. Commit and push experiments/manuscript_gpu/results/"
