#!/usr/bin/env bash
# Published AnyLoc (ViT-G/14) against the exact manuscript mapper, on an NVIDIA GPU.
#
#   1. Python environment: reuses experiments/gate_verification/.venv if it exists
#      (PyTorch CUDA from the October run); otherwise creates one here.
#   2. Downloads only what is missing: the 7 manuscript traversals and, with
#      RUN_HELDOUT=1, the 12 held-out ones.
#   3. GPU descriptors (DINOv2-S, AnyLoc-GeM S/G, AnyLoc-VLAD-G indoor vocabulary).
#   4. Replay of the manuscript mapper. The development step must reproduce the
#      manuscript exactly (25/2/23 at 2 m and the cosine ablation) or the script stops.
#   5. Summary in experiments/anyloc_published/results/ (small files: commit and push them).
#
# Usage, from the repository root (survives a dropped SSH/VS Code session):
#   nohup bash experiments/anyloc_published/run_all.sh > anyloc_published.log 2>&1 &
#   RUN_HELDOUT=1 nohup bash experiments/anyloc_published/run_all.sh > anyloc_published.log 2>&1 &
#   tail -f anyloc_published.log
# Re-running resumes: finished descriptors and replays are skipped.
#
# Environment: DATA_ROOT (default <repo>/encoder), OUT (default <repo>/vts_devcontainer/vts_ws/output/anyloc_published),
# JOBS (replay processes, 6), BATCH (ViT-G batch, 4), RUN_HELDOUT (0), DL_PARALLEL (4).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE=$REPO/experiments/anyloc_published
export DATA_ROOT=${DATA_ROOT:-$REPO/encoder}
OUT=${OUT:-$REPO/vts_devcontainer/vts_ws/output/anyloc_published}
JOBS=${JOBS:-6}
BATCH=${BATCH:-4}
RUN_HELDOUT=${RUN_HELDOUT:-0}
DL_PARALLEL=${DL_PARALLEL:-4}
mkdir -p "$OUT" "$DATA_ROOT/seq_data"
export PATH=$HOME/.local/bin:$PATH
step() { echo; echo "=== $(date '+%F %T') $*"; }

step "0. Checks"
for tool in git curl tar unzip md5sum nvidia-smi; do
  command -v $tool >/dev/null || { echo "missing tool: $tool"; exit 1; }
done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "git: $(git -C "$REPO" rev-parse --short HEAD) $(git -C "$REPO" status --porcelain | wc -l) local changes"
echo "free disk under $DATA_ROOT: $(df -h "$DATA_ROOT" | awk 'NR==2 {print $4}')"
[ -f "$HERE/reference/dinov2_vits14_mac.npz" ] || { echo "missing reference/dinov2_vits14_mac.npz (git pull?)"; exit 1; }

# --------------------------------------------------------------------------- #
step "1. Python environment"
VENV=$REPO/experiments/gate_verification/.venv
if [ ! -x "$VENV/bin/python" ]; then
  VENV=$HERE/.venv
  if [ ! -x "$VENV/bin/python" ]; then
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)
    cuda=${cuda:-118}
    if   [ "$cuda" -ge 128 ]; then tag=cu128
    elif [ "$cuda" -ge 126 ]; then tag=cu126
    elif [ "$cuda" -ge 124 ]; then tag=cu124
    elif [ "$cuda" -ge 121 ]; then tag=cu121
    else tag=cu118; fi
    uv venv --python 3.12 "$VENV"
    uv pip install --python "$VENV/bin/python" torch torchvision --index-url "https://download.pytorch.org/whl/$tag"
  fi
fi
uv pip install --python "$VENV/bin/python" -q "numpy<2" scipy pyyaml pillow "opencv-python-headless<4.12"
PY=$VENV/bin/python
$PY -c "import torch; assert torch.cuda.is_available(), 'PyTorch does not see the GPU'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

# --------------------------------------------------------------------------- #
step "2. Data (only missing recordings)"
source "$REPO/experiments/datasets.sh"
{
  echo "cold freiburg a seq2_night1"; echo "cold freiburg b seq3_sunny1"
  echo "cold saarbruecken a seq2_night2"; echo "cold saarbruecken b seq4_sunny1"
  echo "cid 120e974e613c09246d0013e6358af3d5 apartment1_1.zip 66e8af5ca7e2d7d1d7af446f664a3c92 apartment1_1"
  echo "cid 502fe6af078df57e4b4977c6d2a6ec4d apartment2_1.zip 88b21eade87e340abb7a8e70b350d3ef apartment2_1"
  echo "cid 30bd0a5912d167685f6cab0afcd7c869 apartment3_1.zip 0d59a683f8a8526f65e61f59e77129bb apartment3_1"
  if [ "$RUN_HELDOUT" = 1 ]; then
    echo "cold freiburg a seq1_cloudy2"; echo "cold freiburg a seq2_cloudy2"; echo "cold freiburg b seq3_cloudy2"
    echo "cold saarbruecken a seq1_cloudy1"; echo "cold saarbruecken a seq2_cloudy2"
    echo "cold saarbruecken b seq3_cloudy1"; echo "cold saarbruecken b seq4_cloudy2"
    echo "cold ljubljana a seq1_night3"; echo "cold ljubljana a seq2_cloudy2"
    echo "cid 33d89167ba7001dd4169ad3e3195dd5c apartment1_2.zip 0cc885d26819a9ee759edbef7ae2c6f1 apartment1_2"
    echo "cid c877c22d29f6b0c9bfabc38f3bdde9ef apartment2_2.zip cfe372360e8efdc55c9dddb1b0b3a731 apartment2_2"
    echo "cid 0926aed6b63f3caefdb0e7ba18f0b4bf apartment3_2.zip fb1010f6354f36b97038da4025db679a apartment3_2"
  fi
} > "$OUT/downloads.txt"
xargs -P "$DL_PARALLEL" -L 1 bash -c 'kind=$1; shift; if [ "$kind" = cold ]; then cold_get "$@"; else cid_get "$@"; fi' _ \
  < "$OUT/downloads.txt"

# --------------------------------------------------------------------------- #
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_DISABLE_TELEMETRY=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$HERE"
DEV="freiburg_a freiburg_ext saarbruecken_a saarbruecken_ext cid_sims_apartment1_1 cid_sims_apartment2_1 cid_sims_apartment3_1"

step "3. Reproduction gate (manuscript mapper, Mac descriptors; no GPU)"
$PY replay.py --out "$OUT" --split dev --variants topo topo_cosine geometry_only --jobs "$JOBS"
grep -q '"PASS"' "$OUT/reproduction_dev.json" || { echo "Reproduction FAILED: stop and send anyloc_published.log"; exit 1; }

step "4. GPU descriptors, development traversals"
$PY extract.py --out "$OUT" --sequences $DEV --batch "$BATCH"
step "5. Replay, development traversals"
$PY replay.py --out "$OUT" --split dev --jobs "$JOBS"

if [ "$RUN_HELDOUT" = 1 ]; then
  HELD="heldout_freiburg_a_seq1 heldout_freiburg_a_seq2 heldout_freiburg_b_seq3 heldout_saarbruecken_a_seq1
        heldout_saarbruecken_a_seq2 heldout_saarbruecken_b_seq3 heldout_saarbruecken_b_seq4
        heldout_ljubljana_a_seq1 heldout_ljubljana_a_seq2 heldout_cid_apartment1_2 heldout_cid_apartment2_2
        heldout_cid_apartment3_2"
  step "6. GPU descriptors, held-out traversals"
  $PY extract.py --out "$OUT" --sequences $HELD --batch "$BATCH"
  step "7. Replay, held-out traversals"
  $PY replay.py --out "$OUT" --split heldout --jobs "$JOBS"
fi

step "8. Summary"
$PY summarize.py --out "$OUT" > /dev/null
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$HERE/results/gpu.txt"
cp "$OUT/reproduction_dev.json" "$HERE/results/"
cp "$OUT/descriptors/identities.json" "$HERE/results/" 2>/dev/null || true
cat "$HERE/results/summary.md"
step "Done. Commit and push experiments/anyloc_published/results/"
