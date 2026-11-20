#!/usr/bin/env bash
# Downloads the final test traversals (experiments/anyloc_published/test_traversals.tsv) and
# extracts their descriptors. It runs no mapping method and computes no metric.
#
#   nohup bash experiments/manuscript_gpu/fetch_test_set.sh > fetch_test_set.log 2>&1 &
#
# Resumable: present recordings and descriptors are skipped. About 100 GB of downloads
# (archives are deleted after extraction) and roughly 70 GB on disk afterwards.
# Environment: DATA_ROOT (<repo>/encoder), ANYLOC_OUT, DL_PARALLEL (4), BATCH (4, ViT-G).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE=$REPO/experiments/manuscript_gpu
LIST=$REPO/experiments/anyloc_published/test_traversals.tsv
export DATA_ROOT=${DATA_ROOT:-$REPO/encoder}
ANYLOC_OUT=${ANYLOC_OUT:-$REPO/vts_devcontainer/vts_ws/output/anyloc_published}
DL_PARALLEL=${DL_PARALLEL:-4}
BATCH=${BATCH:-4}
export PATH=$HOME/.local/bin:$PATH
export HF_HUB_DISABLE_TELEMETRY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$DATA_ROOT/seq_data" "$ANYLOC_OUT"
step() { echo; echo "=== $(date '+%F %T') $*"; }

step "0. Checks"
for tool in git curl tar unzip md5sum nvidia-smi uv; do
  command -v $tool >/dev/null || { echo "missing tool: $tool"; exit 1; }
done
echo "git: $(git -C "$REPO" rev-parse --short HEAD); free disk under $DATA_ROOT: $(df -h "$DATA_ROOT" | awk 'NR==2 {print $4}')"

step "1. Python environment"
source "$HERE/env.sh"

step "2. Downloads ($(grep -vc '^#' "$LIST") traversals)"
source "$REPO/experiments/datasets.sh"
awk -F'\t' '!/^#/ { print $2, $4, ($2 == "cid" ? $3 : "") }' "$LIST" | fetch_all \
  || echo "Some downloads failed (see above); rerun to retry them. The inventory lists what is missing."

step "3. Frame streams (inventory in experiments/manuscript_gpu/results/test_inventory.json)"
SEQUENCES=$(cd "$HERE" && $PY test_inventory.py)
echo "$(echo $SEQUENCES | wc -w) of $(grep -vc '^#' "$LIST") traversals yield a frame stream"

step "4. Descriptors: DINOv3 (gated: hf auth login), DINOv2, AnyLoc ViT-G"
(cd "$REPO/experiments/anyloc_published" && $PY extract.py --out "$ANYLOC_OUT" --sequences $SEQUENCES \
  --encoders dinov3_s dinov2_s anyloc_gem_g anyloc_vlad_g --batch "$BATCH")

step "Done. Nothing has been evaluated. Commit and push experiments/manuscript_gpu/results/test_inventory.json"
