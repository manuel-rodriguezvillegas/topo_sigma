#!/usr/bin/env bash
# Videos of the mapper on development traversals, on the university GPU (CPU only; a few
# minutes each). It needs the replays of run_all.sh in $OUT and stops if a replay does not reproduce them.
#
#   nohup bash experiments/manuscript_gpu/run_video.sh > manuscript_video.log 2>&1 &
#
# Writes $OUT/video/<profile>/<traversal>.* for the profiles readme (MP4 and animated WebP), journal
# (1080p MP4 and poster) and web (short MP4 and poster); nothing is committed.
# Environment: DATA_ROOT (<repo>/encoder), ANYLOC_OUT, OUT, PROFILES (all three), SEQUENCES (Freiburg A
# and CID-SIMS apartment 3, the COLD and CID-SIMS maps with most loop closures to show).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE=$REPO/experiments/manuscript_gpu
export DATA_ROOT=${DATA_ROOT:-$REPO/encoder}
ANYLOC_OUT=${ANYLOC_OUT:-$REPO/vts_devcontainer/vts_ws/output/anyloc_published}
OUT=${OUT:-$REPO/vts_devcontainer/vts_ws/output/manuscript_gpu}
PROFILES=${PROFILES:-readme journal web}
SEQUENCES=${SEQUENCES:-freiburg_a cid_sims_apartment3_1}
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_DISABLE_TELEMETRY=1
step() { echo; echo "=== $(date '+%F %T') $*"; }

step "1. Python environment"
source "$HERE/env.sh"
uv pip install --python "$PY" -q imageio-ffmpeg pillow
cd "$HERE"

step "2. Tests"
$PY -m pytest -q tests

step "3. Videos (each replay must reproduce its recorded result first)"
# shellcheck disable=SC2086
$PY video.py --out "$OUT" --descriptors "$ANYLOC_OUT" --videos "$OUT/video" --profiles $PROFILES \
  --sequences $SEQUENCES
ls -lhR "$OUT/video"
step "Done. Videos in $OUT/video"
