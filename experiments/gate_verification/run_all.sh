#!/usr/bin/env bash
# Gate-verification diagnostic, end to end, on a Linux machine with an NVIDIA GPU.
# Development data only (COLD Freiburg/Saarbrücken + CID-SIMS apartments _1/_2);
# test recordings are never downloaded.
#
#   1. Python 3.12 environment (uv), PyTorch for the local CUDA driver, LightGlue, MASt3R.
#   2. Downloads (resumable): 52 COLD recordings (~28 GB of tars) and 6 CID-SIMS
#      traversals (~29 GB of zips, MD5-checked). Only the needed folders are kept.
#   3. DINOv3 stores, baseline maps, loop evaluation (must match the Mac reference).
#   4. Diagnostic (LightGlue + MASt3R on the top-5 appearance candidates) and summary.
#
# Usage, from the repository root (keeps running if the VS Code tunnel drops):
#   export HF_TOKEN=hf_...      # account with access to facebook/dinov3-vits16-pretrain-lvd1689m
#   nohup bash experiments/gate_verification/run_all.sh > gate_verification.log 2>&1 &
#   tail -f gate_verification.log
# Re-running resumes: finished steps and downloads are skipped.
#
# Environment: DATA_ROOT (default <repo>/encoder), JOBS (mapping processes, 8),
# DL_PARALLEL (parallel downloads, 4), KEEP_ARCHIVES (1 keeps tar/zip files).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE=$REPO/experiments/gate_verification
WS=$REPO/vts_devcontainer/vts_ws
DATA=${DATA_ROOT:-$REPO/encoder}
JOBS=${JOBS:-8}
DL_PARALLEL=${DL_PARALLEL:-4}
VENV=$HERE/.venv
OUT=$WS/output/gate_verification
STAMPS=$OUT/stamps
mkdir -p "$OUT" "$STAMPS" "$DATA/seq_data"
export PATH=$HOME/.local/bin:$PATH

step() { echo; echo "=== $(date '+%F %T') $*"; }
done_once() { [ -f "$STAMPS/$1" ]; }
mark() { touch "$STAMPS/$1"; }

# --------------------------------------------------------------------------- #
step "0. Checks"
for tool in git curl tar unzip md5sum nvidia-smi; do
  command -v $tool >/dev/null || { echo "missing tool: $tool"; exit 1; }
done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HOME/.cache/huggingface/token" ]; then
  echo "DINOv3 is gated: export HF_TOKEN=hf_... (or run 'hf auth login') first."; exit 1
fi
echo "free disk under $DATA: $(df -h "$DATA" | awk 'NR==2 {print $4}') (needs ~65 GB with archives deleted)"

# --------------------------------------------------------------------------- #
step "1. Python environment"
if ! done_once env; then
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)
  cuda=${cuda:-118}
  if   [ "$cuda" -ge 128 ]; then tag=cu128
  elif [ "$cuda" -ge 126 ]; then tag=cu126
  elif [ "$cuda" -ge 124 ]; then tag=cu124
  elif [ "$cuda" -ge 121 ]; then tag=cu121
  else tag=cu118; fi
  echo "driver CUDA $cuda -> PyTorch wheels $tag"
  uv venv --python 3.12 "$VENV"
  uv pip install --python "$VENV/bin/python" torch torchvision --index-url "https://download.pytorch.org/whl/$tag"
  uv pip install --python "$VENV/bin/python" "numpy<2" scipy pyyaml pillow "opencv-python-headless<4.12" \
    "transformers>=4.56" huggingface_hub safetensors einops roma trimesh tqdm scikit-learn matplotlib packaging \
    "lightglue @ git+https://github.com/cvg/LightGlue.git"
  if [ ! -d "$HERE/third_party/mast3r/dust3r" ]; then
    rm -rf "$HERE/third_party/mast3r"
    git clone --recursive https://github.com/naver/mast3r "$HERE/third_party/mast3r"
  fi
  "$VENV/bin/python" - <<EOF
import sys, torch
sys.path.insert(0, "$HERE/third_party/mast3r")
import mast3r.utils.path_to_dust3r  # noqa
from mast3r.model import AsymmetricMASt3R  # noqa
from lightglue import LightGlue, SuperPoint  # noqa
assert torch.cuda.is_available(), "PyTorch does not see the GPU"
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
EOF
  mark env
fi
PY=$VENV/bin/python
git -C "$HERE/third_party/mast3r" rev-parse HEAD > "$OUT/mast3r_commit.txt"

step "1b. Verifier smoke test (before hours of downloads)"
if ! done_once smoke; then
  (cd "$HERE" && $PY - <<EOF
from pathlib import Path
from verifiers import LightGlueVerifier, Mast3rVerifier
assets = sorted(Path("third_party/mast3r").rglob("Chateau*.png"))[:2]
assert len(assets) == 2, "no sample images found in the MASt3R repository"
a, b = map(str, assets)
print("LightGlue:", LightGlueVerifier("cuda").score(a, b))
row = Mast3rVerifier(Path("third_party/mast3r"), "cuda").score_many([(a, b)])[0]
print("MASt3R:", row)
assert "m_error" not in row and row["m_matches"] > 0 and "m_dist" in row, "MASt3R verifier failed"
EOF
  )
  mark smoke
fi

# --------------------------------------------------------------------------- #
step "2. Downloads (development data only)"
cold_get() {  # lab part seq_cond
  local lab=$1 part=$2 name=$3
  local target="$DATA/seq_data/cold-${lab}_part_${part}_${name}"
  if [ -f "$target/odom_scans/odom.tdf" ] && [ -d "$target/std_cam" ]; then echo "$target: present"; return; fi
  local tarball="$DATA/seq_data/_${lab}_${part}_${name}.tar"
  for attempt in 1 2 3 4 5; do
    curl -sS -L -C - --retry 5 -o "$tarball" "https://www.cas.kth.se/COLD/db/cold-${lab}/part_${part}/${name}.tar" && break
    sleep 15
  done
  rm -rf "$target.partial" && mkdir -p "$target.partial"
  local top=$(tar -tf "$tarball" 2>/dev/null | head -1 | cut -d/ -f1)
  tar -xf "$tarball" -C "$target.partial" --strip-components=1 --wildcards \
      "${top}/std_cam/*" "${top}/odom_scans/*" "${top}/localization/*" \
    || { echo "$tarball: EXTRACTION FAILED (delete it to re-download)"; return 1; }
  mv "$target.partial" "$target"
  [ "${KEEP_ARCHIVES:-0}" = 1 ] || rm -f "$tarball"
  echo "$target: OK ($(ls "$target/std_cam" | wc -l) images)"
}
cid_get() {  # file_id zip md5 target
  local id=$1 zip=$2 md5=$3 target="$DATA/$4"
  if [ -f "$target/odom.txt" ] && [ -d "$target/color" ]; then echo "$target: present"; return; fi
  local archive="$DATA/$zip"
  for attempt in 1 2 3 4 5; do
    curl -sS -L -C - --retry 5 -o "$archive" \
      "https://download.scidb.cn/download?fileId=$id&dataSetType=organization&fileName=$zip" && break
    sleep 15
  done
  if [ "$(md5sum "$archive" | cut -d' ' -f1)" != "$md5" ]; then
    echo "$archive: MD5 MISMATCH (delete it to re-download)"; return 1
  fi
  local staging="$DATA/_staging_$4"
  rm -rf "$staging" && mkdir -p "$staging"
  unzip -q -o "$archive" '*color/*' '*odom.txt' '*groundtruth.txt' -d "$staging" \
    || { echo "$archive: UNZIP FAILED"; return 1; }
  if [ -d "$staging/color" ]; then mv "$staging" "$target"
  else mv "$staging/$(ls "$staging" | head -1)" "$target" && rm -rf "$staging"; fi
  [ "${KEEP_ARCHIVES:-0}" = 1 ] || rm -f "$archive"
  echo "$target: OK ($(ls "$target/color" | wc -l) images)"
}
export -f cold_get cid_get
export DATA KEEP_ARCHIVES
if ! done_once downloads; then
  # The 52 COLD development recordings that pass the odometry quality check.
  {
    for c in cloudy1 cloudy2 cloudy3 night1 night2 sunny2 sunny4; do echo "cold freiburg a seq1_$c"; done
    for c in cloudy1 cloudy2 cloudy3 night1 night2 night3 sunny1 sunny2 sunny3 sunny4; do echo "cold freiburg a seq2_$c"; done
    for c in cloudy1 cloudy2 cloudy3 sunny1; do echo "cold freiburg b seq3_$c"; done
    for s in seq1 seq2; do for c in cloudy1 cloudy2 cloudy3 night1 night2 night3; do echo "cold saarbruecken a ${s}_$c"; done; done
    for c in cloudy1 cloudy2 cloudy4 cloudy5 night1 night2 night3 sunny1 sunny2 sunny3; do echo "cold saarbruecken b seq3_$c"; done
    for c in cloudy1 cloudy2 cloudy3 night1 night2 night3 sunny1 sunny2 sunny3; do echo "cold saarbruecken b seq4_$c"; done
    echo "cid 120e974e613c09246d0013e6358af3d5 apartment1_1.zip 66e8af5ca7e2d7d1d7af446f664a3c92 apartment1_1"
    echo "cid 33d89167ba7001dd4169ad3e3195dd5c apartment1_2.zip 0cc885d26819a9ee759edbef7ae2c6f1 apartment1_2"
    echo "cid 502fe6af078df57e4b4977c6d2a6ec4d apartment2_1.zip 88b21eade87e340abb7a8e70b350d3ef apartment2_1"
    echo "cid c877c22d29f6b0c9bfabc38f3bdde9ef apartment2_2.zip cfe372360e8efdc55c9dddb1b0b3a731 apartment2_2"
    echo "cid 30bd0a5912d167685f6cab0afcd7c869 apartment3_1.zip 0d59a683f8a8526f65e61f59e77129bb apartment3_1"
    echo "cid 0926aed6b63f3caefdb0e7ba18f0b4bf apartment3_2.zip fb1010f6354f36b97038da4025db679a apartment3_2"
  } > "$OUT/downloads.txt"
  # CID zips first (slowest server), interleaved with COLD.
  { grep '^cid' "$OUT/downloads.txt"; grep '^cold' "$OUT/downloads.txt"; } \
    | xargs -P "$DL_PARALLEL" -L 1 bash -c 'kind=$1; shift; if [ "$kind" = cold ]; then cold_get "$@"; else cid_get "$@"; fi' _ || true
  missing=0
  while read -r kind a b c d; do
    if [ "$kind" = cold ]; then [ -f "$DATA/seq_data/cold-${a}_part_${b}_${c}/odom_scans/odom.tdf" ] || { echo "missing COLD $a $b $c"; missing=1; }
    else [ -f "$DATA/$d/odom.txt" ] || { echo "missing CID $d"; missing=1; }; fi
  done < "$OUT/downloads.txt"
  [ "$missing" = 0 ] || { echo "Some downloads failed; re-run the script to resume."; exit 1; }
  mark downloads
fi

# --------------------------------------------------------------------------- #
export TOPOSIGMA_DATA_ROOT=$DATA PYTHONPATH=$WS/src/vts_core HF_HUB_DISABLE_TELEMETRY=1
CONFIG=$OUT/gpu.yaml
printf 'encoders:\n  device: cuda\n  batch_size: 32\n' > "$CONFIG"
TS="$PY -m vts_core.cli"

step "3a. DINOv3 stores"
done_once stores || { $TS frames build --all-missing --config "$CONFIG"; mark stores; }
step "3b. Baseline maps"
done_once maps || { $TS map --all --jobs "$JOBS" --config "$CONFIG"; mark maps; }
step "3c. Loop evaluation"
$TS evaluate loops --config "$CONFIG" | tail -n 25

# --------------------------------------------------------------------------- #
step "4. Diagnostic (resumable per session)"
(cd "$HERE" && $PY diagnose.py --out "$OUT/diagnostic" --verifiers lg,m)
step "5. Summary"
(cd "$HERE" && $PY summarize.py "$OUT/diagnostic") > /dev/null
cp "$WS/output/toposigma/evaluate/loops.json" "$OUT/diagnostic/loops.json"
nvidia-smi --query-gpu=name --format=csv,noheader > "$OUT/diagnostic/gpu.txt"
tar -czf "$REPO/gate_verification_results.tar.gz" -C "$OUT" diagnostic mast3r_commit.txt
cat "$OUT/diagnostic/summary.md"
step "Done. Bring back: $REPO/gate_verification_results.tar.gz"
