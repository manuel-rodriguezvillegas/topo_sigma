#!/usr/bin/env bash
# End-to-end node-creation sensitivity study.
#
# The final adaptive k=2 maps already exist and are never rebuilt here. This
# runner creates four additional adaptive variants and one fixed-prominence
# baseline across all seven sequences (35 new maps). Every variant uses an
# explicit output suffix, and the underlying runners reject detector overrides
# without one, so output/submission/<environment>/ cannot be overwritten.
#
# Existing completed variants are reused by default. Set FORCE=1 only to
# intentionally rebuild every sensitivity map.
set -euo pipefail

run_adaptive() {
    local k="$1"
    local suffix="$2"
    echo "=== Node sensitivity: adaptive k=${k} ==="
    VALLEY_MODE=adaptive VALLEY_K="${k}" VARIANT_SUFFIX="${suffix}" \
        RUN_ABLATIONS=0 ./run_experiments.sh
    VALLEY_MODE=adaptive VALLEY_K="${k}" VARIANT_SUFFIX="${suffix}" \
        RUN_ABLATIONS=0 ./run_cid_sims_experiments.sh
}

run_fixed() {
    # Read fresh label-free calibration against the adaptive k=2 node count.
    local delta
    delta="$(python3 - "${OUTPUT_ROOT:-output/submission}/node_segmentation_experiment.json" <<'PYCODE'
import json
import sys
with open(sys.argv[1]) as stream:
    print(json.load(stream)["fixed_baseline_calibration"]["delta"])
PYCODE
)"
    local suffix="_node_fixed"
    echo "=== Node sensitivity: fixed delta=${delta} ==="
    VALLEY_MODE=fixed VALLEY_DELTA="${delta}" VARIANT_SUFFIX="${suffix}" \
        RUN_ABLATIONS=0 ./run_experiments.sh
    VALLEY_MODE=fixed VALLEY_DELTA="${delta}" VARIANT_SUFFIX="${suffix}" \
        RUN_ABLATIONS=0 ./run_cid_sims_experiments.sh
}

run_adaptive 1.0 _node_adaptive_k1p0
run_adaptive 1.5 _node_adaptive_k1p5
run_adaptive 2.5 _node_adaptive_k2p5
run_adaptive 3.0 _node_adaptive_k3p0
run_fixed

echo "All end-to-end node-sensitivity runs complete."
echo "The existing unsuffixed directories remain the adaptive k=2 reference."
