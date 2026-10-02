#!/usr/bin/env bash
# Focused node replay: no ROS, image inference or full ablation suite.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD/src/vts_core:$PWD/src/vts_evaluation${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m vts_evaluation.closure_comparison "$@"
