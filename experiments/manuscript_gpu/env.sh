# Python environment of the GPU runners. Source it after setting REPO; it defines PY.
VENV=$REPO/experiments/gate_verification/.venv
[ -x "$VENV/bin/python" ] || VENV=$REPO/experiments/anyloc_published/.venv
PY=$VENV/bin/python
uv pip install --python "$PY" -q pytest matplotlib "transformers>=5.2,<6" sentencepiece
$PY -c "import torch; assert torch.cuda.is_available(), 'PyTorch does not see the GPU'; print(torch.cuda.get_device_name(0))"
