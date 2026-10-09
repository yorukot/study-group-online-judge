#!/usr/bin/env bash
set -euo pipefail

module purge
module load cuda/13.0

cd "$JUDGE_SUBMISSION_DIR"
export PYTHONPATH="$JUDGE_TRUSTED_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$JUDGE_SUBMISSION_DIR/.venv/bin/python" -u - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; refusing CPU evaluation.")

torch.ones(1, device="cuda").sum().item()
print("GPU ready:", torch.cuda.get_device_name(0), flush=True)
PY

"$JUDGE_SUBMISSION_DIR/.venv/bin/python" -u -m judge.run_task \
  --task "$JUDGE_TASK_ID" \
  --submission "$JUDGE_SUBMISSION_DIR" \
  --output "$JUDGE_OUTPUT_DIR/result.json"
