#!/usr/bin/env bash
# setup_cuda.sh — one-shot environment setup for the Linux/CUDA server
#
# Hardware target:  2× NVIDIA RTX A6000 (49 GB VRAM each), CUDA 12.8
# Python target:    3.10 – 3.12
#
# Usage (from the project root):
#   bash setup_cuda.sh
#
# What it does:
#   1. Creates a fresh venv at .venv_cuda/
#   2. Installs all dependencies from requirements-cuda.txt
#      (torch comes from the CUDA 12.8 wheel index)
#   3. Runs the unit-test suite to verify the install
#   4. Verifies ImageReward loads and scores
#
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJ"

# ── 1. Pick Python ────────────────────────────────────────────────────────
# Use python3.12 if available, else python3.11, else python3.10, else python3
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" &>/dev/null; then
    PY_BIN="$(command -v "$candidate")"
    break
  fi
done
if [[ -z "${PY_BIN:-}" ]]; then
  echo "❌ No Python 3 found in PATH. Install python3.10+ first." >&2
  exit 1
fi
echo "Using Python: $PY_BIN ($("$PY_BIN" --version 2>&1))"

# ── 2. Create venv ────────────────────────────────────────────────────────
VENV="$PROJ/.venv_cuda"
if [[ -d "$VENV" ]]; then
  echo "⚠️  .venv_cuda already exists — skipping creation (re-using)."
else
  echo "Creating venv at .venv_cuda ..."
  "$PY_BIN" -m venv "$VENV"
fi
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

# ── 3. Upgrade pip / setuptools ───────────────────────────────────────────
echo
echo "==> Upgrading pip ..."
"$PIP" install --upgrade pip

# ── 4. Install dependencies ───────────────────────────────────────────────
echo
echo "==> Installing from requirements-cuda.txt ..."
echo "    (torch CUDA wheel is ~2 GB — this will take a few minutes)"
"$PIP" install -r "$PROJ/requirements-cuda.txt"

# ── 5. Verify CUDA is visible ─────────────────────────────────────────────
echo
echo "==> Checking CUDA availability ..."
"$PY" - <<'PYEOF'
import torch, sys
if not torch.cuda.is_available():
    print("❌ CUDA not available after install!", file=sys.stderr)
    print("   Check your CUDA driver version with: nvidia-smi", file=sys.stderr)
    sys.exit(1)
n = torch.cuda.device_count()
print(f"✅ CUDA available: {n} GPU(s)")
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    print(f"   GPU {i}: {props.name}  {props.total_memory/1024**3:.1f} GB VRAM")
PYEOF

# ── 6. Run unit tests ─────────────────────────────────────────────────────
echo
echo "==> Running unit tests (expect 265 passed) ..."
"$VENV/bin/pytest" tests/ -q

# ── 7. Verify ImageReward ─────────────────────────────────────────────────
echo
echo "==> Verifying ImageReward scorer ..."
echo "    (first run downloads ~2 GB checkpoint)"
"$PY" verify_imagereward.py

echo
echo "============================================================"
echo " Setup complete. Activate the env and run the tournament:"
echo
echo "   source .venv_cuda/bin/activate"
echo
echo "   python main.py tournament \\"
echo "     --data        \"data/Metric Evaluation Text to Image.xlsx\" \\"
echo "     --config      config.yaml \\"
echo "     --sheet-range \"Mermaid Metric:Underwatercity Metric\" \\"
echo "     --cell-range  \"E2:E11\" \\"
echo "     --export-csv  results/evaluation_scores.csv \\"
echo "     --output      results/"
echo "============================================================"
