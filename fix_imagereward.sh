#!/usr/bin/env bash
#
# fix_imagereward.sh — install the native ImageReward package and verify the
# project's ImageReward scorer can load + score with it.
#
# Run from the project root (the folder containing main.py). Safe to re-run.
# It does NOT downgrade torch/transformers (image-reward only needs
# transformers>=4.27.4); it adds fairscale, timm, diffusers, datasets, etc.
#
# Usage:
#   ./fix_imagereward.sh
#   (or: bash fix_imagereward.sh)
#
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJ"

# ── Pick the Python interpreter ─────────────────────────────────────────────
# Priority: currently-active venv → .venv312 (Intel iMac) → .venv_arm (Apple
# Silicon) → system python3.
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PY="${VIRTUAL_ENV}/bin/python"
elif [[ -x ".venv312/bin/python" ]]; then
  PY="$PROJ/.venv312/bin/python"
elif [[ -x ".venv_arm/bin/python" ]]; then
  PY="$PROJ/.venv_arm/bin/python"
else
  PY="$(command -v python3)"
fi

echo "============================================================"
echo " ImageReward install + verify"
echo "============================================================"
echo "Project : $PROJ"
echo "Python  : $PY"
"$PY" --version
echo

# ── 1. Install ──────────────────────────────────────────────────────────────
# setuptools<81      : provides `pkg_resources`, which `clip` imports. Python
#                      3.12 venvs ship WITHOUT setuptools, and setuptools >=81
#                      removed pkg_resources — so we pin <81 to guarantee it.
# diffusers>=0.29,<0.30 : image-reward imports diffusers (for its ReFL training
#                      module). The latest diffusers needs torch>=2.4
#                      (torch.xpu / torch.distributed.device_mesh); 0.29.x is the
#                      version that works with torch 2.2/2.3 AND a modern
#                      huggingface_hub. diffusers is NOT used for scoring.
# image-reward       : the reward model (THUDM/ImageReward)
# clip-anytorch      : provides the `clip` module ImageReward imports
#                      (the package does NOT pull this in automatically)
# NOTE: image-reward pins timm==0.6.13. pip may print a conflict warning about
#       open-clip-torch wanting a newer timm — this is harmless; open_clip and
#       the tournament still import fine, and CLIP isn't used during a run.
echo "==> [1/2] Installing setuptools<81 + diffusers 0.29.x + image-reward + clip-anytorch ..."
"$PY" -m pip install --upgrade pip >/dev/null 2>&1 || true
"$PY" -m pip install "setuptools<81" "diffusers>=0.29,<0.30" image-reward clip-anytorch

echo
# ── 2. Verify ─────────────────────────────────────────────────────────────--
echo "==> [2/2] Verifying ImageReward loads + scores via the project scorer ..."
echo "    (first run downloads the ImageReward checkpoint, ~2 GB)"
"$PY" verify_imagereward.py

echo
echo "============================================================"
echo " Done. If you saw 'PASS', ImageReward is ready — re-run:"
echo
echo "   $PY main.py tournament \\"
echo "     --data        \"data/Metric Evaluation Text to Image.xlsx\" \\"
echo "     --config      config.yaml \\"
echo "     --sheet-range \"Mermaid Metric:Underwatercity Metric\" \\"
echo "     --cell-range  \"E2:E11\" \\"
echo "     --export-csv  results/evaluation_scores.csv \\"
echo "     --output      results/"
echo "============================================================"
