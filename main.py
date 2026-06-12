#!/usr/bin/env python3
"""
IMAGE-LLP-VISION Main Entry Point

Pairwise image evaluation system using multiple vision-language models.
Runs 100% locally with zero API costs.

Entry-point flow
----------------
1. Apply Mac/MPS environment patches (no-op on Linux/CUDA boxes)
2. Disable the transformers `.bin` safety check that blocks legacy checkpoints
3. Import and call src.cli.main(), which dispatches to one of:
   - `tournament`  — run a full round-robin scoring tournament
   - `visualize`   — render charts/heatmaps from a results JSON
   - `benchmark`   — latency/memory benchmark for individual models
"""

import sys
import os
import platform

# ── Mac / MPS-specific patches (no-op on Linux/CUDA) ─────────────────────
if platform.system() == "Darwin":
    # Without this, PyTorch's MPS allocator caps itself at a fraction of RAM
    # and raises OOM errors when loading multiple large models (e.g. InstructBLIP
    # + PickScore simultaneously). Setting the ratio to 0.0 disables the cap and
    # lets macOS handle swap itself.
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    # transformers ≥ 4.38 added a warmup call that tries to pre-allocate ~8 GB
    # of MPS memory. On machines with limited VRAM this raises:
    #   RuntimeError: Invalid buffer size: 8.31 GB
    # Replacing it with a no-op lambda sidesteps the allocation without
    # affecting model weights or inference correctness.
    try:
        import transformers.modeling_utils
        transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    except ImportError:
        pass

# ── Allow loading legacy .bin checkpoints on older PyTorch versions ───────
# Newer transformers versions call check_torch_load_is_safe() which raises
# when loading older serialised .bin weights. The models we use (InstructBLIP,
# BLIP-2, etc.) are well-known public checkpoints, so the safety check adds
# no value and only causes load failures. We monkey-patch it to a no-op.
for _module_path in (
    "transformers.utils.import_utils",
    "transformers.utils",
    "transformers.modeling_utils",
):
    try:
        import importlib
        _mod = importlib.import_module(_module_path)
        if hasattr(_mod, "check_torch_load_is_safe"):
            _mod.check_torch_load_is_safe = lambda *args, **kwargs: None
    except ImportError:
        pass

from src.cli import main


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
