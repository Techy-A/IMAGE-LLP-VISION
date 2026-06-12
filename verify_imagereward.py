#!/usr/bin/env python3
"""
verify_imagereward.py — confirm the project's ImageReward scorer can load the
native `image-reward` package and produce valid scores.

Exercised path: src/scoring/image_reward.py -> _load_native(), which now tries
the native model names ("ImageReward-v1.0" / "ImageReward") rather than the
un-loadable HF repo id. Exits non-zero on failure.

Run from the project root:  python verify_imagereward.py
"""

import logging
import sys

from PIL import Image

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main() -> int:
    try:
        from src.scoring.image_reward import ImageReward
    except Exception as e:
        print(f"FAIL: cannot import the project ImageReward scorer: {e}")
        return 1

    # Verify on CPU for determinism (the tournament itself uses auto/MPS/CUDA).
    scorer = ImageReward(device="cpu")
    try:
        scorer.load("THUDM/ImageReward")  # id from config.yaml; loader falls back to native names
    except Exception as e:
        print(f"FAIL: ImageReward.load() raised: {e}")
        print("      Is the 'image-reward' package installed?  pip install image-reward")
        return 1

    backend = "native" if getattr(scorer, "using_native", False) else "huggingface"
    print(f"Loaded ImageReward via: {backend}")
    if backend != "native":
        print("WARN: native library not used — the HuggingFace path for "
              "THUDM/ImageReward is not loadable; install 'image-reward'.")

    # Build two simple test images.
    img = Image.new("RGB", (224, 224), color="skyblue")
    other = Image.new("RGB", (224, 224), color="black")

    try:
        self_score = scorer.compare(img, img)     # same image vs itself
        diff_score = scorer.compare(img, other)   # different images
    except Exception as e:
        print(f"FAIL: ImageReward.compare() raised: {e}")
        return 1

    print(f"compare(img, img)   = {self_score:.4f}   (expected ~0.5)")
    print(f"compare(img, other) = {diff_score:.4f}   (any value in [0,1])")

    ok = (
        0.0 <= self_score <= 1.0
        and 0.0 <= diff_score <= 1.0
        and abs(self_score - 0.5) < 1e-3   # identical inputs -> neutral
    )
    if not ok:
        print("FAIL: scores out of range or self-comparison not neutral.")
        return 1

    print("\nPASS: ImageReward installed, loaded (native), and scoring correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
