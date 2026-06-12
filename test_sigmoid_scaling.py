#!/usr/bin/env python3
"""
Test the improved sigmoid scaling for VQAScore and VLMJudge.

Demonstrates how the new scaling prevents score collapse.
"""

import math

def sigmoid_old(x):
    """Original sigmoid (VQAScore)"""
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_new_vqa(x):
    """New sigmoid for VQAScore (amplified by 5x)"""
    return 1.0 / (1.0 + math.exp(-x * 5.0))

def sigmoid_old_vlm(x):
    """Original sigmoid for VLMJudge (scaled by /3.0)"""
    return 1.0 / (1.0 + math.exp(-x / 3.0))

def sigmoid_new_vlm(x):
    """New sigmoid for VLMJudge (scaled by /1.5)"""
    return 1.0 / (1.0 + math.exp(-x / 1.5))

print("=" * 80)
print("Sigmoid Scaling Comparison")
print("=" * 80)

print("\n### VQAScore (score range: 0.0 - 1.0)")
print("-" * 80)
print(f"{'Diff':>8} | {'Old Sigmoid':>12} | {'New Sigmoid (×5)':>17} | {'Improvement':>12}")
print("-" * 80)

vqa_diffs = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
for diff in vqa_diffs:
    old = sigmoid_old(diff)
    new = sigmoid_new_vqa(diff)
    improvement = new - old
    print(f"{diff:+8.3f} | {old:12.6f} | {new:17.6f} | {improvement:+12.6f}")

print("\n### VLMJudge (score range: 1.0 - 10.0)")
print("-" * 80)
print(f"{'Diff':>8} | {'Old (÷3.0)':>12} | {'New (÷1.5)':>12} | {'Improvement':>12}")
print("-" * 80)

vlm_diffs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
for diff in vlm_diffs:
    old = sigmoid_old_vlm(diff)
    new = sigmoid_new_vlm(diff)
    improvement = new - old
    print(f"{diff:+8.1f} | {old:12.6f} | {new:12.6f} | {improvement:+12.6f}")

print("\n" + "=" * 80)
print("Analysis")
print("=" * 80)

print("""
### VQAScore Improvements:
- Old sigmoid collapsed small differences to ~0.5
  Example: diff=0.05 → 0.5125 (barely different from 0.5)
- New sigmoid preserves variance:
  Example: diff=0.05 → 0.5622 (clearly different from 0.5)
  
This means if image_a scores 0.55 and image_b scores 0.50:
  - OLD: sigmoid(0.05) = 0.5125  (almost a tie)
  - NEW: sigmoid(0.05 × 5) = 0.5622  (clear preference for image_a)

### VLMJudge Improvements:
- Old sigmoid needed 3-point differences to be decisive  
  Example: diff=1.5 → 0.6225 (weak preference)
- New sigmoid is more sensitive:
  Example: diff=1.5 → 0.7311 (strong preference)
  
This means if image_a gets 6.5/10 and image_b gets 5.0/10:
  - OLD: sigmoid(1.5 / 3.0) = 0.6225  (weak)
  - NEW: sigmoid(1.5 / 1.5) = 0.7311  (strong)

### Expected Impact on Tournament Results:
- VQAScore will no longer be frozen at exactly 0.5
- VLMJudge will show more variance across images
- Both scorers will contribute meaningful signal to the ensemble
- Standard deviation should increase to ~0.08-0.15 (similar to PickScore)
""")

print("=" * 80)
