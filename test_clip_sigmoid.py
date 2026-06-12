#!/usr/bin/env python3
"""
Determine optimal sigmoid amplification for CLIPScore.

CLIP similarities for same-prompt images are typically in 0.20-0.35 range
with small differences of 0.01-0.03 between images.
"""

import math

def sigmoid(x, amplification=1.0):
    """Apply sigmoid with amplification factor"""
    return 1.0 / (1.0 + math.exp(-x * amplification))

print("=" * 80)
print("CLIPScore Sigmoid Amplification Analysis")
print("=" * 80)

print("\n## Typical CLIP Similarity Scores")
print("-" * 80)
print("""
For same-prompt images (e.g., all "mermaid Art Nouveau portrait"):
- Similarity range: 0.20 - 0.35
- Typical differences: 0.01 - 0.03
- Large differences: 0.05 - 0.08 (rare, indicates significant quality gap)
""")

print("\n## Testing Different Amplification Factors")
print("-" * 80)

test_diffs = [0.01, 0.02, 0.03, 0.05, 0.08]
amplifications = [10, 15, 20, 25, 30]

print(f"\n{'Diff':>6} | " + " | ".join([f"×{a:2d}={sigmoid(0.03, a):.4f}" for a in amplifications]))
print("-" * 80)

for diff in test_diffs:
    values = [sigmoid(diff, a) for a in amplifications]
    row = f"{diff:>6.3f} | " + " | ".join([f"{v:>11.4f}" for v in values])
    print(row)

print("\n## Analysis by Amplification Factor")
print("-" * 80)

for amp in amplifications:
    print(f"\n### Amplification: ×{amp}")
    print(f"{'Diff':>8} | {'Sigmoid':>10} | {'Interpretation'}")
    print("-" * 50)
    
    for diff in [0.01, 0.02, 0.03, 0.05]:
        sig = sigmoid(diff, amp)
        
        if sig < 0.52:
            interp = "Near tie"
        elif sig < 0.58:
            interp = "Weak preference"
        elif sig < 0.70:
            interp = "Moderate preference"
        elif sig < 0.85:
            interp = "Strong preference"
        else:
            interp = "Decisive"
        
        print(f"{diff:>8.3f} | {sig:>10.4f} | {interp}")

print("\n" + "=" * 80)
print("Recommendation")
print("=" * 80)

print("""
### Recommended Amplification: ×20

Reasoning:
1. diff=0.01 → 0.5498 (weak preference, appropriate for tiny difference)
2. diff=0.02 → 0.5987 (moderate, good discrimination)
3. diff=0.03 → 0.6457 (strong preference, but not saturated)
4. diff=0.05 → 0.7311 (decisive, for clear winners)
5. diff=0.08 → 0.8176 (very strong, rare cases)

This provides:
- Good sensitivity to small differences (0.01-0.03 range)
- Doesn't saturate too early (stays below 0.90 for typical diffs)
- Creates meaningful spread without over-amplification
- Aligns with VQAScore's ×5 philosophy (similar scale after accounting for input ranges)

Comparison with other scorers:
- VQAScore uses ×5.0 on [0.0, 1.0] range differences
- VLMJudge uses /1.5 on [1.0, 10.0] range differences (effectively ×0.67)
- CLIPScore uses ×20 on [0.0, 0.1] typical range differences
- All achieve similar dynamic range in final scores

Alternative: ×25 if you want more aggressive discrimination, but risks
over-emphasizing measurement noise in CLIP similarities.
""")

print("=" * 80)
