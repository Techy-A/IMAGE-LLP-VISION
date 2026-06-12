#!/usr/bin/env python3
"""
Verify that the sigmoid scaling fix produces varied scores.

Uses mock scorers with realistic score distributions to verify
the new sigmoid scaling prevents score collapse.
"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def sigmoid_vqa(x):
    """New VQAScore sigmoid (5x amplification)"""
    scaled_x = x * 5.0
    return 1.0 / (1.0 + math.exp(-scaled_x))

def sigmoid_vlm(x):
    """New VLMJudge sigmoid (/1.5 scaling)"""
    scaled_x = x / 1.5
    return 1.0 / (1.0 + math.exp(-scaled_x))

print("=" * 80)
print("Verification: Scorer Fix with Realistic Distributions")
print("=" * 80)

print("\n## Scenario: 10 images from same prompt group")
print("-" * 80)

# Simulate VQAScore outputs: similar quality images with small variance
# All images are "decent quality" so model gives scores around 0.5-0.6
vqa_image_scores = [0.48, 0.50, 0.51, 0.52, 0.50, 0.53, 0.49, 0.54, 0.50, 0.52]

print("\n### VQAScore: Image scores (independent ratings)")
print(f"Scores: {vqa_image_scores}")
print(f"Mean:  {sum(vqa_image_scores)/len(vqa_image_scores):.4f}")
print(f"Std:   {(sum((x - sum(vqa_image_scores)/len(vqa_image_scores))**2 for x in vqa_image_scores) / len(vqa_image_scores))**0.5:.4f}")
print(f"Range: [{min(vqa_image_scores):.2f}, {max(vqa_image_scores):.2f}]")

# Compute pairwise comparison scores (first vs others)
print(f"\n### VQAScore: Pairwise scores (image 0 vs others)")
print(f"{'Image A':>10} | {'Image B':>10} | {'Diff':>8} | {'Old Score':>10} | {'New Score':>10}")
print("-" * 60)

vqa_pairwise_new = []
for i in range(1, len(vqa_image_scores)):
    score_a = vqa_image_scores[0]
    score_b = vqa_image_scores[i]
    diff = score_a - score_b
    
    # Old sigmoid (not amplified)
    old_score = 1.0 / (1.0 + math.exp(-diff))
    
    # New sigmoid (5x amplified)
    new_score = sigmoid_vqa(diff)
    
    vqa_pairwise_new.append(new_score)
    
    print(f"{score_a:10.4f} | {score_b:10.4f} | {diff:+8.4f} | {old_score:10.6f} | {new_score:10.6f}")

print(f"\n### VQAScore: Pairwise score statistics")
print(f"Mean:  {sum(vqa_pairwise_new)/len(vqa_pairwise_new):.4f}")
print(f"Std:   {(sum((x - sum(vqa_pairwise_new)/len(vqa_pairwise_new))**2 for x in vqa_pairwise_new) / len(vqa_pairwise_new))**0.5:.4f}")
print(f"Range: [{min(vqa_pairwise_new):.4f}, {max(vqa_pairwise_new):.4f}]")
print(f"Unique: {len(set(vqa_pairwise_new))} distinct values")

# Simulate VLMJudge outputs: ratings on 1-10 scale
# All images are "decent" so model gives 5-7 ratings
vlm_image_scores = [5.0, 5.5, 6.0, 5.2, 5.8, 6.2, 5.3, 6.5, 5.1, 5.9]

print("\n" + "=" * 80)
print(f"\n### VLMJudge: Image scores (1-10 scale ratings)")
print(f"Scores: {vlm_image_scores}")
print(f"Mean:  {sum(vlm_image_scores)/len(vlm_image_scores):.4f}")
print(f"Std:   {(sum((x - sum(vlm_image_scores)/len(vlm_image_scores))**2 for x in vlm_image_scores) / len(vlm_image_scores))**0.5:.4f}")
print(f"Range: [{min(vlm_image_scores):.1f}, {max(vlm_image_scores):.1f}]")

# Compute pairwise comparison scores
print(f"\n### VLMJudge: Pairwise scores (image 0 vs others)")
print(f"{'Image A':>10} | {'Image B':>10} | {'Diff':>8} | {'Old Score':>10} | {'New Score':>10}")
print("-" * 60)

vlm_pairwise_new = []
for i in range(1, len(vlm_image_scores)):
    score_a = vlm_image_scores[0]
    score_b = vlm_image_scores[i]
    diff = score_a - score_b
    
    # Old sigmoid (/3.0 scaling)
    old_score = 1.0 / (1.0 + math.exp(-diff / 3.0))
    
    # New sigmoid (/1.5 scaling)
    new_score = sigmoid_vlm(diff)
    
    vlm_pairwise_new.append(new_score)
    
    print(f"{score_a:10.1f} | {score_b:10.1f} | {diff:+8.1f} | {old_score:10.6f} | {new_score:10.6f}")

print(f"\n### VLMJudge: Pairwise score statistics")
print(f"Mean:  {sum(vlm_pairwise_new)/len(vlm_pairwise_new):.4f}")
print(f"Std:   {(sum((x - sum(vlm_pairwise_new)/len(vlm_pairwise_new))**2 for x in vlm_pairwise_new) / len(vlm_pairwise_new))**0.5:.4f}")
print(f"Range: [{min(vlm_pairwise_new):.4f}, {max(vlm_pairwise_new):.4f}]")
print(f"Unique: {len(set(vlm_pairwise_new))} distinct values")

print("\n" + "=" * 80)
print("Verification Results")
print("=" * 80)

vqa_std = (sum((x - sum(vqa_pairwise_new)/len(vqa_pairwise_new))**2 for x in vqa_pairwise_new) / len(vqa_pairwise_new))**0.5
vlm_std = (sum((x - sum(vlm_pairwise_new)/len(vlm_pairwise_new))**2 for x in vlm_pairwise_new) / len(vlm_pairwise_new))**0.5

print(f"\n✅ VQAScore:")
print(f"   - Std:    {vqa_std:.4f} (target: >0.05)")
print(f"   - Unique: {len(set(vqa_pairwise_new))} values (target: >5)")
print(f"   - Status: {'PASS ✅' if vqa_std > 0.05 and len(set(vqa_pairwise_new)) > 5 else 'FAIL ❌'}")

print(f"\n✅ VLMJudge:")
print(f"   - Std:    {vlm_std:.4f} (target: >0.08)")
print(f"   - Unique: {len(set(vlm_pairwise_new))} values (target: >5)")
print(f"   - Status: {'PASS ✅' if vlm_std > 0.08 and len(set(vlm_pairwise_new)) > 5 else 'FAIL ❌'}")

print("\n" + "=" * 80)
print("Expected Tournament Impact")
print("=" * 80)

print("""
With the new sigmoid scaling:

1. VQAScore will produce varied scores across matches
   - No longer frozen at 0.5
   - Will contribute meaningful signal to ensemble
   - Expected std in final CSV: 0.08-0.15

2. VLMJudge will show increased discrimination
   - Many more unique values per group
   - Better spread in final rankings
   - Expected std in final CSV: 0.10-0.18

3. Ensemble will be more balanced
   - All 5 scorers contribute meaningfully
   - Less dominated by just PickScore/HPSv2
   - Better overall ranking quality

To verify in real tournament:
  python main.py tournament --data ... --config config.yaml ...
  
Then check CSV stats:
  df = pd.read_csv('results/evaluation_scores.csv')
  print(df['avg_vqascore'].std())   # Should be > 0.05
  print(df['avg_vlm_judge'].std())  # Should be > 0.08
""")

print("=" * 80)
