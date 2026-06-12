#!/usr/bin/env python3
"""
Analyze why VQAScore and VLMJudge produce frozen scores.

Instead of running inference, analyze the scoring logic itself.
"""

import math

def sigmoid(x):
    """Sigmoid function"""
    return 1.0 / (1.0 + math.exp(-x))

print("=" * 80)
print("Analysis: Why VQAScore and VLMJudge produce frozen scores")
print("=" * 80)

print("\n## Problem Diagnosis")
print("-" * 80)

print("""
Both VQAScore and VLMJudge use the same scoring pattern:

1. Score image_a individually → score_a (e.g., 0.5 on 0-1 scale for VQA, or 5.0 on 1-10 scale for VLM)
2. Score image_b individually → score_b (e.g., 0.5 or 5.0)  
3. Compute difference: diff = score_a - score_b
4. Apply sigmoid(diff) to get final score

The issue: If the model gives similar scores to all images in a group (because
they all depict the same prompt and have similar quality), then:
  - score_a ≈ score_b
  - diff ≈ 0
  - sigmoid(0) = 0.5
""")

print("\n## Example Calculations")
print("-" * 80)

print("\n### VQAScore (scores in [0.0, 1.0] range)")
print("If all images get parsed as neutral (0.5) because model responses are ambiguous:")

scenarios_vqa = [
    ("Image A", 0.5, "Image B", 0.5),
    ("Image A", 0.55, "Image B", 0.52),
    ("Image A", 0.48, "Image B", 0.50),
]

for name_a, score_a, name_b, score_b in scenarios_vqa:
    diff = score_a - score_b
    result = sigmoid(diff)
    print(f"  {name_a}: {score_a:.2f}, {name_b}: {score_b:.2f} → diff={diff:+.3f} → sigmoid={result:.6f}")

print("\n### VLMJudge (scores in [1.0, 10.0] range, scaled in sigmoid)")
print("If all images get rated around 5.0 (neutral on 1-10 scale):")

scenarios_vlm = [
    ("Image A", 5.0, "Image B", 5.0),
    ("Image A", 5.5, "Image B", 5.0),
    ("Image A", 6.0, "Image B", 5.0),
    ("Image A", 7.0, "Image B", 5.0),
]

for name_a, score_a, name_b, score_b in scenarios_vlm:
    diff = score_a - score_b
    # VLMJudge uses scaled sigmoid: sigmoid(diff / 3.0)
    scaled_diff = diff / 3.0
    result = sigmoid(scaled_diff)
    print(f"  {name_a}: {score_a:.1f}, {name_b}: {score_b:.1f} → diff={diff:+.1f} → scaled={scaled_diff:+.3f} → sigmoid={result:.6f}")

print("\n## Root Causes")
print("-" * 80)

print("""
1. **VQAScore**: Model responses might not contain clear yes/no indicators
   - Ambiguous responses get parsed as 0.5 (neutral)
   - If model says "it depends" or gives nuanced answers, _parse_response() defaults to 0.5
   - Solution: Check actual model responses; might need better parsing or different prompts

2. **VLMJudge**: Model might be giving ratings clustered around 5.0
   - If model rates all images in a group as "5" (decent quality), differences are small
   - sigmoid(small_diff) ≈ 0.5
   - Solution: Use more discriminative prompts or adjust sigmoid scaling

3. **Fundamental Issue**: Independent scoring doesn't work well for same-prompt images
   - All images depict the same prompt (e.g., "mermaid Art Nouveau portrait")
   - Model might rate them all as "7/10 quality" because they're all competent renditions
   - Tiny differences (7.0 vs 7.2) get lost in sigmoid normalization
""")

print("\n## Potential Fixes")
print("-" * 80)

print("""
### Option 1: Direct Comparison Prompts (RECOMMENDED)
Instead of scoring each image independently, show BOTH images to the model:

VLMJudge new prompt:
  "Compare these two images. Which one has better overall quality? 
   Answer with 'first' or 'second', then explain why."

Then parse the response for "first" vs "second" and convert to a score.
This would give: 
  - 0.8-0.9 if model prefers first image
  - 0.5 if model says "they're equal" 
  - 0.1-0.2 if model prefers second image

### Option 2: Adjust Sigmoid Scaling
Make sigmoid more sensitive to small differences:

Current VQAScore: sigmoid(diff) where diff is in [-1, 1]
Better: sigmoid(diff * 5) to amplify small differences

Current VLMJudge: sigmoid(diff / 3.0) where diff is in [-9, 9]  
Better: sigmoid(diff / 1.5) to be more sensitive

### Option 3: Better Response Parsing
For VQAScore, if model is giving nuanced responses instead of "yes/no",
extract more signal from the text:
  - Count positive adjectives vs negative adjectives
  - Look for comparatives ("better than", "worse than")
  - Extract confidence indicators ("definitely", "probably", "maybe")

### Option 4: Different Questions
Ask questions that force the model to discriminate more:
  - "On a scale of 1-10, how aesthetically pleasing is this image?"
  - "Would a professional photographer rate this as publication-quality? Explain."
  - "What are the main quality issues in this image?"
""")

print("\n## Recommended Action")
print("-" * 80)

print("""
1. **Immediate fix**: Switch to direct comparison prompts (Option 1)
   - Modify VLMJudge to take both images and ask "which is better?"
   - This is how human judges work and how PickScore/HPSv2 work internally

2. **Quick workaround**: Adjust sigmoid scaling (Option 2)  
   - Change VQAScore: sigmoid(diff * 5) instead of sigmoid(diff)
   - Change VLMJudge: sigmoid(diff / 1.5) instead of sigmoid(diff / 3.0)

3. **Add debug logging**: Already done!
   - Run a real match with the debug script to see actual model responses
   - This will confirm whether the issue is parsing or model behavior
""")

print("=" * 80)
