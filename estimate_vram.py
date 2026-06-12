#!/usr/bin/env python3
"""
Estimate VRAM/memory footprint for the 6-scorer ensemble.

Provides estimates for both CUDA (4bit quantization) and MPS (fp16) configurations.
"""

print("=" * 80)
print("VRAM/Memory Footprint Estimation for 6-Scorer Ensemble")
print("=" * 80)

# Model sizes (approximate, based on parameter counts and precision)
models = {
    "InstructBLIP (Salesforce/instructblip-flan-t5-xl)": {
        "params": "3B",
        "fp16_gb": 6.0,
        "4bit_gb": 1.5,
        "notes": "Shared by VQAScore and VLMJudge"
    },
    "PickScore (yuvalkirstain/PickScore_v1)": {
        "params": "400M",
        "fp16_gb": 0.8,
        "4bit_gb": 0.2,
        "notes": "CLIP-based, fine-tuned on preferences"
    },
    "HPSv2 (adams-story/HPSv2-hf)": {
        "params": "400M",
        "fp16_gb": 0.8,
        "4bit_gb": 0.2,
        "notes": "CLIP-based, human preference model"
    },
    "ImageReward (THUDM/ImageReward)": {
        "params": "400M",
        "fp16_gb": 1.0,
        "4bit_gb": 0.3,
        "notes": "BLIP-based reward model"
    },
    "CLIP (openai/clip-vit-large-patch14)": {
        "params": "428M",
        "fp16_gb": 0.9,
        "4bit_gb": 0.25,
        "notes": "NEW - For prompt-image alignment"
    }
}

print("\n## Individual Model Sizes")
print("-" * 80)
print(f"{'Model':<50} {'Params':<8} {'FP16 (GB)':<12} {'4bit (GB)':<12}")
print("-" * 80)

total_fp16 = 0.0
total_4bit = 0.0

for model_name, specs in models.items():
    print(f"{model_name:<50} {specs['params']:<8} {specs['fp16_gb']:<12.2f} {specs['4bit_gb']:<12.2f}")
    print(f"  └─ {specs['notes']}")
    total_fp16 += specs['fp16_gb']
    total_4bit += specs['4bit_gb']

print("-" * 80)
print(f"{'TOTAL (without overhead)':<50} {'':<8} {total_fp16:<12.2f} {total_4bit:<12.2f}")

# Add overhead for activations, gradients, etc.
overhead_factor = 1.3  # 30% overhead for activations, intermediate tensors, etc.
total_fp16_with_overhead = total_fp16 * overhead_factor
total_4bit_with_overhead = total_4bit * overhead_factor

print(f"{'TOTAL (with 30% overhead)':<50} {'':<8} {total_fp16_with_overhead:<12.2f} {total_4bit_with_overhead:<12.2f}")

print("\n" + "=" * 80)
print("Configuration-Specific Estimates")
print("=" * 80)

print("\n### Mac (MPS) - FP16 Configuration")
print("-" * 80)
print(f"""
Total Memory: ~{total_fp16_with_overhead:.1f} GB
- InstructBLIP (fp16):  {models['InstructBLIP (Salesforce/instructblip-flan-t5-xl)']['fp16_gb']:.1f} GB (shared by VQA + VLMJudge)
- PickScore (fp16):     {models['PickScore (yuvalkirstain/PickScore_v1)']['fp16_gb']:.1f} GB
- HPSv2 (fp16):         {models['HPSv2 (adams-story/HPSv2-hf)']['fp16_gb']:.1f} GB
- ImageReward (fp16):   {models['ImageReward (THUDM/ImageReward)']['fp16_gb']:.1f} GB
- CLIP (fp16):          {models['CLIP (openai/clip-vit-large-patch14)']['fp16_gb']:.1f} GB (NEW)
- Overhead (30%):       {(total_fp16_with_overhead - total_fp16):.1f} GB

Previous (5 scorers):   ~{(total_fp16 - models['CLIP (openai/clip-vit-large-patch14)']['fp16_gb']) * overhead_factor:.1f} GB
NEW (6 scorers):        ~{total_fp16_with_overhead:.1f} GB
Added:                  +{models['CLIP (openai/clip-vit-large-patch14)']['fp16_gb'] * overhead_factor:.1f} GB

Recommendation: Requires ~{total_fp16_with_overhead:.0f}GB unified memory (Mac M1/M2 with 16GB may struggle)
""")

print("\n### Linux/CUDA - 4bit Configuration")
print("-" * 80)
print(f"""
Total VRAM: ~{total_4bit_with_overhead:.1f} GB
- InstructBLIP (4bit):  {models['InstructBLIP (Salesforce/instructblip-flan-t5-xl)']['4bit_gb']:.1f} GB (shared by VQA + VLMJudge)
- PickScore (fp16):     {models['PickScore (yuvalkirstain/PickScore_v1)']['4bit_gb']:.1f} GB
- HPSv2 (fp16):         {models['HPSv2 (adams-story/HPSv2-hf)']['4bit_gb']:.1f} GB
- ImageReward (fp16):   {models['ImageReward (THUDM/ImageReward)']['4bit_gb']:.1f} GB
- CLIP (fp16):          {models['CLIP (openai/clip-vit-large-patch14)']['4bit_gb']:.1f} GB (NEW)
- Overhead (30%):       {(total_4bit_with_overhead - total_4bit):.1f} GB

Previous (5 scorers):   ~{(total_4bit - models['CLIP (openai/clip-vit-large-patch14)']['4bit_gb']) * overhead_factor:.1f} GB
NEW (6 scorers):        ~{total_4bit_with_overhead:.1f} GB
Added:                  +{models['CLIP (openai/clip-vit-large-patch14)']['4bit_gb'] * overhead_factor:.1f} GB

Recommendation: Fits comfortably on most modern GPUs (RTX 3060 12GB+, RTX 4090 24GB, A6000 48GB)
""")

print("\n" + "=" * 80)
print("Impact Analysis")
print("=" * 80)

clip_overhead_fp16 = models['CLIP (openai/clip-vit-large-patch14)']['fp16_gb'] * overhead_factor
clip_overhead_4bit = models['CLIP (openai/clip-vit-large-patch14)']['4bit_gb'] * overhead_factor

print(f"""
### CLIP Addition Impact:

**MPS/FP16 (Mac)**:
- Additional memory: +{clip_overhead_fp16:.1f} GB (~{(clip_overhead_fp16 / (total_fp16_with_overhead - clip_overhead_fp16)) * 100:.0f}% increase)
- Total goes from ~{(total_fp16_with_overhead - clip_overhead_fp16):.0f}GB to ~{total_fp16_with_overhead:.0f}GB
- CLIP is relatively lightweight (~10% of total footprint)

**CUDA/4bit (Linux)**:
- Additional VRAM: +{clip_overhead_4bit:.1f} GB (~{(clip_overhead_4bit / (total_4bit_with_overhead - clip_overhead_4bit)) * 100:.0f}% increase)
- Total goes from ~{(total_4bit_with_overhead - clip_overhead_4bit):.0f}GB to ~{total_4bit_with_overhead:.0f}GB
- Negligible impact on modern GPUs

### Benefits vs Cost:

**Memory Cost**: +{clip_overhead_fp16:.1f}GB (MPS) / +{clip_overhead_4bit:.1f}GB (CUDA)

**Value Added**:
- Only scorer measuring prompt fidelity (all others measure aesthetics/quality)
- Fast inference (~50ms per comparison, faster than InstructBLIP-based scorers)
- Provides crucial signal for text-to-image evaluation
- 20% weight in ensemble reflects its unique contribution

**Conclusion**: The memory cost is justified by CLIPScore's unique signal.
It's the only scorer that directly measures "does this image match the prompt?"
rather than just "is this image pretty?". This is critical for evaluating
text-to-image generation systems.
""")

print("=" * 80)
