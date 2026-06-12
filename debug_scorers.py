#!/usr/bin/env python3
"""
Debug script to diagnose VQAScore and VLMJudge frozen scores.

Runs just 3 matches with verbose logging to see raw scorer outputs.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Set logging to INFO to see debug statements
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

from src.csv_loader import CSVLoader
from src.config_manager import ConfigManager
from src.models.instructblip_adapter import InstructBLIPAdapter
from src.scoring.vqascore import VQAScore
from src.scoring.vlm_judge import VLMJudge
from src.scoring.pickscore import PickScore
from src.scoring.hpsv2 import HPSv2


def main():
    print("=" * 80)
    print("DEBUG: VQAScore and VLMJudge Raw Outputs")
    print("=" * 80)
    
    # Load config
    config_path = "config.yaml"
    cfg = ConfigManager(config_path).load_config()
    
    # Load a few images
    data_path = "data/Metric Evaluation Text to Image.xlsx"
    csv_loader = CSVLoader()
    images = csv_loader.load_metadata(
        data_path,
        sheet_range="Mermaid Metric:Mermaid Metric",  # Just one sheet
        cell_range="E2:E4"  # Just 3 images
    )
    
    print(f"\n📊 Loaded {len(images)} images from Mermaid Metric group")
    for img in images:
        print(f"   • {img.id}")
    
    # Load InstructBLIP (shared)
    print(f"\n🔧 Loading InstructBLIP...")
    instructblip_cfg = cfg.models.get("instructblip")
    instructblip = InstructBLIPAdapter()
    instructblip.load({
        "hf_id": instructblip_cfg.hf_id,
        "quantization": instructblip_cfg.quantization
    })
    print(f"✅ InstructBLIP loaded")
    
    # Create scorers
    vqa = VQAScore(instructblip)
    vlm = VLMJudge(instructblip)
    
    # Also load PickScore for comparison
    ps_cfg = cfg.models.get("pickscore")
    ps = PickScore(device="auto")
    ps.load(ps_cfg.hf_id if ps_cfg else "yuvalkirstain/PickScore_v1")
    
    # Load images as PIL
    from src.image_loader import ImageLoader
    loader = ImageLoader()
    
    pil_images = []
    for img in images:
        pil_img = loader.load_image(img.path)
        if pil_img:
            pil_images.append((img.id, pil_img))
    
    print(f"\n📷 Loaded {len(pil_images)} PIL images")
    
    # Run 3 comparisons
    print(f"\n🔍 Running 3 comparisons with detailed logging...\n")
    
    for i in range(min(3, len(pil_images) - 1)):
        img_a_id, img_a = pil_images[i]
        img_b_id, img_b = pil_images[i + 1]
        
        print(f"\n{'=' * 80}")
        print(f"MATCH #{i+1}: {img_a_id} vs {img_b_id}")
        print(f"{'=' * 80}")
        
        # VQAScore
        print(f"\n--- VQAScore ---")
        vqa_score = vqa.compare(img_a, img_b)
        print(f"FINAL VQAScore: {vqa_score:.6f}")
        
        # VLMJudge
        print(f"\n--- VLMJudge ---")
        vlm_score = vlm.compare(img_a, img_b)
        print(f"FINAL VLMJudge: {vlm_score:.6f}")
        
        # PickScore (for comparison)
        print(f"\n--- PickScore (baseline) ---")
        ps_score = ps.compare(img_a, img_b)
        print(f"FINAL PickScore: {ps_score:.6f}")
        
        print(f"\n📊 Summary for match #{i+1}:")
        print(f"   VQAScore:  {vqa_score:.6f}")
        print(f"   VLMJudge:  {vlm_score:.6f}")
        print(f"   PickScore: {ps_score:.6f}")
    
    print(f"\n{'=' * 80}")
    print(f"DEBUG COMPLETE")
    print(f"{'=' * 80}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
