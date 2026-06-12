"""
Test script for loading multi-sheet XLSX file with image paths.

This demonstrates how to load images from cells E2:E11 across
sheets 1-5 in the Metric Evaluation Text to Image.xlsx file.
"""

import logging
from src.csv_loader import CSVLoader

# Setup logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # Initialize loader
    loader = CSVLoader()
    
    # Path to the XLSX file
    xlsx_path = "data/Metric Evaluation Text to Image.xlsx"
    
    print(f"\n{'='*70}")
    print("Loading images from multi-sheet XLSX file")
    print(f"{'='*70}\n")
    
    # Load with multi-sheet mode
    # - sheet_range: "Sheet1:Sheet5" means read from Sheet1 through Sheet5
    # - cell_range: "E2:E11" means read cells E2, E3, E4, ..., E11 from each sheet
    metadata_list = loader.load_metadata(
        csv_path=xlsx_path,
        sheet_range="Mermaid Metric:Underwatercity Metric",
        cell_range="E2:E11"
    )
    
    print(f"\n{'='*70}")
    print(f"Results: Loaded {len(metadata_list)} images")
    print(f"{'='*70}\n")
    
    # Display first few records
    if metadata_list:
        print("First 5 records:")
        print("-" * 70)
        for i, meta in enumerate(metadata_list[:5], 1):
            print(f"\n{i}. ID: {meta.id}")
            print(f"   Path: {meta.path}")
            print(f"   Prompt: {meta.prompt[:60] if meta.prompt else None}...")
            print(f"   Model: {meta.model}")
            print(f"   Source: {meta.attributes.get('source_sheet')} "
                  f"cell {meta.attributes.get('source_cell')}")
        
        if len(metadata_list) > 5:
            print(f"\n   ... and {len(metadata_list) - 5} more images")
    
    # Check for errors
    errors = loader.get_load_errors()
    if errors:
        print(f"\n{'='*70}")
        print(f"Errors encountered: {len(errors)}")
        print(f"{'='*70}\n")
        for identifier, error_msg in errors[:5]:
            print(f"- {identifier}: {error_msg}")
        if len(errors) > 5:
            print(f"\n  ... and {len(errors) - 5} more errors")
    
    print(f"\n{'='*70}")
    print("Expected: 50 images (5 sheets × 10 rows per sheet)")
    print(f"Actual:   {len(metadata_list)} images loaded")
    print(f"{'='*70}\n")
    
    # Alternative: Load single-sheet mode (for comparison)
    print("\n" + "="*70)
    print("Alternative: Single-sheet mode (if XLSX has standard columns)")
    print("="*70 + "\n")
    
    loader2 = CSVLoader()
    try:
        # This would work if the XLSX had 'id' and 'path' columns in first sheet
        metadata_single = loader2.load_metadata(csv_path=xlsx_path)
        print(f"Single-sheet mode loaded: {len(metadata_single)} images")
    except Exception as e:
        print(f"Single-sheet mode failed (expected if no 'id'/'path' columns): {e}")

if __name__ == "__main__":
    main()
