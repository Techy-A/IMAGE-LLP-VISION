#!/usr/bin/env python3
"""Test per-group exports and visualizations."""

import sys
import json
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from verify_endtoend import main as run_tournament
from src.cli import _replay_elo_history
from src.results_exporter import ResultsExporter
from src.visualization.heatmap_generator import HeatmapGenerator
from src.visualization.chart_generator import ChartGenerator

def test_pergroup_outputs():
    """Test per-group CSV and visualization generation."""
    print("=" * 80)
    print("TESTING PER-GROUP EXPORTS AND VISUALIZATIONS")
    print("=" * 80)
    
    # Run tournament (uses existing synthetic results)
    print("\n📊 Loading tournament results...")
    
    results_json = Path("results_synthetic/tournament_results.json")
    if not results_json.exists():
        print("❌ Running tournament first...")
        run_tournament()
    
    # Load results
    with open(results_json) as f:
        data = json.load(f)
    
    matches = data['matches']
    images_data = data['images']
    final_rankings = data['final_rankings']
    elo_ratings = data['elo_ratings']
    bt_ratings = data['bt_ratings']
    metadata = data['metadata']
    
    # Reconstruct ImageMetadata and TournamentResult
    from src.data_models import ImageMetadata, TournamentResult, TournamentMetadata, MatchResult
    from datetime import datetime
    
    images = []
    for img_data in images_data:
        # Create temp file for testing
        temp_path = Path(img_data['path'])
        if not temp_path.exists():
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            Image.new('RGB', (100, 100)).save(temp_path)
        
        images.append(ImageMetadata(
            id=img_data['id'],
            path=img_data['path'],
            prompt=img_data.get('prompt'),
            model=img_data.get('model'),
            group=img_data.get('group')
        ))
    
    # Reconstruct matches
    match_objs = []
    for m in matches:
        match_objs.append(MatchResult(
            image_a_id=m['image_a_id'],
            image_b_id=m['image_b_id'],
            scores_by_scorer=m['scores_by_scorer'],
            ensemble_score=m['ensemble_score'],
            winner_id=m['winner_id'],
            timestamp=datetime.fromisoformat(m['timestamp'])
        ))
    
    tour_metadata = TournamentMetadata(
        start_time=datetime.fromisoformat(metadata['start_time']),
        end_time=datetime.fromisoformat(metadata['end_time']) if metadata.get('end_time') else None,
        total_images=metadata['total_images'],
        total_matches=metadata['total_matches']
    )
    
    result = TournamentResult(
        matches=match_objs,
        elo_ratings=elo_ratings,
        bt_ratings=bt_ratings,
        final_rankings=[(r[0], r[1]) for r in final_rankings],
        metadata=tour_metadata
    )
    
    # Export per-group CSVs
    print(f"\n📤 Exporting per-group CSVs...")
    exporter = ResultsExporter()
    group_csvs = exporter.export_to_csv_per_group(result, images, "results_synthetic")
    
    print(f"✅ Generated {len(group_csvs)} per-group CSVs")
    
    # Group images
    from collections import defaultdict
    groups = defaultdict(list)
    image_id_to_group = {}
    
    for img in images:
        group = img.group if img.group else "_ungrouped"
        groups[group].append(img.id)
        image_id_to_group[img.id] = group
    
    print(f"\n✅ Loaded {len(images)} images in {len(groups)} groups")
    for group_name, img_ids in sorted(groups.items()):
        print(f"   • {group_name}: {len(img_ids)} images")
    
    # Create visualization output directory
    viz_dir = Path("results_synthetic/viz")
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize generators
    heatmap_gen = HeatmapGenerator()
    chart_gen = ChartGenerator()
    
    print(f"\n🎨 Generating per-group visualizations...")
    
    for group_name in sorted(groups.keys()):
        group_image_ids = set(groups[group_name])
        safe_name = group_name.replace(" ", "_").replace("/", "_")
        
        print(f"\n   Group: {group_name}")
        
        # Filter matches for this group
        group_matches = [
            m for m in matches
            if m['image_a_id'] in group_image_ids and m['image_b_id'] in group_image_ids
        ]
        
        print(f"      Matches: {len(group_matches)}")
        
        # 1. Heatmap
        matchups = [
            (m['image_a_id'], m['image_b_id'], m['ensemble_score'])
            for m in group_matches
        ]
        path = viz_dir / f'heatmap_{safe_name}.png'
        heatmap_gen.generate_pairwise_heatmap(matchups, str(path))
        print(f"      ✓ {path.name}")
        
        # 2. Rankings
        group_rankings = [
            (img_id, rating) for img_id, rating in final_rankings
            if img_id in group_image_ids
        ]
        path = viz_dir / f'rankings_{safe_name}.png'
        chart_gen.generate_final_rankings(group_rankings, str(path), title=f"Rankings: {group_name}")
        print(f"      ✓ {path.name}")
        
        # 3. Elo progression
        all_history = _replay_elo_history(matches)
        group_history = {
            img_id: hist for img_id, hist in all_history.items()
            if img_id in group_image_ids
        }
        path = viz_dir / f'progression_{safe_name}.png'
        chart_gen.generate_elo_progression(group_history, str(path), title=f"Elo Progression: {group_name}")
        print(f"      ✓ {path.name}")
        
        # 4. Score distribution
        scores = [m['ensemble_score'] for m in group_matches]
        path = viz_dir / f'distribution_{safe_name}.png'
        chart_gen.generate_score_distribution(scores, str(path), title=f"Score Distribution: {group_name}")
        print(f"      ✓ {path.name}")
    
    # List all generated files
    print(f"\n📁 Generated files:")
    csv_files = sorted(Path("results_synthetic").glob("evaluation_scores_*.csv"))
    viz_files = sorted(viz_dir.glob("*.png"))
    
    print(f"\n   Per-group CSVs ({len(csv_files)}):")
    for f in csv_files:
        size = f.stat().st_size
        with open(f) as cf:
            lines = sum(1 for _ in cf)
        print(f"      • {f.name} ({lines-1} rows, {size:,} bytes)")
    
    print(f"\n   Visualizations ({len(viz_files)}):")
    for f in viz_files:
        size = f.stat().st_size
        print(f"      • {f.name} ({size:,} bytes)")
    
    # Verify per-group CSV content
    print(f"\n🔍 Verifying per-group CSVs...")
    for csv_file in csv_files:
        import pandas as pd
        df = pd.read_csv(csv_file)
        group_name = csv_file.stem.replace("evaluation_scores_", "").replace("_", " ")
        
        print(f"\n   {csv_file.name}:")
        print(f"      Rows: {len(df)}")
        print(f"      Columns: {', '.join(df.columns[:8])}...")
        print(f"      Top 3 by rank:")
        for _, row in df.head(3).iterrows():
            print(f"         {row['rank']}. {row['image_id'][:30]:<30} (Elo: {row['elo_rating']:.1f}, BT: {row['bt_rating']:.4f})")
    
    print(f"\n✅ Per-group exports and visualizations complete!")
    print(f"✅ Total: {len(csv_files)} CSVs + {len(viz_files)} visualizations")
    
    return 0

if __name__ == "__main__":
    sys.exit(test_pergroup_outputs())
