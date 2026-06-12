"""
Command-line interface for IMAGE-LLP-VISION tournament system.

Provides three operation modes:
- tournament: Run pairwise image tournament
- visualize: Generate visualizations from results
- benchmark: Benchmark model performance
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict

import click
from tqdm import tqdm

from src.config_manager import ConfigManager
from src.csv_loader import CSVLoader
from src.image_loader import ImageLoader
from src.arena import Arena
from src.elo_system import EloSystem
from src.bradley_terry_mle import BradleyTerryMLE
from src.ensemble import Ensemble
from src.checkpoint_manager import CheckpointManager
from src.results_exporter import ResultsExporter
from src.visualization.heatmap_generator import HeatmapGenerator
from src.visualization.chart_generator import ChartGenerator

# Scorer imports (heavy; only loaded at runtime inside tournament())
from src.scoring.hpsv2 import HPSv2
from src.scoring.pickscore import PickScore
from src.scoring.image_reward import ImageReward
from src.scoring.vqascore import VQAScore
from src.scoring.vlm_judge import VLMJudge
from src.models.instructblip_adapter import InstructBLIPAdapter
from src.models.clip_adapter import CLIPAdapter
from src.models.blip_adapter import BLIPAdapter
from src.models.blip2_adapter import BLIP2Adapter


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _replay_elo_history(matches: List[dict]) -> Dict[str, List[float]]:
    """
    Reconstruct per-image Elo progression by replaying matches in order.

    The tournament JSON stores final ratings only, so the progression chart
    is rebuilt by re-running the same Elo updates Arena performs (winner score
    expressed from the winner's perspective).

    Args:
        matches: List of match dicts with image_a_id, image_b_id, ensemble_score

    Returns:
        Dict mapping image_id to its list of Elo ratings over time
    """
    elo = EloSystem(k_factor=32, initial_rating=1500.0)
    history: Dict[str, List[float]] = {}

    for m in matches:
        a_id = m['image_a_id']
        b_id = m['image_b_id']
        score = m['ensemble_score']

        # Seed initial ratings the first time we see each image
        for img_id in (a_id, b_id):
            if img_id not in history:
                history[img_id] = [elo.get_rating(img_id)]

        if score > 0.5:
            winner_id, loser_id = a_id, b_id
            winner_score = score
        else:
            winner_id, loser_id = b_id, a_id
            winner_score = 1.0 - score

        elo.update_ratings(winner_id=winner_id, loser_id=loser_id, score=winner_score)

        history[a_id].append(elo.get_rating(a_id))
        history[b_id].append(elo.get_rating(b_id))

    return history


# Model names that can be benchmarked standalone, mapped to their builders.
_VISION_ADAPTERS = {
    "clip": CLIPAdapter,
    "blip": BLIPAdapter,
    "blip2": BLIP2Adapter,
    "instructblip": InstructBLIPAdapter,
}
_STANDALONE_SCORERS = {
    "pickscore": PickScore,
    "hpsv2": HPSv2,
    "image_reward": ImageReward,
}


def _build_benchmark_target(name: str, cfg):
    """
    Construct and load a benchmarkable model/scorer by name.

    Args:
        name: Model name (e.g. "clip", "pickscore", "instructblip")
        cfg: Loaded Config object (for hf_id / quantization lookup)

    Returns:
        Tuple of (loaded_object, run_once_callable). run_once_callable takes a
        single PIL image and performs one representative inference.

    Raises:
        ValueError: If the name is unknown or cannot be benchmarked standalone
    """
    name_l = name.lower()
    model_cfg = cfg.models.get(name_l)

    if name_l in _VISION_ADAPTERS:
        if model_cfg is None:
            raise ValueError(f"No 'models.{name_l}' entry in config.yaml")
        adapter = _VISION_ADAPTERS[name_l]()
        adapter.load({"hf_id": model_cfg.hf_id, "quantization": model_cfg.quantization})
        return adapter, (lambda img: adapter.encode_image(img))

    if name_l in _STANDALONE_SCORERS:
        scorer = _STANDALONE_SCORERS[name_l](device="auto")
        if model_cfg is not None:
            scorer.load(model_cfg.hf_id)
        else:
            scorer.load()
        return scorer, (lambda img: scorer.compare(img, img))

    if name_l in ("vqa", "vlm_judge"):
        raise ValueError(
            f"'{name_l}' shares the InstructBLIP model and cannot be benchmarked "
            f"standalone; benchmark 'instructblip' instead"
        )

    raise ValueError(
        f"Unknown model '{name}'. Benchmarkable: "
        f"{sorted(list(_VISION_ADAPTERS) + list(_STANDALONE_SCORERS))}"
    )


def _benchmark_callable(run_once, images: list, iterations: int) -> Dict[str, float]:
    """
    Time a single-image callable over the given images for `iterations` passes.

    Args:
        run_once: Callable taking one image and performing one inference
        images: List of PIL images to run against
        iterations: Number of full passes over the image list

    Returns:
        Dict with avg_latency_ms (per call) and num_calls
    """
    latencies_ms: List[float] = []
    for _ in range(iterations):
        for img in images:
            t0 = time.perf_counter()
            run_once(img)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    n = len(latencies_ms)
    return {
        "avg_latency_ms": (sum(latencies_ms) / n) if n else 0.0,
        "num_calls": n,
    }


def _measure_peak_memory(obj) -> Optional[float]:
    """Best-effort model memory footprint in GB, or None if unavailable."""
    try:
        if hasattr(obj, "get_memory_footprint"):
            return obj.get_memory_footprint() / (1024 ** 3)
        inner = getattr(obj, "model", None)
        if inner is not None and hasattr(inner, "parameters"):
            total = sum(p.numel() * p.element_size() for p in inner.parameters())
            return total / (1024 ** 3)
    except Exception:
        pass
    return None


@click.group()
@click.version_option(version='1.0.0', prog_name='IMAGE-LLP-VISION')
def cli():
    """
    IMAGE-LLP-VISION: Pairwise Image Evaluation System
    
    A production-grade system for ranking images through tournament-style
    comparisons using multiple vision-language models.
    """
    pass


@cli.command()
@click.option('--data', '-d', required=True, type=click.Path(exists=True),
              help='Path to CSV/XLSX file with image metadata')
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to config.yaml file')
@click.option('--checkpoint', type=click.Path(),
              help='Optional checkpoint to resume from')
@click.option('--output', '-o', type=click.Path(),
              help='Optional output directory for results')
@click.option('--sheet-range', type=str,
              help='For XLSX: sheet range like "Sheet1:Sheet5"')
@click.option('--cell-range', type=str,
              help='For XLSX: cell range like "E2:E11"')
@click.option('--export-csv', type=click.Path(),
              help='Path to save per-image evaluation metrics as CSV (e.g. results/scores.csv)')
def tournament(data: str, config: str, checkpoint: Optional[str], output: Optional[str],
               sheet_range: Optional[str], cell_range: Optional[str],
               export_csv: Optional[str]):
    """
    Run tournament mode: Execute pairwise image tournament.
    
    For CSV files, the file must contain 'id' and 'path' columns.
    
    For XLSX files, two modes are supported:
    1. Standard mode: XLSX with 'id' and 'path' columns (like CSV)
    2. Multi-sheet mode: Extract image paths from specific cells across multiple sheets
       Requires both --sheet-range and --cell-range options
    
    Examples:
    
        # CSV or standard XLSX
        python -m src.cli tournament --data data/images.csv --config config.yaml
        
        # Multi-sheet XLSX mode (cells E2:E11 from Sheet1 through Sheet5)
        python -m src.cli tournament --data data/images.xlsx --config config.yaml \\
            --sheet-range "Sheet1:Sheet5" --cell-range "E2:E11"
        
        # Resume from checkpoint
        python -m src.cli tournament --data data/images.csv --config config.yaml --checkpoint checkpoints/latest.json
    """
    try:
        click.echo("=" * 60)
        click.echo("IMAGE-LLP-VISION Tournament Mode")
        click.echo("=" * 60)
        
        # Validate inputs
        data_path = Path(data)
        config_path = Path(config)
        
        if not data_path.exists():
            click.echo(f"❌ Error: Data file not found: {data}", err=True)
            sys.exit(1)
        
        if not config_path.exists():
            click.echo(f"❌ Error: Config file not found: {config}", err=True)
            sys.exit(1)
        
        # Validate sheet/cell range options
        if (sheet_range and not cell_range) or (cell_range and not sheet_range):
            click.echo("❌ Error: Both --sheet-range and --cell-range must be provided together", err=True)
            sys.exit(1)
        
        click.echo(f"\n📊 Loading configuration from: {config}")
        config_manager = ConfigManager(str(config_path))
        cfg = config_manager.load_config()
        
        click.echo(f"📁 Loading image metadata from: {data}")
        if sheet_range and cell_range:
            click.echo(f"   Multi-sheet mode: {sheet_range}, cells {cell_range}")
        
        csv_loader = CSVLoader()
        images = csv_loader.load_metadata(
            str(data_path),
            sheet_range=sheet_range,
            cell_range=cell_range
        )
        click.echo(f"✅ Loaded {len(images)} images")
        
        if len(images) < 2:
            click.echo("❌ Error: Need at least 2 images for tournament", err=True)
            sys.exit(1)
        
        # Check for loading errors
        errors = csv_loader.get_load_errors()
        if errors:
            click.echo(f"⚠️  Warning: {len(errors)} images failed to load")
            for identifier, error_msg in errors[:3]:
                click.echo(f"   - {identifier}: {error_msg}")
            if len(errors) > 3:
                click.echo(f"   ... and {len(errors) - 3} more errors")
        
        # ── 1. Initialise scorers ──────────────────────────────────────────
        # Resilient loading: a scorer that fails to load is skipped (with a
        # warning) rather than aborting the whole tournament. The ensemble
        # renormalises over whichever scorers loaded successfully, so a run can
        # still proceed with a subset (e.g. if ImageReward's package is absent).
        click.echo("\n🔧 Loading evaluation models (this may take several minutes)...")
        scorers = {}

        # --- InstructBLIP: loaded once, shared by VQAScore + VLMJudge ---
        instructblip_cfg = cfg.models.get("instructblip")
        if instructblip_cfg is None:
            raise RuntimeError(
                "config.yaml must have a 'models.instructblip' entry with hf_id and quantization."
            )
        try:
            click.echo("  ↳ Loading InstructBLIP (shared by VQAScore + VLMJudge)...")
            instructblip = InstructBLIPAdapter()
            instructblip.load({"hf_id": instructblip_cfg.hf_id,
                               "quantization": instructblip_cfg.quantization})
            scorers["vqa"] = VQAScore(instructblip)
            scorers["vlm_judge"] = VLMJudge(instructblip)
        except Exception as e:
            click.echo(f"  ⚠️  InstructBLIP failed to load: {e}", err=True)
            click.echo("      → Skipping VQAScore + VLMJudge for this run.", err=True)
            logger.warning(f"InstructBLIP load failed; VQA+VLMJudge unavailable: {e}")

        # --- PickScore ---
        ps_cfg = cfg.models.get("pickscore")
        ps_hf  = ps_cfg.hf_id if ps_cfg else "yuvalkirstain/PickScore_v1"
        try:
            click.echo(f"  ↳ Loading PickScore ({ps_hf})...")
            ps = PickScore(device="auto")
            ps.load(ps_hf)
            scorers["pickscore"] = ps
        except Exception as e:
            click.echo(f"  ⚠️  PickScore failed to load: {e}", err=True)
            logger.warning(f"PickScore load failed: {e}")

        # --- HPSv2 ---
        hps_cfg = cfg.models.get("hpsv2")
        hps_hf  = hps_cfg.hf_id if hps_cfg else "xswu/HPSv2"
        try:
            click.echo(f"  ↳ Loading HPSv2 ({hps_hf})...")
            hps = HPSv2(device="auto")
            hps.load(hps_hf)
            scorers["hpsv2"] = hps
        except Exception as e:
            click.echo(f"  ⚠️  HPSv2 failed to load: {e}", err=True)
            logger.warning(f"HPSv2 load failed: {e}")

        # --- ImageReward ---
        ir_cfg = cfg.models.get("image_reward")
        ir_hf  = ir_cfg.hf_id if ir_cfg else "THUDM/ImageReward"
        try:
            click.echo(f"  ↳ Loading ImageReward ({ir_hf})...")
            ir = ImageReward(device="auto")
            ir.load(ir_hf)
            scorers["image_reward"] = ir
        except Exception as e:
            click.echo(f"  ⚠️  ImageReward failed to load: {e}", err=True)
            click.echo("      → Tip: `pip install image-reward` to enable this scorer.", err=True)
            logger.warning(f"ImageReward load failed: {e}")

        # --- CLIPScore (prompt-image alignment) ---
        clip_cfg = cfg.models.get("clip")
        clip_hf  = clip_cfg.hf_id if clip_cfg else "openai/clip-vit-large-patch14"
        try:
            click.echo(f"  ↳ Loading CLIP for prompt alignment ({clip_hf})...")
            from src.scoring.clipscore import CLIPScore
            clip_scorer = CLIPScore(device="auto")
            clip_scorer.load(clip_hf)
            scorers["clip_alignment"] = clip_scorer
        except Exception as e:
            click.echo(f"  ⚠️  CLIPScore failed to load: {e}", err=True)
            logger.warning(f"CLIPScore load failed: {e}")

        # ── 2. Validate we have usable scorers ────────────────────────────
        if not scorers:
            click.echo("❌ Error: no scorers could be loaded; cannot run tournament.", err=True)
            sys.exit(1)

        if not any(name in cfg.scoring.weights for name in scorers):
            click.echo(
                "❌ Error: none of the loaded scorers have a weight in "
                "config.yaml 'scoring.weights'; ensemble cannot be computed.",
                err=True,
            )
            sys.exit(1)

        click.echo(f"✅ Active scorers ({len(scorers)}/5): {', '.join(sorted(scorers))}")
        missing = sorted(set(cfg.scoring.weights) - set(scorers))
        if missing:
            click.echo(
                f"⚠️  Degraded run — ensemble will renormalise without: {', '.join(missing)}"
            )

        # ── 3. Build rating + ensemble components ─────────────────────────
        ensemble_inst   = Ensemble(weights=cfg.scoring.weights)
        elo_inst        = EloSystem(k_factor=32, initial_rating=1500.0)
        bt_mle_inst     = BradleyTerryMLE(beta=cfg.scoring.bt_mle.beta)
        ckpt_dir        = "checkpoints"
        ckpt_manager    = CheckpointManager(checkpoint_dir=ckpt_dir)

        # ── 4. Build Arena ────────────────────────────────────────────────
        arena_cfg = {
            "checkpoint_every": cfg.training.checkpoint_every,
        }
        arena = Arena(
            images=images,
            scorers=scorers,
            elo_system=elo_inst,
            bt_mle=bt_mle_inst,
            ensemble=ensemble_inst,
            checkpoint_manager=ckpt_manager,
            model_rotator=None,
            config=arena_cfg,
        )

        # ── 5. Resume from checkpoint if provided ─────────────────────────
        if checkpoint:
            click.echo(f"\n♻️  Resuming from checkpoint: {checkpoint}")
            arena.resume_from_checkpoint(checkpoint)

        # ── 6. Run tournament ─────────────────────────────────────────────
        total_pairs = len(images) * (len(images) - 1) // 2
        click.echo(
            f"\n🏁 Starting tournament: {len(images)} images, "
            f"{total_pairs} pairwise matches"
        )
        result = arena.run_tournament()

        click.echo(
            f"\n🏆 Tournament completed! "
            f"{len(result.matches)} matches, "
            f"{len(result.final_rankings)} ranked images."
        )

        # ── 7. Export results ─────────────────────────────────────────────
        exporter = ResultsExporter()

        # 7a. Per-image metrics CSV
        if export_csv:
            click.echo(f"\n📤 Exporting per-image evaluation metrics...")
            try:
                # Combined CSV
                csv_path = exporter.export_to_csv(result, images, export_csv)
                click.echo(f"✅ Combined CSV: {csv_path}")
                
                # Per-group CSVs
                csv_dir = Path(export_csv).parent
                group_csvs = exporter.export_to_csv_per_group(result, images, str(csv_dir))
                click.echo(f"✅ Per-group CSVs: {len(group_csvs)} files")
                for group_name, path in sorted(group_csvs.items()):
                    click.echo(f"   • {Path(path).name}")
            except Exception as export_err:
                click.echo(f"⚠️  CSV export failed: {export_err}", err=True)
                logger.warning(f"CSV export failed: {export_err}")

        # 7b. Full tournament results JSON (consumed by `visualize` mode)
        if output:
            click.echo(f"\n💾 Saving full tournament results (JSON)...")
            try:
                out_path = Path(output)
                if out_path.suffix.lower() == ".json":
                    json_target = str(out_path)
                else:
                    json_target = str(out_path / "tournament_results.json")
                json_path = exporter.export_to_json(result, images, json_target)
                click.echo(f"✅ Results saved to: {json_path}")
            except Exception as json_err:
                click.echo(f"⚠️  JSON export failed: {json_err}", err=True)
                logger.warning(f"JSON export failed: {json_err}")
        
    except KeyboardInterrupt:
        click.echo("\n\n⚠️  Tournament interrupted by user", err=True)
        click.echo("💾 Saving checkpoint...")
        sys.exit(130)
    
    except Exception as e:
        click.echo(f"\n❌ Tournament failed: {e}", err=True)
        logger.exception("Tournament execution failed")
        sys.exit(1)


@cli.command()
@click.option('--results', '-r', required=True, type=click.Path(exists=True),
              help='Path to tournament results JSON file')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output directory for visualizations')
@click.option('--types', '-t', multiple=True,
              type=click.Choice(['heatmap', 'rankings', 'progression', 'distribution', 'all']),
              default=['all'],
              help='Types of visualizations to generate')
def visualize(results: str, output: str, types: tuple):
    """
    Visualize mode: Generate visualizations from tournament results.
    
    Examples:
    
        # Generate all visualizations
        python -m src.cli visualize --results results.json --output viz/
        
        # Generate specific visualizations
        python -m src.cli visualize --results results.json --output viz/ --types rankings --types progression
    """
    try:
        click.echo("=" * 60)
        click.echo("IMAGE-LLP-VISION Visualization Mode")
        click.echo("=" * 60)
        
        # Validate inputs
        results_path = Path(results)
        output_dir = Path(output)
        
        if not results_path.exists():
            click.echo(f"❌ Error: Results file not found: {results}", err=True)
            sys.exit(1)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine which visualizations to generate
        viz_types = set(types)
        if 'all' in viz_types:
            viz_types = {'heatmap', 'rankings', 'progression', 'distribution'}
        
        click.echo(f"\n📊 Loading results from: {results}")
        click.echo(f"📁 Output directory: {output}")
        click.echo(f"🎨 Generating visualizations: {', '.join(sorted(viz_types))}\n")

        # Load tournament results JSON (produced by `tournament --output`)
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        matches = data.get('matches', [])
        final_rankings = data.get('final_rankings', [])
        images_data = data.get('images', [])

        if not matches and not final_rankings:
            click.echo(
                "❌ Error: results file contains no matches or rankings. "
                "Did you run `tournament --output <dir>` first?",
                err=True,
            )
            sys.exit(1)
        
        # Group images
        from collections import defaultdict
        groups = defaultdict(list)
        image_id_to_group = {}
        
        for img in images_data:
            group = img.get('group') or img.get('prompt') or "_ungrouped"
            groups[group].append(img['id'])
            image_id_to_group[img['id']] = group

        # Initialize generators
        heatmap_gen = HeatmapGenerator()
        chart_gen = ChartGenerator()

        generated: List[str] = []
        
        # Generate per-group visualizations
        if len(groups) > 1:
            click.echo(f"\n🎨 Generating per-group visualizations for {len(groups)} groups...")
            
            for group_name in sorted(groups.keys()):
                group_image_ids = set(groups[group_name])
                safe_name = group_name.replace(" ", "_").replace("/", "_")
                
                click.echo(f"\n   Group: {group_name} ({len(group_image_ids)} images)")
                
                # Filter matches for this group
                group_matches = [
                    m for m in matches
                    if m['image_a_id'] in group_image_ids and m['image_b_id'] in group_image_ids
                ]
                
                # 1. Group heatmap
                if 'heatmap' in viz_types:
                    matchups = [
                        (m['image_a_id'], m['image_b_id'], m['ensemble_score'])
                        for m in group_matches
                    ]
                    if matchups:
                        path = str(output_dir / f'heatmap_{safe_name}.png')
                        heatmap_gen.generate_pairwise_heatmap(matchups, path)
                        generated.append(path)
                        click.echo(f"      ✓ heatmap_{safe_name}.png")
                
                # 2. Group rankings
                if 'rankings' in viz_types:
                    group_rankings = [
                        (img_id, rating) for img_id, rating in final_rankings
                        if img_id in group_image_ids
                    ]
                    if group_rankings:
                        path = str(output_dir / f'rankings_{safe_name}.png')
                        chart_gen.generate_final_rankings(group_rankings, path, title=f"Rankings: {group_name}")
                        generated.append(path)
                        click.echo(f"      ✓ rankings_{safe_name}.png")
                
                # 3. Group Elo progression
                if 'progression' in viz_types:
                    all_history = _replay_elo_history(matches)
                    group_history = {
                        img_id: hist for img_id, hist in all_history.items()
                        if img_id in group_image_ids
                    }
                    if group_history:
                        path = str(output_dir / f'progression_{safe_name}.png')
                        chart_gen.generate_elo_progression(group_history, path, title=f"Elo Progression: {group_name}")
                        generated.append(path)
                        click.echo(f"      ✓ progression_{safe_name}.png")
                
                # 4. Group score distribution
                if 'distribution' in viz_types:
                    scores = [m['ensemble_score'] for m in group_matches]
                    if scores:
                        path = str(output_dir / f'distribution_{safe_name}.png')
                        chart_gen.generate_score_distribution(scores, path, title=f"Score Distribution: {group_name}")
                        generated.append(path)
                        click.echo(f"      ✓ distribution_{safe_name}.png")
        
        # Generate combined/global visualizations
        click.echo(f"\n🎨 Generating combined visualizations...")

        # 1. Pairwise comparison heatmap (all groups)
        if 'heatmap' in viz_types:
            matchups = [
                (m['image_a_id'], m['image_b_id'], m['ensemble_score'])
                for m in matches
            ]
            if matchups:
                path = str(output_dir / 'heatmap.png')
                heatmap_gen.generate_pairwise_heatmap(matchups, path)
                generated.append(path)
            else:
                click.echo("   ↳ Skipping heatmap (no matches in results)")

        # 2. Final rankings bar chart
        if 'rankings' in viz_types:
            rankings = [(r[0], r[1]) for r in final_rankings]
            if rankings:
                path = str(output_dir / 'rankings.png')
                chart_gen.generate_final_rankings(rankings, path)
                generated.append(path)
            else:
                click.echo("   ↳ Skipping rankings (no rankings in results)")

        # 3. Elo rating progression (replayed from match order)
        if 'progression' in viz_types:
            history = _replay_elo_history(matches)
            if history:
                path = str(output_dir / 'progression.png')
                chart_gen.generate_elo_progression(history, path)
                generated.append(path)
            else:
                click.echo("   ↳ Skipping progression (no matches in results)")

        # 4. Score distribution histogram
        if 'distribution' in viz_types:
            scores = [m['ensemble_score'] for m in matches]
            if scores:
                path = str(output_dir / 'distribution.png')
                chart_gen.generate_score_distribution(scores, path)
                generated.append(path)
            else:
                click.echo("   ↳ Skipping distribution (no matches in results)")

        if generated:
            click.echo(f"\n✅ Generated {len(generated)} visualization(s):")
            for p in generated:
                click.echo(f"   • {p}")
        else:
            click.echo("\n⚠️  No visualizations were generated.", err=True)

    except Exception as e:
        click.echo(f"\n❌ Visualization failed: {e}", err=True)
        logger.exception("Visualization generation failed")
        sys.exit(1)


@cli.command()
@click.option('--models', '-m', multiple=True, required=True,
              help='Model names to benchmark (can specify multiple)')
@click.option('--images', '-i', required=True, type=click.Path(exists=True),
              help='Path to CSV file with test images')
@click.option('--iterations', '-n', type=int, default=10,
              help='Number of iterations per model (default: 10)')
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to config.yaml file')
def benchmark(models: tuple, images: str, iterations: int, config: str):
    """
    Benchmark mode: Test model performance and memory usage.
    
    Examples:
    
        # Benchmark single model
        python -m src.cli benchmark --models clip --images test.csv --config config.yaml
        
        # Benchmark multiple models
        python -m src.cli benchmark --models clip --models blip --images test.csv --config config.yaml -n 20
    """
    try:
        click.echo("=" * 60)
        click.echo("IMAGE-LLP-VISION Benchmark Mode")
        click.echo("=" * 60)
        
        # Validate inputs
        images_path = Path(images)
        config_path = Path(config)
        
        if not images_path.exists():
            click.echo(f"❌ Error: Images file not found: {images}", err=True)
            sys.exit(1)
        
        if not config_path.exists():
            click.echo(f"❌ Error: Config file not found: {config}", err=True)
            sys.exit(1)
        
        if iterations <= 0:
            click.echo(f"❌ Error: Iterations must be positive, got {iterations}", err=True)
            sys.exit(1)
        
        click.echo(f"\n🧪 Benchmarking models: {', '.join(models)}")
        click.echo(f"📁 Test images: {images}")
        click.echo(f"🔄 Iterations: {iterations}\n")

        # Load configuration (for hf_id / quantization)
        cfg = ConfigManager(str(config_path)).load_config()

        # Load test images
        loader = CSVLoader()
        image_metas = loader.load_metadata(str(images_path))
        image_loader = ImageLoader()
        pil_images = [
            img for img in (image_loader.load_image(m.path) for m in image_metas)
            if img is not None
        ]
        if not pil_images:
            click.echo(
                "❌ Error: no valid test images could be loaded from the file.",
                err=True,
            )
            sys.exit(1)
        click.echo(f"📷 Loaded {len(pil_images)} test image(s)\n")

        results: List[Dict] = []
        for name in models:
            click.echo(f"🔧 Loading and benchmarking '{name}'...")
            try:
                obj, run_once = _build_benchmark_target(name, cfg)
            except Exception as build_err:
                click.echo(f"   ⚠️  Skipped '{name}': {build_err}", err=True)
                results.append({"model": name, "error": str(build_err)})
                continue

            try:
                run_once(pil_images[0])  # warm-up (not timed)
                stats = _benchmark_callable(run_once, pil_images, iterations)
                stats["model"] = name
                stats["peak_memory_gb"] = _measure_peak_memory(obj)
                results.append(stats)
                mem_str = (
                    f"{stats['peak_memory_gb']:.2f} GB"
                    if stats["peak_memory_gb"] is not None else "n/a"
                )
                click.echo(
                    f"   ✓ {stats['avg_latency_ms']:.1f} ms/image over "
                    f"{stats['num_calls']} calls, peak memory {mem_str}"
                )
            except Exception as run_err:
                click.echo(f"   ⚠️  '{name}' failed during inference: {run_err}", err=True)
                results.append({"model": name, "error": str(run_err)})
            finally:
                if hasattr(obj, "unload"):
                    try:
                        obj.unload()
                    except Exception:
                        pass

        # Summary table
        click.echo("\n" + "=" * 56)
        click.echo(f"{'Model':<18}{'Latency (ms)':>18}{'Peak Mem (GB)':>20}")
        click.echo("-" * 56)
        for r in results:
            if "error" in r:
                click.echo(f"{r['model']:<18}{'ERROR':>18}{'—':>20}")
            else:
                mem = (
                    f"{r['peak_memory_gb']:.2f}"
                    if r.get("peak_memory_gb") is not None else "n/a"
                )
                click.echo(f"{r['model']:<18}{r['avg_latency_ms']:>18.1f}{mem:>20}")
        click.echo("=" * 56)

        click.echo("\n✅ Benchmark completed successfully!")

    except Exception as e:
        click.echo(f"\n❌ Benchmark failed: {e}", err=True)
        logger.exception("Benchmark execution failed")
        sys.exit(1)


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == '__main__':
    main()
