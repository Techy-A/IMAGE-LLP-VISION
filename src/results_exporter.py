"""
Results exporter for IMAGE-LLP-VISION tournament outcomes.

Produces a per-image CSV file aggregating all evaluation metrics:
- Individual scorer averages (PickScore, HPSv2, ImageReward, VQAScore, VLM-judge)
- Ensemble average score
- Win/loss record and win rate
- Elo rating
- Bradley-Terry MLE rating and final rank
"""

import csv
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data_models import ImageMetadata, TournamentResult

logger = logging.getLogger(__name__)


# SCORER_COLUMNS maps the key used in MatchResult.scores_by_scorer (left)
# to the corresponding CSV column header (right). This mapping is used both
# when building CSV rows and when flipping scores for image_b's perspective.
# Add a new entry here whenever a new scorer is wired into the Arena.
SCORER_COLUMNS = [
    ("pickscore",       "avg_pickscore"),
    ("hpsv2",           "avg_hpsv2"),
    ("image_reward",    "avg_image_reward"),
    ("vqa",             "avg_vqascore"),
    ("vlm_judge",       "avg_vlm_judge"),
    ("clip_alignment",  "avg_clip_alignment"),
]

CSV_COLUMNS = [
    "rank",
    "image_id",
    "group",
    "image_path",
    "prompt",
    "model",
    "elo_rating",
    "bt_rating",
    "wins",
    "losses",
    "matches_played",
    "win_rate",
    "avg_ensemble_score",
    "avg_pickscore",
    "avg_hpsv2",
    "avg_image_reward",
    "avg_vqascore",
    "avg_vlm_judge",
    "avg_clip_alignment",
]


class ResultsExporter:
    """
    Exports tournament results to a per-image CSV file.

    Each row corresponds to one image and contains:
    - Identification columns (id, path, prompt, model)
    - Elo and Bradley-Terry ratings with rank
    - Win/loss record and win rate
    - Average score from each scorer (from image's own perspective)
    - Average ensemble score

    Score perspective: All MatchResult ensemble_scores and scorer scores are
    recorded from image_a's perspective (> 0.5 means image_a won). When
    computing a given image's average score, matches where it was image_b
    are inverted (1.0 - score) so the value always represents
    "how well this image did" rather than "image_a's raw score".
    """

    def export_to_csv_per_group(
        self,
        result: TournamentResult,
        images: List[ImageMetadata],
        output_dir: str,
    ) -> Dict[str, str]:
        """
        Export per-group CSVs with group-specific rankings.
        
        Args:
            result: Tournament result
            images: List of ImageMetadata
            output_dir: Directory for output files
            
        Returns:
            Dict mapping group name to CSV file path
        """
        from pathlib import Path
        from collections import defaultdict
        
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        
        # Group images
        groups = defaultdict(list)
        for img in images:
            group = img.group if img.group else "_ungrouped"
            groups[group].append(img)
        
        group_files = {}
        
        for group_name, group_images in groups.items():
            # Filter data for this group
            group_image_ids = {img.id for img in group_images}
            
            # Build per-group stats
            meta_by_id = {img.id: img for img in group_images}
            stats = self._aggregate_match_stats(result, group_image_ids)
            
            # Rank within group by BT rating
            group_rankings = [
                (img_id, result.bt_ratings.get(img_id, 0))
                for img_id in group_image_ids
            ]
            group_rankings.sort(key=lambda x: x[1], reverse=True)
            rank_by_id = {img_id: rank for rank, (img_id, _) in enumerate(group_rankings, 1)}
            
            # Build rows
            rows = []
            for img_id in group_image_ids:
                meta = meta_by_id.get(img_id)
                img_stat = stats[img_id]
                wins = img_stat["wins"]
                losses = img_stat["losses"]
                played = wins + losses
                win_rate = (wins / played) if played > 0 else 0.0
                
                row = {
                    "rank": rank_by_id.get(img_id, ""),
                    "image_id": img_id,
                    "image_path": meta.path if meta else "",
                    "prompt": (meta.prompt or "") if meta else "",
                    "model": (meta.model or "") if meta else "",
                    "elo_rating": round(result.elo_ratings.get(img_id, 0.0), 4),
                    "bt_rating": round(result.bt_ratings.get(img_id, 0.0), 6),
                    "wins": wins,
                    "losses": losses,
                    "matches_played": played,
                    "win_rate": round(win_rate, 4),
                    "avg_ensemble_score": self._safe_mean(img_stat["ensemble_scores"]),
                }
                
                for scorer_key, col_name in SCORER_COLUMNS:
                    row[col_name] = self._safe_mean(img_stat["scorer_scores"][scorer_key])
                
                rows.append(row)
            
            # Sort by rank
            rows.sort(key=lambda r: r["rank"])
            
            # Write CSV
            safe_name = group_name.replace(" ", "_").replace("/", "_")
            csv_path = output / f"evaluation_scores_{safe_name}.csv"
            
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            
            group_files[group_name] = str(csv_path)
            logger.info(f"Exported {len(rows)} images for group '{group_name}' to {csv_path}")
        
        return group_files
    
    def export_to_csv(
        self,
        result: TournamentResult,
        images: List[ImageMetadata],
        output_path: str,
    ) -> str:
        """
        Export tournament results to CSV.

        Args:
            result: Completed TournamentResult from Arena.run_tournament()
            images: Original list of ImageMetadata (for path/prompt/model fields)
            output_path: Destination file path (directories are created automatically)

        Returns:
            Absolute path of the written CSV file

        Raises:
            ValueError: If result or images are empty
            IOError: If file cannot be written
        """
        if not result.matches and not result.elo_ratings:
            raise ValueError("TournamentResult contains no data to export")

        if not images:
            raise ValueError("images list cannot be empty")

        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        # Build lookup: image_id -> ImageMetadata
        meta_by_id: Dict[str, ImageMetadata] = {img.id: img for img in images}

        # Build rank lookup: image_id -> rank (1-based)
        rank_by_id: Dict[str, int] = {
            image_id: rank
            for rank, (image_id, _) in enumerate(result.final_rankings, start=1)
        }

        # Collect all image IDs (union of ratings + images list)
        all_ids = set(result.elo_ratings.keys()) | {img.id for img in images}

        # Aggregate per-image stats from match results
        stats = self._aggregate_match_stats(result, all_ids)

        rows = []
        for image_id in all_ids:
            meta = meta_by_id.get(image_id)
            image_stat = stats[image_id]
            wins = image_stat["wins"]
            losses = image_stat["losses"]
            played = wins + losses
            win_rate = (wins / played) if played > 0 else 0.0

            row: Dict[str, object] = {
                "rank":              rank_by_id.get(image_id, ""),
                "image_id":          image_id,
                "group":             meta.group if meta else "",
                "image_path":        meta.path if meta else "",
                "prompt":            (meta.prompt or "") if meta else "",
                "model":             (meta.model or "") if meta else "",
                "elo_rating":        round(result.elo_ratings.get(image_id, 0.0), 4),
                "bt_rating":         round(result.bt_ratings.get(image_id, 0.0), 6),
                "wins":              wins,
                "losses":            losses,
                "matches_played":    played,
                "win_rate":          round(win_rate, 4),
                "avg_ensemble_score": self._safe_mean(image_stat["ensemble_scores"]),
            }

            # Add per-scorer averages
            for scorer_key, col_name in SCORER_COLUMNS:
                row[col_name] = self._safe_mean(image_stat["scorer_scores"][scorer_key])

            rows.append(row)

        # Sort by rank (ascending), then by elo descending for unranked
        rows.sort(key=lambda r: (
            r["rank"] if isinstance(r["rank"], int) else 9999,
            -(r["elo_rating"] if isinstance(r["elo_rating"], float) else 0.0),
        ))

        # Write CSV
        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Exported {len(rows)} image results to {output}")
        return str(output)

    def export_to_json(
        self,
        result: TournamentResult,
        images: List[ImageMetadata],
        output_path: str,
    ) -> str:
        """
        Export the full tournament result to a JSON file.

        This is the machine-readable counterpart to export_to_csv. The file
        contains every match, all Elo / Bradley-Terry ratings, the final
        rankings, image metadata, and tournament metadata. It is the input
        consumed by ``visualize`` mode.

        Args:
            result: Completed TournamentResult from Arena.run_tournament()
            images: Original list of ImageMetadata (for id/path/prompt/model)
            output_path: Destination file path (directories created automatically)

        Returns:
            Absolute path of the written JSON file

        Raises:
            ValueError: If result has no data to export
            IOError: If file cannot be written
        """
        if not result.matches and not result.elo_ratings:
            raise ValueError("TournamentResult contains no data to export")

        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        meta = result.metadata
        payload = {
            "version": "1.0",
            "metadata": {
                "start_time": self._iso(meta.start_time),
                "end_time": self._iso(meta.end_time),
                "total_images": meta.total_images,
                "total_matches": meta.total_matches,
                "skipped_images": list(meta.skipped_images),
                "failed_scorers": dict(meta.failed_scorers),
            },
            "images": [
                {
                    "id": img.id,
                    "path": img.path,
                    "prompt": img.prompt,
                    "model": img.model,
                    "group": img.group,
                }
                for img in images
            ],
            "elo_ratings": result.elo_ratings,
            "bt_ratings": result.bt_ratings,
            "final_rankings": [
                [image_id, rating] for image_id, rating in result.final_rankings
            ],
            "matches": [
                {
                    "image_a_id": m.image_a_id,
                    "image_b_id": m.image_b_id,
                    "scores_by_scorer": m.scores_by_scorer,
                    "ensemble_score": m.ensemble_score,
                    "winner_id": m.winner_id,
                    "timestamp": self._iso(m.timestamp),
                }
                for m in result.matches
            ],
        }

        with open(output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Exported tournament JSON ({len(result.matches)} matches) to {output}"
        )
        return str(output)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iso(value: Optional[datetime]) -> Optional[str]:
        """Serialize a datetime to ISO format, passing through None."""
        return value.isoformat() if value is not None else None

    def _aggregate_match_stats(
        self,
        result: TournamentResult,
        all_ids,
    ) -> Dict[str, Dict]:
        """Build per-image aggregation dict from all completed matches."""
        stats: Dict[str, Dict] = {
            image_id: {
                "wins": 0,
                "losses": 0,
                "ensemble_scores": [],
                "scorer_scores": defaultdict(list),
            }
            for image_id in all_ids
        }

        for match in result.matches:
            a_id = match.image_a_id
            b_id = match.image_b_id
            e_score = match.ensemble_score  # always image_a perspective

            # ── image_a perspective ───────────────────────────────────
            # ensemble_score is stored from image_a's POV: 0.7 means image_a
            # performed at 70%. Append directly to image_a's list.
            if a_id in stats:
                stats[a_id]["ensemble_scores"].append(e_score)
                if match.winner_id == a_id:
                    stats[a_id]["wins"] += 1
                else:
                    stats[a_id]["losses"] += 1
                for scorer_key, _ in SCORER_COLUMNS:
                    if scorer_key in match.scores_by_scorer:
                        stats[a_id]["scorer_scores"][scorer_key].append(
                            match.scores_by_scorer[scorer_key]
                        )

            # ── image_b perspective (scores must be inverted) ────────────
            # A raw score of 0.7 means image_a got 70%, so image_b got 30%.
            # We record (1.0 − score) for image_b so that every entry in
            # each image's score list is "how well that image did" — the
            # CSV avg_* columns are therefore directly comparable across images
            # regardless of whether they were in the image_a or image_b slot.
            if b_id in stats:
                stats[b_id]["ensemble_scores"].append(1.0 - e_score)
                if match.winner_id == b_id:
                    stats[b_id]["wins"] += 1
                else:
                    stats[b_id]["losses"] += 1
                for scorer_key, _ in SCORER_COLUMNS:
                    if scorer_key in match.scores_by_scorer:
                        stats[b_id]["scorer_scores"][scorer_key].append(
                            1.0 - match.scores_by_scorer[scorer_key]
                        )

        return stats

    @staticmethod
    def _safe_mean(values: list) -> str:
        """Return rounded mean or empty string if no values."""
        if not values:
            return ""
        return round(sum(values) / len(values), 4)
