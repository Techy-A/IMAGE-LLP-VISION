"""
Heatmap generation for tournament visualization.

This module provides heatmap visualizations for pairwise comparisons
and per-scorer ratings, using seaborn and matplotlib.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)


class HeatmapGenerator:
    """
    Generator for tournament heatmap visualizations.
    
    Creates heatmaps showing:
    - Pairwise comparison results matrix
    - Per-scorer rating comparisons
    
    Uses seaborn for high-quality heatmap rendering with
    proper color scales and annotations.
    """
    
    def __init__(self, style: str = "whitegrid", figsize: Tuple[int, int] = (12, 10)):
        """
        Initialize heatmap generator.
        
        Args:
            style: Seaborn style (default: "whitegrid")
            figsize: Default figure size as (width, height)
        """
        self.style = style
        self.default_figsize = figsize
        
        # Set style
        sns.set_style(style)
        
        logger.info(f"HeatmapGenerator initialized with style: {style}")
    
    def generate_pairwise_heatmap(
        self,
        matchups: List[Tuple[str, str, float]],
        output_path: str,
        title: Optional[str] = None,
        cmap: str = "RdYlGn",
        figsize: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Generate heatmap showing pairwise comparison results.
        
        Creates a matrix where cell (i, j) shows the score when
        image i was compared to image j. Diagonal is NaN (no self-comparison).
        
        Args:
            matchups: List of (image_a_id, image_b_id, score) tuples
            output_path: Path to save heatmap image
            title: Optional custom title (default: "Pairwise Comparison Matrix")
            cmap: Colormap name (default: "RdYlGn" for red-yellow-green)
            figsize: Optional figure size (uses default if None)
        
        Raises:
            ValueError: If matchups is empty
            RuntimeError: If heatmap generation fails
        """
        if not matchups:
            raise ValueError("matchups cannot be empty")
        
        try:
            logger.info(f"Generating pairwise heatmap with {len(matchups)} matchups")
            
            # Extract unique image IDs
            image_ids = set()
            for image_a, image_b, _ in matchups:
                image_ids.add(image_a)
                image_ids.add(image_b)
            
            image_ids = sorted(image_ids)
            n_images = len(image_ids)
            
            # Create index mapping
            id_to_idx = {img_id: idx for idx, img_id in enumerate(image_ids)}
            
            # Initialize matrix with NaN (for self-comparisons and missing pairs)
            matrix = np.full((n_images, n_images), np.nan)
            
            # ── Fill comparison matrix ────────────────────────────────────────
            # Every match score is stored from image_a's perspective.
            # matrix[i][j] = "how well image_i did when compared to image_j"
            # Because the scorer returns image_a's score, and here i=image_a,
            # we store the raw score at [i, j].
            #
            # For the transpose [j, i] we record image_b's perspective:
            # (1 - score). This is guaranteed because for any match:
            #   score(a vs b) + score(b vs a) = 1.0
            # (they are complementary probabilities).
            # The matrix is therefore anti-symmetric around 0.5:
            #   matrix[i,j] + matrix[j,i] == 1.0  for all i≠j
            for image_a, image_b, score in matchups:
                i = id_to_idx[image_a]
                j = id_to_idx[image_b]

                # image_a vs image_b: score is image_a's result
                matrix[i, j] = score

                # image_b vs image_a: complementary score (image_b's result)
                matrix[j, i] = 1.0 - score
            
            # Create figure
            figsize = figsize or self.default_figsize
            fig, ax = plt.subplots(figsize=figsize)
            
            # Generate heatmap
            sns.heatmap(
                matrix,
                annot=False,  # Don't annotate cells (too many for large matrices)
                fmt='.2f',
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                center=0.5,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Score (>0.5 = Row Wins)'},
                xticklabels=image_ids,
                yticklabels=image_ids,
                ax=ax
            )
            
            # Set title
            if title is None:
                title = f"Pairwise Comparison Matrix ({n_images} images)"
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Labels
            ax.set_xlabel("Image ID (Column)", fontsize=12)
            ax.set_ylabel("Image ID (Row)", fontsize=12)
            
            # Rotate labels for readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Pairwise heatmap saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate pairwise heatmap: {e}")
            raise RuntimeError(f"Pairwise heatmap generation failed: {e}") from e
    
    def generate_scorer_heatmap(
        self,
        scores_by_scorer: Dict[str, Dict[str, float]],
        output_path: str,
        title: Optional[str] = None,
        cmap: str = "viridis",
        figsize: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Generate heatmap showing per-scorer ratings.
        
        Creates a matrix where rows are images and columns are scorers,
        showing how each scorer rates each image.
        
        Args:
            scores_by_scorer: Dict mapping scorer_name to Dict[image_id -> rating]
            output_path: Path to save heatmap image
            title: Optional custom title (default: "Scorer Ratings Comparison")
            cmap: Colormap name (default: "viridis")
            figsize: Optional figure size (uses default if None)
        
        Raises:
            ValueError: If scores_by_scorer is empty
            RuntimeError: If heatmap generation fails
        """
        if not scores_by_scorer:
            raise ValueError("scores_by_scorer cannot be empty")
        
        try:
            logger.info(f"Generating scorer heatmap with {len(scores_by_scorer)} scorers")
            
            # Extract unique image IDs and scorer names
            image_ids = set()
            for scorer_ratings in scores_by_scorer.values():
                image_ids.update(scorer_ratings.keys())
            
            image_ids = sorted(image_ids)
            scorer_names = sorted(scores_by_scorer.keys())
            
            n_images = len(image_ids)
            n_scorers = len(scorer_names)
            
            # Create matrix (images x scorers)
            matrix = np.full((n_images, n_scorers), np.nan)
            
            # Fill matrix with ratings
            for scorer_idx, scorer_name in enumerate(scorer_names):
                scorer_ratings = scores_by_scorer[scorer_name]
                for image_idx, image_id in enumerate(image_ids):
                    if image_id in scorer_ratings:
                        matrix[image_idx, scorer_idx] = scorer_ratings[image_id]
            
            # Create figure
            figsize = figsize or (max(8, n_scorers * 2), max(8, n_images * 0.3))
            fig, ax = plt.subplots(figsize=figsize)
            
            # Generate heatmap
            sns.heatmap(
                matrix,
                annot=True,  # Annotate cells with values
                fmt='.2f',
                cmap=cmap,
                square=False,
                linewidths=0.5,
                cbar_kws={'label': 'Rating'},
                xticklabels=scorer_names,
                yticklabels=image_ids,
                ax=ax
            )
            
            # Set title
            if title is None:
                title = f"Scorer Ratings Comparison ({n_images} images, {n_scorers} scorers)"
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Labels
            ax.set_xlabel("Scorer", fontsize=12)
            ax.set_ylabel("Image ID", fontsize=12)
            
            # Rotate labels for readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Scorer heatmap saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate scorer heatmap: {e}")
            raise RuntimeError(f"Scorer heatmap generation failed: {e}") from e
    
    def __repr__(self) -> str:
        """String representation of HeatmapGenerator."""
        return f"HeatmapGenerator(style='{self.style}', figsize={self.default_figsize})"
