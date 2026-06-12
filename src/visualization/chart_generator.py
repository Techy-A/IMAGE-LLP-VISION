"""
Chart generation for tournament visualization.

This module provides chart visualizations including Elo progression,
final rankings, and score distributions using matplotlib.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)


class ChartGenerator:
    """
    Generator for tournament chart visualizations.
    
    Creates charts showing:
    - Elo rating progression over time
    - Final rankings as bar chart
    - Score distribution histograms
    
    Uses matplotlib for flexible chart rendering with proper
    labels, titles, and legends.
    """
    
    def __init__(self, style: str = "whitegrid", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize chart generator.
        
        Args:
            style: Seaborn style (default: "whitegrid")
            figsize: Default figure size as (width, height)
        """
        self.style = style
        self.default_figsize = figsize
        
        # Set style
        sns.set_style(style)
        
        logger.info(f"ChartGenerator initialized with style: {style}")
    
    def generate_elo_progression(
        self,
        history: Dict[str, List[float]],
        output_path: str,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        max_lines: int = 20
    ) -> None:
        """
        Generate line chart showing Elo rating progression over time.
        
        Args:
            history: Dict mapping image_id to list of ratings over time
            output_path: Path to save chart image
            title: Optional custom title (default: "Elo Rating Progression")
            figsize: Optional figure size (uses default if None)
            max_lines: Maximum number of image lines to show (default: 20)
                      Shows top N by final rating to avoid clutter
        
        Raises:
            ValueError: If history is empty
            RuntimeError: If chart generation fails
        """
        if not history:
            raise ValueError("history cannot be empty")
        
        try:
            logger.info(f"Generating Elo progression chart for {len(history)} images")
            
            # Sort images by final rating and take top max_lines
            image_ids = sorted(
                history.keys(),
                key=lambda img_id: history[img_id][-1] if history[img_id] else 0,
                reverse=True
            )[:max_lines]
            
            # Create figure
            figsize = figsize or self.default_figsize
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot each image's progression
            for image_id in image_ids:
                ratings = history[image_id]
                iterations = list(range(len(ratings)))
                ax.plot(iterations, ratings, marker='o', markersize=3, 
                       label=image_id, linewidth=2, alpha=0.7)
            
            # Set title
            if title is None:
                title = f"Elo Rating Progression (Top {len(image_ids)} Images)"
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Labels
            ax.set_xlabel("Match Number", fontsize=12)
            ax.set_ylabel("Elo Rating", fontsize=12)
            
            # Grid
            ax.grid(True, alpha=0.3)
            
            # Legend (place outside plot area if many lines)
            if len(image_ids) <= 10:
                ax.legend(loc='best', fontsize=9)
            else:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Elo progression chart saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate Elo progression chart: {e}")
            raise RuntimeError(f"Elo progression chart generation failed: {e}") from e
    
    def generate_final_rankings(
        self,
        rankings: List[Tuple[str, float]],
        output_path: str,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        top_n: Optional[int] = None,
        color: str = "steelblue"
    ) -> None:
        """
        Generate bar chart of final rankings.
        
        Args:
            rankings: List of (image_id, rating) tuples sorted by rating (descending)
            output_path: Path to save chart image
            title: Optional custom title (default: "Final Rankings")
            figsize: Optional figure size (uses default if None)
            top_n: Optional limit to show only top N images
            color: Bar color (default: "steelblue")
        
        Raises:
            ValueError: If rankings is empty
            RuntimeError: If chart generation fails
        """
        if not rankings:
            raise ValueError("rankings cannot be empty")
        
        try:
            logger.info(f"Generating final rankings chart for {len(rankings)} images")
            
            # Limit to top_n if specified
            if top_n is not None and top_n > 0:
                rankings = rankings[:top_n]
            
            # Extract image IDs and ratings
            image_ids = [img_id for img_id, _ in rankings]
            ratings = [rating for _, rating in rankings]
            
            # Create figure
            figsize = figsize or (max(10, len(rankings) * 0.5), 8)
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create bar chart
            y_pos = np.arange(len(image_ids))
            bars = ax.barh(y_pos, ratings, color=color, alpha=0.8, edgecolor='black')
            
            # Add value labels on bars
            for i, (bar, rating) in enumerate(zip(bars, ratings)):
                ax.text(rating, i, f' {rating:.2f}', 
                       va='center', ha='left', fontsize=9)
            
            # Set labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(image_ids)
            
            # Invert y-axis (highest rating at top)
            ax.invert_yaxis()
            
            # Set title
            if title is None:
                title = f"Final Rankings (Top {len(rankings)} Images)"
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Labels
            ax.set_xlabel("Rating", fontsize=12)
            ax.set_ylabel("Image ID", fontsize=12)
            
            # Grid
            ax.grid(True, axis='x', alpha=0.3)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Final rankings chart saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate final rankings chart: {e}")
            raise RuntimeError(f"Final rankings chart generation failed: {e}") from e
    
    def generate_score_distribution(
        self,
        scores: List[float],
        output_path: str,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        bins: int = 30,
        color: str = "skyblue"
    ) -> None:
        """
        Generate histogram of score distribution.
        
        Shows distribution of match scores to understand overall
        competition balance.
        
        Args:
            scores: List of match scores in [0.0, 1.0]
            output_path: Path to save chart image
            title: Optional custom title (default: "Score Distribution")
            figsize: Optional figure size (uses default if None)
            bins: Number of histogram bins (default: 30)
            color: Histogram color (default: "skyblue")
        
        Raises:
            ValueError: If scores is empty
            RuntimeError: If chart generation fails
        """
        if not scores:
            raise ValueError("scores cannot be empty")
        
        try:
            logger.info(f"Generating score distribution chart for {len(scores)} scores")
            
            # Create figure
            figsize = figsize or self.default_figsize
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create histogram
            n, bins_array, patches = ax.hist(
                scores,
                bins=bins,
                color=color,
                alpha=0.7,
                edgecolor='black',
                range=(0.0, 1.0)
            )
            
            # Add mean line
            mean_score = np.mean(scores)
            ax.axvline(mean_score, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_score:.3f}')
            
            # Add median line
            median_score = np.median(scores)
            ax.axvline(median_score, color='green', linestyle='--', linewidth=2,
                      label=f'Median: {median_score:.3f}')
            
            # Set title
            if title is None:
                title = f"Score Distribution ({len(scores)} matches)"
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Labels
            ax.set_xlabel("Match Score", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            
            # Legend
            ax.legend(loc='best', fontsize=10)
            
            # Grid
            ax.grid(True, axis='y', alpha=0.3)
            
            # Set x-axis limits
            ax.set_xlim(0.0, 1.0)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Score distribution chart saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate score distribution chart: {e}")
            raise RuntimeError(f"Score distribution chart generation failed: {e}") from e
    
    def __repr__(self) -> str:
        """String representation of ChartGenerator."""
        return f"ChartGenerator(style='{self.style}', figsize={self.default_figsize})"
