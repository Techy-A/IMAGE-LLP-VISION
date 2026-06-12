"""
Ensemble aggregator for combining multiple scorer outputs.

This module implements weighted ensemble aggregation to combine scores
from multiple image quality scorers into a single consensus score.
Handles scorer failures by dynamically renormalizing weights.
"""

import logging
import math
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


class Ensemble:
    """
    Weighted ensemble aggregator for multiple scorers.
    
    Combines scores from different quality metrics using weighted averaging.
    Automatically handles scorer failures by excluding failed scorers and
    renormalizing weights among successful scorers.
    
    Attributes:
        weights: Dictionary mapping scorer names to weights
        _original_weights: Original weights for reference
    """
    
    def __init__(self, weights: Dict[str, float]):
        """
        Initialize ensemble aggregator with scorer weights.
        
        Args:
            weights: Dictionary mapping scorer name to weight
                    Example: {"vqa": 0.3, "pickscore": 0.2, "image_reward": 0.2,
                             "vlm_judge": 0.3}
                    Weights must sum to 1.0 (within floating point tolerance)
        
        Raises:
            ValueError: If weights are invalid (empty, negative, don't sum to 1.0)
        """
        if not weights:
            raise ValueError("weights cannot be empty")
        
        # Validate all weights are non-negative
        for scorer_name, weight in weights.items():
            if weight < 0:
                raise ValueError(
                    f"Weight for scorer '{scorer_name}' must be non-negative, "
                    f"got {weight}"
                )
        
        # Validate weights sum to 1.0 (with floating point tolerance)
        weight_sum = sum(weights.values())
        if not math.isclose(weight_sum, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Scorer weights must sum to 1.0, got {weight_sum}. "
                f"Weights: {weights}"
            )
        
        self.weights = weights.copy()
        self._original_weights = weights.copy()
        
        logger.info(f"Ensemble initialized with weights: {self.weights}")
    
    def aggregate_scores(self, scores: Dict[str, float]) -> float:
        """
        Aggregate multiple scorer outputs into single score.
        
        Computes weighted average of all provided scores. If some scorers
        failed (not present in scores dict), their weights are excluded
        and remaining weights are renormalized.
        
        Args:
            scores: Dictionary mapping scorer name to score in [0.0, 1.0]
                   May contain subset of scorers if some failed
        
        Returns:
            Aggregated score in [0.0, 1.0]
            Returns 0.5 (neutral) if no valid scores available
        
        Raises:
            ValueError: If scores contains invalid values
        """
        if not scores:
            logger.warning("No scores provided for aggregation, returning neutral 0.5")
            return 0.5

        # Validate all scores are in valid range
        for scorer_name, score in scores.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(
                    f"Score for scorer '{scorer_name}' must be in [0.0, 1.0], "
                    f"got {score}"
                )

        # Identify which scorers succeeded this match
        available_scorers = set(scores.keys())
        expected_scorers = set(self.weights.keys())
        missing_scorers = expected_scorers - available_scorers

        if missing_scorers:
            logger.warning(
                f"Scorers {missing_scorers} failed or missing. "
                f"Renormalizing weights among available scorers."
            )

        # ── Weight renormalization ────────────────────────────────────────
        # Only keep weights for scorers that actually produced a score.
        # This is purely a proportional rescaling: if the original weights
        # were {vqa: 0.3, pickscore: 0.2, hpsv2: 0.5} and vqa failed,
        # the surviving weights {pickscore: 0.2, hpsv2: 0.5} are renormalised
        # to {pickscore: 0.286, hpsv2: 0.714}. The final score is still a
        # proper weighted average in [0, 1]. Scorers not present in
        # self.weights at all (e.g. unknown keys) are silently ignored.
        available_weights = {
            name: self.weights[name]
            for name in available_scorers
            if name in self.weights
        }

        if not available_weights:
            # Every scorer that produced output is unknown to the config;
            # fall back to neutral (no information).
            logger.error(
                "No weights found for available scorers. "
                "Returning neutral score 0.5"
            )
            return 0.5

        weight_sum = sum(available_weights.values())
        if weight_sum == 0:
            # All surviving scorers have zero weight; extremely unlikely but safe.
            logger.error("Sum of available weights is zero. Returning neutral 0.5")
            return 0.5

        # Rescale so surviving weights sum to 1.0
        normalized_weights = {
            name: weight / weight_sum
            for name, weight in available_weights.items()
        }

        # Weighted average over all scorers that succeeded
        aggregated_score = sum(
            normalized_weights[name] * scores[name]
            for name in available_weights.keys()
        )

        # Clamp to [0, 1] to defend against floating-point edge cases
        aggregated_score = max(0.0, min(1.0, aggregated_score))

        logger.debug(
            f"Aggregated {len(scores)} scores: {scores} "
            f"with weights {normalized_weights} -> {aggregated_score:.4f}"
        )

        return aggregated_score
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get current scorer weights.
        
        Returns:
            Copy of current weights dictionary
        """
        return self.weights.copy()
    
    def get_original_weights(self) -> Dict[str, float]:
        """
        Get original scorer weights (before any modifications).
        
        Returns:
            Copy of original weights dictionary
        """
        return self._original_weights.copy()
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update scorer weights.
        
        Useful for adjusting ensemble behavior during runtime or
        excluding unreliable scorers.
        
        Args:
            new_weights: New weights dictionary (must sum to 1.0)
        
        Raises:
            ValueError: If new_weights are invalid
        """
        # Validate new weights
        if not new_weights:
            raise ValueError("new_weights cannot be empty")
        
        for scorer_name, weight in new_weights.items():
            if weight < 0:
                raise ValueError(
                    f"Weight for scorer '{scorer_name}' must be non-negative, "
                    f"got {weight}"
                )
        
        weight_sum = sum(new_weights.values())
        if not math.isclose(weight_sum, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Scorer weights must sum to 1.0, got {weight_sum}"
            )
        
        old_weights = self.weights.copy()
        self.weights = new_weights.copy()
        
        logger.info(f"Weights updated from {old_weights} to {self.weights}")
    
    def exclude_scorer(self, scorer_name: str) -> None:
        """
        Exclude a scorer and renormalize remaining weights.
        
        Useful for removing consistently failing or unreliable scorers.
        
        Args:
            scorer_name: Name of scorer to exclude
        
        Raises:
            ValueError: If scorer_name not in weights or would leave no scorers
        """
        if scorer_name not in self.weights:
            raise ValueError(
                f"Scorer '{scorer_name}' not found in weights: {self.weights.keys()}"
            )
        
        if len(self.weights) == 1:
            raise ValueError(
                f"Cannot exclude scorer '{scorer_name}' - it's the only scorer"
            )
        
        # Remove scorer
        del self.weights[scorer_name]
        
        # Renormalize remaining weights
        weight_sum = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= weight_sum
        
        logger.info(
            f"Excluded scorer '{scorer_name}'. "
            f"Renormalized weights: {self.weights}"
        )
    
    def reset_weights(self) -> None:
        """
        Reset weights to original configuration.
        """
        self.weights = self._original_weights.copy()
        logger.info(f"Weights reset to original: {self.weights}")
    
    def __repr__(self) -> str:
        """String representation of Ensemble."""
        return f"Ensemble(weights={self.weights})"
