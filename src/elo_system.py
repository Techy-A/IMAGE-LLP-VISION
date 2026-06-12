"""
Elo rating system for pairwise image tournament rankings.

This module implements the Elo rating algorithm for updating image ratings
based on match outcomes. The system maintains ratings that reflect relative
image quality through tournament-style comparisons.
"""

import logging
import math
from typing import Dict, List, Tuple, Optional


logger = logging.getLogger(__name__)


class EloSystem:
    """
    Elo rating system for tournament-based image ranking.
    
    The Elo rating system uses a logistic model to calculate the expected
    outcome of matches and updates ratings based on actual outcomes. Higher
    ratings indicate higher quality images.
    
    Attributes:
        k_factor: Sensitivity of rating changes (higher = more volatile)
        initial_rating: Starting rating for new images
        ratings: Dictionary mapping image IDs to current ratings
    """
    
    def __init__(self, k_factor: int = 32, initial_rating: float = 1500.0):
        """
        Initialize Elo rating system.
        
        Args:
            k_factor: Maximum rating change per match (default: 32)
                     Higher values make ratings more volatile
                     Common values: 16 (stable), 32 (standard), 64 (volatile)
            initial_rating: Starting rating for new images (default: 1500.0)
                           Standard Elo starts at 1500
        
        Raises:
            ValueError: If k_factor or initial_rating are invalid
        """
        if k_factor <= 0:
            raise ValueError(f"k_factor must be positive, got {k_factor}")
        
        if initial_rating <= 0:
            raise ValueError(f"initial_rating must be positive, got {initial_rating}")
        
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = {}
        
        logger.info(
            f"Elo system initialized with k_factor={k_factor}, "
            f"initial_rating={initial_rating}"
        )
    
    def update_ratings(
        self,
        winner_id: str,
        loser_id: str,
        score: float = 1.0
    ) -> None:
        """
        Update Elo ratings based on match result.
        
        The winner's rating increases and the loser's rating decreases.
        The magnitude of change depends on:
        - The rating difference (upset wins cause larger changes)
        - The k_factor (rating volatility parameter)
        - The score (for partial wins/ties)
        
        Args:
            winner_id: ID of the winning image
            loser_id: ID of the losing image
            score: Match score in [0.0, 1.0] where 1.0 = complete win,
                   0.5 = tie, values between indicate partial wins
                   (default: 1.0 for complete win)
        
        Raises:
            ValueError: If winner_id equals loser_id or score invalid
        """
        if winner_id == loser_id:
            raise ValueError(
                f"winner_id and loser_id must be different, "
                f"both are '{winner_id}'"
            )
        
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"score must be in [0.0, 1.0], got {score}")
        
        # Initialize ratings if needed (first time we see this image)
        if winner_id not in self.ratings:
            self.ratings[winner_id] = self.initial_rating

        if loser_id not in self.ratings:
            self.ratings[loser_id] = self.initial_rating

        # Current ratings before this match
        winner_rating = self.ratings[winner_id]
        loser_rating = self.ratings[loser_id]

        # ── Elo expected-score calculation ───────────────────────────────
        # E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        # The "400" divisor means a 400-point rating gap → expected win rate
        # of ~91%. A 200-point gap → ~76%. Equal ratings → exactly 0.5.
        # loser_expected = 1 - winner_expected (probabilities sum to 1).
        winner_expected = self._expected_score(winner_rating, loser_rating)
        loser_expected = 1.0 - winner_expected

        # Actual scores:
        #   winner gets `score` (should be >= 0.5, representing degree of victory)
        #   loser gets `1 - score`
        # Using score < 1.0 (e.g. 0.7) means the winner had a narrow victory,
        # which results in a smaller rating gain than a decisive win (score=1.0).
        winner_actual = score
        loser_actual = 1.0 - score

        # Rating changes = k_factor * (actual - expected)
        # If winner was already heavily favoured, winner_expected ≈ 1.0 and
        # the change is small. If it’s an upset, the expected was low and
        # the change is large. Both images' ratings shift symmetrically.
        winner_change = self.k_factor * (winner_actual - winner_expected)
        loser_change = self.k_factor * (loser_actual - loser_expected)

        # Apply rating changes
        self.ratings[winner_id] = winner_rating + winner_change
        self.ratings[loser_id] = loser_rating + loser_change

        logger.debug(
            f"Elo update: {winner_id} ({winner_rating:.1f} -> "
            f"{self.ratings[winner_id]:.1f}, +{winner_change:.1f}), "
            f"{loser_id} ({loser_rating:.1f} -> "
            f"{self.ratings[loser_id]:.1f}, {loser_change:.1f})"
        )
    
    def get_rating(self, image_id: str) -> float:
        """
        Get current Elo rating for an image.
        
        Args:
            image_id: ID of the image
        
        Returns:
            Current Elo rating (or initial_rating if image not rated yet)
        """
        return self.ratings.get(image_id, self.initial_rating)
    
    def get_rankings(self) -> List[Tuple[str, float]]:
        """
        Get sorted rankings by Elo rating.
        
        Returns:
            List of (image_id, rating) tuples sorted by rating (descending)
            Highest rated images appear first
        """
        rankings = sorted(
            self.ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return rankings
    
    def get_all_ratings(self) -> Dict[str, float]:
        """
        Get all current ratings.
        
        Returns:
            Dictionary mapping image IDs to ratings
        """
        return self.ratings.copy()
    
    def set_rating(self, image_id: str, rating: float) -> None:
        """
        Manually set rating for an image.
        
        Useful for restoring ratings from checkpoints.
        
        Args:
            image_id: ID of the image
            rating: New rating value
        
        Raises:
            ValueError: If rating is invalid
        """
        if rating <= 0:
            raise ValueError(f"rating must be positive, got {rating}")
        
        self.ratings[image_id] = rating
    
    def reset(self) -> None:
        """
        Reset all ratings to initial state.
        
        Clears all rating history.
        """
        self.ratings.clear()
        logger.info("Elo ratings reset")
    
    @staticmethod
    def _expected_score(rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for image A vs image B.

        Standard Elo formula:
            E_A = 1 / (1 + 10^((R_B - R_A) / 400))

        The divisor 400 is a historical convention from chess Elo that makes a
        400-point gap correspond to roughly a 10:1 win odds ratio. This divisor
        is consistent between Arena's Elo updates and this method.

        Args:
            rating_a: Rating of image A
            rating_b: Rating of image B

        Returns:
            Expected score for A in [0.0, 1.0]
            0.5 = equal match, >0.5 = A favored, <0.5 = B favored
        """
        exponent = (rating_b - rating_a) / 400.0
        expected = 1.0 / (1.0 + math.pow(10, exponent))

        return expected
    
    def get_expected_score(self, image_a_id: str, image_b_id: str) -> float:
        """
        Get expected score for a match between two images.
        
        Args:
            image_a_id: ID of first image
            image_b_id: ID of second image
        
        Returns:
            Expected score for image A in [0.0, 1.0]
        """
        rating_a = self.get_rating(image_a_id)
        rating_b = self.get_rating(image_b_id)
        
        return self._expected_score(rating_a, rating_b)
    
    def __repr__(self) -> str:
        """String representation of Elo system."""
        return (
            f"EloSystem(k_factor={self.k_factor}, "
            f"initial_rating={self.initial_rating}, "
            f"num_rated_images={len(self.ratings)})"
        )
