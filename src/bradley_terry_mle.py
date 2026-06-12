"""
Bradley-Terry Maximum Likelihood Estimation for skill ratings.

This module implements the Bradley-Terry model with MLE to compute skill
ratings from pairwise comparison data. The Bradley-Terry model assumes that
the probability of item i beating item j is:

    P(i > j) = skill_i / (skill_i + skill_j)

MLE is used to find the skill ratings that maximize the likelihood of
observed match outcomes.
"""

import logging
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


logger = logging.getLogger(__name__)


class BradleyTerryMLE:
    """
    Bradley-Terry Maximum Likelihood Estimation for skill ratings.
    
    Computes skill ratings from pairwise comparison results using an
    iterative MLE algorithm. Includes regularization to handle images
    with zero wins.
    
    Attributes:
        beta: Regularization parameter for zero-win images
        max_iterations: Maximum iterations for MLE convergence
        tolerance: Convergence tolerance for rating changes
    """
    
    def __init__(
        self,
        beta: float = 1e-10,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ):
        """
        Initialize Bradley-Terry MLE system.
        
        Args:
            beta: Regularization parameter for images with 0 wins (default: 1e-10)
                  Prevents division by zero, adds pseudo-count to wins
            max_iterations: Maximum iterations for MLE (default: 100)
            tolerance: Convergence tolerance (default: 1e-6)
                       Algorithm stops when rating changes < tolerance
        
        Raises:
            ValueError: If parameters are invalid
        """
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")
        
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
        
        self.beta = beta
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        logger.info(
            f"Bradley-Terry MLE initialized with beta={beta}, "
            f"max_iterations={max_iterations}, tolerance={tolerance}"
        )
    
    def fit(
        self,
        matchups: List[Tuple[str, str, float]],
        image_ids: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute skill ratings from pairwise comparisons via MLE.
        
        Uses iterative algorithm to find maximum likelihood skill estimates:
        1. Initialize all skills to 1.0
        2. Iteratively update skills based on win/loss statistics
        3. Normalize so sum of skills equals number of images
        
        Args:
            matchups: List of (image_a_id, image_b_id, score) tuples
                     score in [0.0, 1.0] where >0.5 means image_a won
            image_ids: Optional list of all image IDs in the tournament.
                       If provided, ensures all images receive skill ratings
                       even if they did not participate in any matchups.
        
        Returns:
            Dictionary mapping image_id to skill rating
            Higher skill = higher quality image
        
        Raises:
            ValueError: If matchups is empty or contains invalid data
        """
        if not matchups:
            raise ValueError("matchups cannot be empty")
        
        logger.info(f"Fitting Bradley-Terry MLE on {len(matchups)} matchups")

        # Extract all unique image IDs
        if image_ids is None:
            extracted_ids = set()
            for image_a, image_b, score in matchups:
                extracted_ids.add(image_a)
                extracted_ids.add(image_b)
            image_ids_list = sorted(extracted_ids)
        else:
            image_ids_list = sorted(set(image_ids))

        n_images = len(image_ids_list)

        logger.info(f"Found {n_images} unique images")

        # Compute win statistics: fractional wins per image and comparison counts
        wins, comparisons = self._compute_statistics(matchups)

        # Initialize all skills to 1.0 (uniform prior before seeing any data)
        skills = {image_id: 1.0 for image_id in image_ids_list}

        # ── Iterative MLE update (Hunter 2004 MM algorithm) ───────────────
        # The Bradley-Terry log-likelihood is concave; Hunter's minorization-
        # maximization gives a fixed-point iteration that is guaranteed to
        # converge to the global maximum.
        #
        # Update rule for skill_i:
        #
        #   skill_i = (wins_i + β) / ∑_j [ n_ij / (skill_i + skill_j) ]
        #
        # where:
        #   wins_i   = total fractional wins for image i across all matches
        #   n_ij     = number of matches between i and j (in either order)
        #   β       = regularisation pseudo-count (prevents zero-skill)
        for iteration in range(self.max_iterations):
            old_skills = skills.copy()

            for image_id in image_ids_list:
                # Denominator: sum over all opponents j of n_ij / (s_i + s_j)
                # This accumulates the "difficulty" of all of image_i's matches.
                denominator = 0.0

                for opponent_id in image_ids_list:
                    if opponent_id == image_id:
                        continue

                    # Count comparisons between i and opponent in both orderings
                    n_comparisons = (
                        comparisons.get((image_id, opponent_id), 0) +
                        comparisons.get((opponent_id, image_id), 0)
                    )

                    if n_comparisons > 0:
                        denominator += n_comparisons / (
                            skills[image_id] + skills[opponent_id]
                        )

                # Numerator: wins + β to regularise images with 0 real wins
                # (prevents their skill converging to 0, which would cause
                # division by zero in subsequent iterations).
                numerator = wins.get(image_id, 0.0) + self.beta
                denominator = denominator + self.beta

                # Apply update
                skills[image_id] = numerator / denominator

            # ── Normalise so skills sum to n_images ───────────────────────
            # Without normalisation the absolute scale drifts each iteration.
            # Normalising to sum=n_images makes skill ≈1.0 mean "average image".
            total_skill = sum(skills.values())
            for image_id in skills:
                skills[image_id] = (skills[image_id] / total_skill) * n_images

            # Convergence: stop early if no skill changed more than `tolerance`
            max_change = max(
                abs(skills[img_id] - old_skills[img_id])
                for img_id in image_ids_list
            )

            if max_change < self.tolerance:
                logger.info(
                    f"Bradley-Terry MLE converged after {iteration + 1} iterations "
                    f"(max_change={max_change:.2e})"
                )
                break
        else:
            logger.warning(
                f"Bradley-Terry MLE did not converge after {self.max_iterations} "
                f"iterations (max_change={max_change:.2e})"
            )

        return skills
    
    def _compute_statistics(
        self,
        matchups: List[Tuple[str, str, float]]
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], int]]:
        """
        Compute win statistics from matchups.
        
        Args:
            matchups: List of (image_a_id, image_b_id, score) tuples
        
        Returns:
            Tuple of:
            - wins: Dict mapping image_id to total wins
            - comparisons: Dict mapping (image_a, image_b) to comparison count
        """
        wins: Dict[str, float] = defaultdict(float)
        comparisons: Dict[Tuple[str, str], int] = defaultdict(int)

        for image_a, image_b, score in matchups:
            # Validate score
            if not (0.0 <= score <= 1.0):
                logger.warning(
                    f"Invalid score {score} for matchup ({image_a}, {image_b}). "
                    f"Skipping."
                )
                continue

            # Accumulate *fractional* wins: score=0.7 means image_a contributed
            # 0.7 wins and image_b contributed 0.3 wins. This allows the
            # Bradley-Terry model to use soft/continuous match outcomes rather
            # than binary wins, which is important because our ensemble scores
            # are rarely exactly 0 or 1.
            wins[image_a] += score
            wins[image_b] += (1.0 - score)

            # Track how many times each (image_a, image_b) pair occurred
            # (direction matters when looking up per-pair counts in fit())
            comparisons[(image_a, image_b)] += 1

        return dict(wins), dict(comparisons)
    
    def get_rankings(self, skills: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Get sorted rankings from skill ratings.
        
        Args:
            skills: Dictionary mapping image IDs to skill ratings
        
        Returns:
            List of (image_id, skill) tuples sorted by skill (descending)
        """
        rankings = sorted(
            skills.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return rankings
    
    def __repr__(self) -> str:
        """String representation of Bradley-Terry MLE."""
        return (
            f"BradleyTerryMLE(beta={self.beta}, "
            f"max_iterations={self.max_iterations}, "
            f"tolerance={self.tolerance})"
        )
