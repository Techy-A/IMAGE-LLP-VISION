"""
Arena orchestration for round-robin image tournament execution.

This module implements the Arena controller that coordinates all tournament
operations including match generation, scorer invocation, rating updates,
checkpoint management, and graceful scorer-failure handling.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from itertools import combinations

from PIL.Image import Image as PILImage

from src.data_models import (
    ImageMetadata, MatchResult, TournamentResult, TournamentMetadata,
    CheckpointState
)
from src.scoring.base_scorer import BaseScorer
from src.elo_system import EloSystem
from src.bradley_terry_mle import BradleyTerryMLE
from src.ensemble import Ensemble
from src.checkpoint_manager import CheckpointManager
from src.model_rotator import ModelRotator
from src.image_loader import ImageLoader


logger = logging.getLogger(__name__)


class Arena:
    """
    Main tournament orchestration controller.
    
    Coordinates round-robin tournament execution by:
    - Generating all pairwise image comparisons
    - Invoking scorers for each match
    - Aggregating scores via ensemble
    - Updating Elo ratings after each match
    - Saving checkpoints periodically
    - Computing final Bradley-Terry MLE ratings
    - Handling scorer failures by recording them and renormalising the ensemble
    
    Attributes:
        images: List of images in tournament
        scorers: Dictionary of scorer instances by name
        elo: Elo rating system
        bt_mle: Bradley-Terry MLE system
        ensemble: Ensemble aggregator
        checkpoint_manager: Checkpoint persistence
        model_rotator: Model memory management
        config: Tournament configuration
    """
    
    def __init__(
        self,
        images: List[ImageMetadata],
        scorers: Dict[str, BaseScorer],
        elo_system: EloSystem,
        bt_mle: BradleyTerryMLE,
        ensemble: Ensemble,
        checkpoint_manager: CheckpointManager,
        model_rotator: Optional[ModelRotator],
        config: Dict[str, Any]
    ):
        """
        Initialize Arena with tournament components.
        
        Args:
            images: List of ImageMetadata for images in tournament
            scorers: Dictionary mapping scorer name to scorer instance
            elo_system: Elo rating system instance (template for per-group systems)
            bt_mle: Bradley-Terry MLE instance
            ensemble: Ensemble aggregator instance
            checkpoint_manager: Checkpoint manager instance
            model_rotator: Optional model rotator for memory management
            config: Configuration dictionary with tournament settings
        
        Raises:
            ValueError: If inputs are invalid
        """
        if not images:
            raise ValueError("images list cannot be empty")
        
        if not scorers:
            raise ValueError("scorers dictionary cannot be empty")
        
        self.images = images
        self.scorers = scorers
        self.bt_mle = bt_mle
        self.ensemble = ensemble
        self.checkpoints = checkpoint_manager
        self.rotator = model_rotator
        self.config = config
        
        # Image loader for loading image files
        self.image_loader = ImageLoader()
        
        # ── Per-group Elo systems ─────────────────────────────────────────
        # Each XLSX sheet (or CSV group) is a separate evaluation context:
        # e.g. "Mermaid Metric" vs "Underwatercity Metric" compare completely
        # different prompts. Mixing them into a single Elo pool would let a
        # great image from group A steal rating points from a good image in
        # group B. Instead, every group gets its own fresh EloSystem whose
        # ratings are only updated by within-group matches.
        self.groups: Dict[str, List[ImageMetadata]] = {}
        self.elo_systems: Dict[str, EloSystem] = {}

        for img in images:
            group_name = img.group if img.group else "_ungrouped"
            if group_name not in self.groups:
                self.groups[group_name] = []
                # Clone Elo hyper-parameters (k_factor, initial_rating) from
                # the template instance passed by the CLI; we never USE the
                # template object itself.
                self.elo_systems[group_name] = EloSystem(
                    k_factor=elo_system.k_factor,
                    initial_rating=elo_system.initial_rating
                )
            self.groups[group_name].append(img)

        logger.info(
            f"Arena initialized with {len(images)} images across "
            f"{len(self.groups)} groups, {len(scorers)} scorers"
        )
        for group_name, group_images in self.groups.items():
            logger.info(f"  Group '{group_name}': {len(group_images)} images")

        # Tournament state — mutable as matches are processed
        self.completed_matches: List[MatchResult] = []
        self.pending_pairs: List[Tuple[str, str]] = []
        self.metadata = TournamentMetadata(
            start_time=datetime.now(),
            total_images=len(images),
            config_snapshot=config.copy()
        )

        # Reverse index: image_id → group_name, used inside run_tournament
        # to route each match result to the correct per-group EloSystem.
        self.image_to_group: Dict[str, str] = {}
        for group_name, group_images in self.groups.items():
            for img in group_images:
                self.image_to_group[img.id] = group_name
    
    def run_tournament(self) -> TournamentResult:
        """
        Execute full round-robin tournament.
        
        Generates all pairwise combinations, runs matches, updates ratings,
        saves checkpoints, and computes final rankings.
        
        Returns:
            TournamentResult with matches, ratings, and rankings
        
        Raises:
            RuntimeError: If tournament execution fails
        """
        try:
            logger.info("Starting round-robin tournament")
            
            # Generate all unique pairs if not resuming
            if not self.pending_pairs:
                self.pending_pairs = self._generate_pairs()
                logger.info(f"Generated {len(self.pending_pairs)} unique pairs")
            
            total_matches = len(self.pending_pairs) + len(self.completed_matches)
            self.metadata.total_matches = total_matches
            
            # Run all matches
            checkpoint_interval = self.config.get('checkpoint_every', 100)
            
            while self.pending_pairs:
                # Get next pair
                image_a_id, image_b_id = self.pending_pairs.pop(0)
                
                # Run match
                match_result = self.run_match(image_a_id, image_b_id)
                
                # Record match
                self.completed_matches.append(match_result)
                
                # ── Winner-perspective Elo flip ───────────────────────────
                # ensemble_score is ALWAYS from image_a's point of view:
                #   > 0.5 → image_a won    < 0.5 → image_b won
                #
                # EloSystem.update_ratings(score=…) interprets the score as
                # the WINNER's actual result (1.0 = decisive win, 0.5 = draw).
                # If we pass the raw ensemble_score when image_b wins, we would
                # be telling Elo "the winner achieved 0.3" (a loss value), which
                # causes the loser to GAIN rating points — the exact opposite of
                # what we want. Flipping to (1.0 - ensemble_score) converts the
                # score to the winner's perspective before it is passed down.
                if match_result.ensemble_score > 0.5:
                    winner_id = image_a_id
                    loser_id = image_b_id
                    winner_score = match_result.ensemble_score          # already > 0.5
                else:
                    winner_id = image_b_id
                    loser_id = image_a_id
                    winner_score = 1.0 - match_result.ensemble_score   # flip to winner perspective

                # Route this match's Elo update to the correct per-group system.
                # Both images are always in the same group (pairs are
                # generated within groups in _generate_pairs), so looking up
                # image_a's group is sufficient.
                group_name = self.image_to_group.get(image_a_id, "_ungrouped")
                elo_system = self.elo_systems[group_name]

                elo_system.update_ratings(
                    winner_id=winner_id,
                    loser_id=loser_id,
                    score=winner_score
                )
                
                # Save checkpoint periodically
                if len(self.completed_matches) % checkpoint_interval == 0:
                    self._save_checkpoint()
                
                # Log progress
                if len(self.completed_matches) % 10 == 0:
                    progress = (len(self.completed_matches) / total_matches) * 100
                    logger.info(
                        f"Tournament progress: {len(self.completed_matches)}/{total_matches} "
                        f"matches ({progress:.1f}%)"
                    )
            
            # Compute final Bradley-Terry MLE ratings per group
            logger.info("Computing Bradley-Terry MLE ratings per group")
            
            # Aggregate all Elo ratings from all groups
            all_elo_ratings: Dict[str, float] = {}
            for elo_system in self.elo_systems.values():
                all_elo_ratings.update(elo_system.get_all_ratings())
            
            # Compute BT-MLE per group
            all_bt_ratings: Dict[str, float] = {}
            all_rankings: List[Tuple[str, float]] = []
            
            for group_name, group_images in self.groups.items():
                # Filter matches for this group
                group_image_ids = {img.id for img in group_images}
                group_matchups = [
                    (match.image_a_id, match.image_b_id, match.ensemble_score)
                    for match in self.completed_matches
                    if match.image_a_id in group_image_ids and match.image_b_id in group_image_ids
                ]
                
                if group_matchups:
                    # Compute BT-MLE for this group
                    image_ids = [img.id for img in group_images]
                    group_bt_ratings = self.bt_mle.fit(group_matchups, image_ids=image_ids)
                    all_bt_ratings.update(group_bt_ratings)
                    
                    # Get rankings for this group
                    group_rankings = self.bt_mle.get_rankings(group_bt_ratings)
                    all_rankings.extend(group_rankings)
                    
                    logger.info(
                        f"  Group '{group_name}': {len(group_matchups)} matches, "
                        f"{len(group_rankings)} ranked images"
                    )
                else:
                    # No matches for this group (shouldn't happen in normal flow)
                    logger.warning(f"  Group '{group_name}': no matches found")
                    for img in group_images:
                        all_bt_ratings[img.id] = 1.0  # Default neutral rating
            
            # Sort all rankings by BT rating (descending)
            all_rankings.sort(key=lambda x: x[1], reverse=True)
            
            # Update metadata
            self.metadata.end_time = datetime.now()
            
            # Create result
            result = TournamentResult(
                matches=self.completed_matches,
                elo_ratings=all_elo_ratings,
                bt_ratings=all_bt_ratings,
                final_rankings=all_rankings,
                metadata=self.metadata
            )
            
            logger.info(
                f"Tournament completed: {len(self.completed_matches)} matches, "
                f"{len(all_rankings)} ranked images"
            )
            
            # Save final checkpoint
            self._save_checkpoint()
            
            return result
            
        except Exception as e:
            logger.error(f"Tournament execution failed: {e}")
            # Save emergency checkpoint
            try:
                self._save_checkpoint()
            except Exception as checkpoint_error:
                logger.error(f"Failed to save emergency checkpoint: {checkpoint_error}")
            raise RuntimeError(f"Tournament execution failed: {e}") from e
    
    def run_match(self, image_a_id: str, image_b_id: str) -> MatchResult:
        """
        Run single pairwise comparison across all scorers.
        
        Args:
            image_a_id: ID of first image
            image_b_id: ID of second image
        
        Returns:
            MatchResult with scores from all scorers
        
        Raises:
            RuntimeError: If match execution fails
        """
        try:
            logger.debug(f"Running match: {image_a_id} vs {image_b_id}")
            
            # DEBUG: Only log detailed scorer output for first 3 matches
            is_early_match = len(self.completed_matches) < 3
            if is_early_match:
                logger.info(f"🔍 DEBUG Match #{len(self.completed_matches)+1}: {image_a_id} vs {image_b_id}")
            
            # Load images
            image_a = self._load_image(image_a_id)
            image_b = self._load_image(image_b_id)
            
            if image_a is None or image_b is None:
                raise RuntimeError(
                    f"Failed to load images for match: {image_a_id}, {image_b_id}"
                )
            
            # Get image metadata for prompt-based scorers
            image_a_meta = next((img for img in self.images if img.id == image_a_id), None)
            image_b_meta = next((img for img in self.images if img.id == image_b_id), None)
            
            # ── Scorer dispatch ───────────────────────────────────────────
            # Each scorer returns a value in [0.0, 1.0] from image_a's
            # perspective. A scorer that raises is RECORDED but DROPPED:
            # its score is omitted from scores_by_scorer, and
            # Ensemble.aggregate_scores() renormalises the surviving weights
            # to still sum to 1.0. Only if EVERY scorer fails does the
            # ensemble return a neutral 0.5.
            #
            # CLIPScore and VQAScore additionally accept prompt_a / prompt_b
            # so they can measure prompt-fidelity (not just visual quality).
            # All other scorers receive only the two images.
            scores_by_scorer = {}

            for scorer_name, scorer in self.scorers.items():
                try:
                    # CLIPScore and VQAScore both accept optional prompts for
                    # prompt-image alignment / fidelity questions.
                    if scorer_name in ("clip_alignment", "vqa") and image_a_meta and image_b_meta:
                        score = scorer.compare(
                            image_a, image_b,
                            prompt_a=image_a_meta.prompt,
                            prompt_b=image_b_meta.prompt
                        )
                    else:
                        # Aesthetic-only scorers: PickScore, HPSv2, ImageReward, VLMJudge
                        score = scorer.compare(image_a, image_b)

                    scores_by_scorer[scorer_name] = score
                    if is_early_match:
                        logger.info(f"   {scorer_name}: {score:.4f}")
                    else:
                        logger.debug(f"{scorer_name}: {score:.4f}")

                except Exception as e:
                    # Increment per-scorer failure counter (surfaced in
                    # TournamentMetadata.failed_scorers in the JSON output).
                    # Do NOT abort the match — let the ensemble handle it.
                    self.metadata.failed_scorers[scorer_name] = \
                        self.metadata.failed_scorers.get(scorer_name, 0) + 1
                    logger.warning(
                        f"Scorer '{scorer_name}' failed for match "
                        f"({image_a_id}, {image_b_id}): {e}. "
                        f"Excluding it; ensemble will renormalise surviving scorers."
                    )
            
            # Aggregate scores
            ensemble_score = self.ensemble.aggregate_scores(scores_by_scorer)
            
            # Determine winner
            winner_id = image_a_id if ensemble_score > 0.5 else image_b_id
            
            # Create match result
            match_result = MatchResult(
                image_a_id=image_a_id,
                image_b_id=image_b_id,
                scores_by_scorer=scores_by_scorer,
                ensemble_score=ensemble_score,
                winner_id=winner_id,
                timestamp=datetime.now()
            )
            
            return match_result
            
        except Exception as e:
            logger.error(f"Match execution failed: {e}")
            raise RuntimeError(f"Match execution failed: {e}") from e
    
    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Resume tournament from saved checkpoint.
        
        Restores Elo ratings (per-group), completed matches, and pending pairs.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Raises:
            RuntimeError: If checkpoint resume fails
        """
        try:
            logger.info(f"Resuming tournament from checkpoint: {checkpoint_path}")
            
            # Load checkpoint
            state = self.checkpoints.load_checkpoint(checkpoint_path)
            
            # Restore state
            self.completed_matches = state.completed_matches
            self.pending_pairs = state.pending_pairs
            
            # Restore Elo ratings to appropriate per-group systems
            for image_id, rating in state.current_elo_ratings.items():
                group_name = self.image_to_group.get(image_id)
                if group_name and group_name in self.elo_systems:
                    self.elo_systems[group_name].set_rating(image_id, rating)
                else:
                    logger.warning(
                        f"Cannot restore rating for {image_id}: group not found"
                    )
            
            # Update metadata
            self.metadata.start_time = state.timestamp
            
            logger.info(
                f"Resumed from checkpoint: {len(self.completed_matches)} completed, "
                f"{len(self.pending_pairs)} pending"
            )
            
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            raise RuntimeError(f"Checkpoint resume failed: {e}") from e
    
    def _generate_pairs(self) -> List[Tuple[str, str]]:
        """
        Generate image pairs for round-robin tournament, within groups only.

        **Why within-group only?**
        Each group corresponds to one XLSX sheet (or one CSV prompt group),
        meaning all images in a group depict the same subject/scene. Comparing
        images across groups would be meaningless — a cat image vs. a landscape
        image tells us nothing about relative model quality for either prompt.

        **Pair count**
        For a group of n images: C(n, 2) = n*(n-1)/2 pairs.
        Example: 5 sheets × 10 images each → 5 × C(10,2) = 5 × 45 = 225 pairs.
        With all 49 images in one pool it would be C(49,2) = 1176 pairs instead
        — the grouping dramatically reduces compute while keeping comparisons
        semantically meaningful.

        Returns:
            List of (image_a_id, image_b_id) tuples
            Total pairs = Σ C(n_i, 2) across all groups with n_i images
        """
        # Group images by their group field
        groups: Dict[str, List[ImageMetadata]] = {}
        ungrouped: List[ImageMetadata] = []
        
        for img in self.images:
            if img.group:
                if img.group not in groups:
                    groups[img.group] = []
                groups[img.group].append(img)
            else:
                ungrouped.append(img)
        
        pairs: List[Tuple[str, str]] = []
        
        # Generate pairs within each group
        for group_name, group_images in groups.items():
            image_ids = [img.id for img in group_images]
            group_pairs = list(combinations(image_ids, 2))
            pairs.extend(group_pairs)
            logger.info(
                f"Group '{group_name}': {len(group_images)} images → "
                f"{len(group_pairs)} pairs"
            )
        
        # If there are ungrouped images, generate pairs among them
        if ungrouped:
            ungrouped_ids = [img.id for img in ungrouped]
            ungrouped_pairs = list(combinations(ungrouped_ids, 2))
            pairs.extend(ungrouped_pairs)
            logger.warning(
                f"Ungrouped images: {len(ungrouped)} images → "
                f"{len(ungrouped_pairs)} pairs (consider adding group metadata)"
            )
        
        logger.info(
            f"Generated {len(pairs)} total pairs from {len(self.images)} images "
            f"across {len(groups)} groups"
        )
        
        return pairs
    
    def _load_image(self, image_id: str) -> Optional[PILImage]:
        """
        Load image by ID.
        
        Args:
            image_id: Image identifier
        
        Returns:
            PIL Image or None if loading fails
        """
        # Find image metadata
        image_meta = next((img for img in self.images if img.id == image_id), None)
        
        if image_meta is None:
            logger.error(f"Image metadata not found for ID: {image_id}")
            return None
        
        # Load image
        image = self.image_loader.load_image(image_meta.path)
        
        if image is None:
            logger.error(f"Failed to load image: {image_meta.path}")
            self.metadata.skipped_images.append(image_id)
        
        return image

    def _save_checkpoint(self) -> None:
        """
        Save current tournament state to checkpoint.
        """
        try:
            # Aggregate all Elo ratings from all groups
            all_elo_ratings: Dict[str, float] = {}
            for elo_system in self.elo_systems.values():
                all_elo_ratings.update(elo_system.get_all_ratings())
            
            state = CheckpointState(
                iteration=len(self.completed_matches),
                completed_matches=self.completed_matches,
                current_elo_ratings=all_elo_ratings,
                pending_pairs=self.pending_pairs,
                timestamp=datetime.now(),
                config_snapshot=self.config.copy()
            )
            
            checkpoint_path = self.checkpoints.save_checkpoint(
                state=state,
                iteration=len(self.completed_matches)
            )
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Don't raise - checkpoint failure shouldn't stop tournament
    
    def get_current_rankings(self) -> List[Tuple[str, float]]:
        """
        Get current Elo rankings during tournament (all groups combined).
        
        Returns:
            List of (image_id, rating) tuples sorted by rating (descending)
        """
        all_rankings = []
        for elo_system in self.elo_systems.values():
            all_rankings.extend(elo_system.get_rankings())
        
        # Sort by rating descending
        all_rankings.sort(key=lambda x: x[1], reverse=True)
        return all_rankings
    
    def __repr__(self) -> str:
        """String representation of Arena."""
        return (
            f"Arena(images={len(self.images)}, scorers={len(self.scorers)}, "
            f"completed={len(self.completed_matches)}, "
            f"pending={len(self.pending_pairs)})"
        )
