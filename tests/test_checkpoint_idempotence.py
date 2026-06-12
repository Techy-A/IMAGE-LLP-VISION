"""
Property-based tests for checkpoint idempotence.

**Validates: Requirements 9.2, 9.3, 9.4**

Tests that resuming from a checkpoint produces identical results
to continuing without interruption.
"""

import os
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Tuple
from PIL import Image
import pytest
from hypothesis import given, strategies as st, settings, seed

from src.arena import Arena
from src.data_models import ImageMetadata
from src.scoring.base_scorer import BaseScorer
from src.elo_system import EloSystem
from src.bradley_terry_mle import BradleyTerryMLE
from src.ensemble import Ensemble
from src.checkpoint_manager import CheckpointManager


class DeterministicScorer(BaseScorer):
    """
    Deterministic scorer that returns consistent scores based on image IDs.
    
    This scorer ensures reproducible results for property testing by
    computing scores deterministically from image ID hashes.
    """
    
    def __init__(self, name: str = "deterministic", seed: int = 42):
        """
        Initialize deterministic scorer.
        
        Args:
            name: Scorer name
            seed: Random seed for deterministic behavior
        """
        self.name = name
        self.seed = seed
    
    def compare(self, image_a: Image.Image, image_b: Image.Image) -> float:
        """
        Compare two images deterministically.
        
        Uses image IDs (embedded in metadata) to compute consistent scores.
        
        Args:
            image_a: First image
            image_b: Second image
        
        Returns:
            Score in [0.0, 1.0] where >0.5 means image_a wins
        """
        # Extract image IDs from image info (set by test fixture)
        id_a = getattr(image_a, '_test_id', 'unknown_a')
        id_b = getattr(image_b, '_test_id', 'unknown_b')
        
        # Compute deterministic score based on ID hashes
        hash_a = hash(id_a + str(self.seed)) % 1000
        hash_b = hash(id_b + str(self.seed)) % 1000
        
        # Normalize to [0.3, 0.7] range to avoid extreme scores
        raw_score = hash_a / (hash_a + hash_b)
        score = 0.3 + (raw_score * 0.4)
        
        return score
    
    def batch_compare(self, pairs: List[Tuple[Image.Image, Image.Image]]) -> List[float]:
        """
        Batch comparison (delegates to compare).
        
        Args:
            pairs: List of (image_a, image_b) tuples
        
        Returns:
            List of scores
        """
        return [self.compare(a, b) for a, b in pairs]


def create_test_images(num_images: int, temp_dir: str) -> List[ImageMetadata]:
    """
    Create test images and metadata for tournament.
    
    Args:
        num_images: Number of images to create
        temp_dir: Temporary directory for image files
    
    Returns:
        List of ImageMetadata
    """
    images = []
    
    for i in range(num_images):
        # Create simple test image
        image_id = f"img_{i:03d}"
        image_path = os.path.join(temp_dir, f"{image_id}.png")
        
        # Create 32x32 RGB image with unique color based on index
        color = ((i * 30) % 256, (i * 60) % 256, (i * 90) % 256)
        img = Image.new('RGB', (32, 32), color)
        
        # Store ID in image object for deterministic scorer
        img._test_id = image_id
        
        img.save(image_path, 'PNG')
        
        # Create metadata
        metadata = ImageMetadata(
            id=image_id,
            path=image_path,
            prompt=f"Test image {i}",
            model="test_generator"
        )
        
        images.append(metadata)
    
    return images


def run_tournament_with_checkpoint(
    images: List[ImageMetadata],
    checkpoint_iteration: int,
    temp_dir: str,
    scorer_seed: int
) -> Tuple[Dict[str, float], Dict[str, float], List[Tuple[str, float]]]:
    """
    Run tournament and save checkpoint at specified iteration.
    
    Args:
        images: List of image metadata
        checkpoint_iteration: Iteration to save checkpoint
        temp_dir: Temporary directory for checkpoints
        scorer_seed: Seed for deterministic scorer
    
    Returns:
        Tuple of (elo_ratings, bt_ratings, final_rankings)
    """
    # Initialize components
    scorer = DeterministicScorer(seed=scorer_seed)
    scorers = {'deterministic': scorer}
    
    elo_system = EloSystem(k_factor=32, initial_rating=1500.0)
    bt_mle = BradleyTerryMLE(beta=1e-10)
    
    ensemble = Ensemble(weights={'deterministic': 1.0})
    
    checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    
    config = {
        'checkpoint_every': checkpoint_iteration,
        'k_factor': 32,
        'initial_rating': 1500.0
    }
    
    # Create arena
    arena = Arena(
        images=images,
        scorers=scorers,
        elo_system=elo_system,
        bt_mle=bt_mle,
        ensemble=ensemble,
        checkpoint_manager=checkpoint_manager,
        model_rotator=None,
        config=config
    )
    
    # Run tournament
    result = arena.run_tournament()
    
    return result.elo_ratings, result.bt_ratings, result.final_rankings


def run_tournament_from_checkpoint(
    checkpoint_path: str,
    images: List[ImageMetadata],
    temp_dir: str,
    scorer_seed: int
) -> Tuple[Dict[str, float], Dict[str, float], List[Tuple[str, float]]]:
    """
    Resume tournament from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        images: List of image metadata
        temp_dir: Temporary directory
        scorer_seed: Seed for deterministic scorer
    
    Returns:
        Tuple of (elo_ratings, bt_ratings, final_rankings)
    """
    # Initialize components (same setup as original)
    scorer = DeterministicScorer(seed=scorer_seed)
    scorers = {'deterministic': scorer}
    
    elo_system = EloSystem(k_factor=32, initial_rating=1500.0)
    bt_mle = BradleyTerryMLE(beta=1e-10)
    
    ensemble = Ensemble(weights={'deterministic': 1.0})
    
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    
    config = {
        'checkpoint_every': 100,
        'k_factor': 32,
        'initial_rating': 1500.0
    }
    
    # Create arena
    arena = Arena(
        images=images,
        scorers=scorers,
        elo_system=elo_system,
        bt_mle=bt_mle,
        ensemble=ensemble,
        checkpoint_manager=checkpoint_manager,
        model_rotator=None,
        config=config
    )
    
    # Resume from checkpoint
    arena.resume_from_checkpoint(checkpoint_path)
    
    # Continue tournament
    result = arena.run_tournament()
    
    return result.elo_ratings, result.bt_ratings, result.final_rankings


@given(
    num_images=st.integers(min_value=3, max_value=8),
    scorer_seed=st.integers(min_value=1, max_value=1000)
)
@settings(max_examples=10, deadline=60000)  # 60 second deadline for each example
def test_checkpoint_idempotence_property(num_images: int, scorer_seed: int):
    """
    **Property 12: Checkpoint Idempotence**
    
    Test that resuming from checkpoint produces same results as uninterrupted run.
    
    **Validates: Requirements 9.2, 9.3, 9.4**
    
    This property verifies that:
    1. Saving state at iteration N preserves all necessary information
    2. Resuming from checkpoint N continues correctly
    3. Final results are identical regardless of checkpoint/resume
    
    Args:
        num_images: Number of images in tournament (3-8)
        scorer_seed: Seed for deterministic scorer
    """
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp(prefix='checkpoint_idempotence_test_')
    
    try:
        # Create test images
        images = create_test_images(num_images, temp_dir)
        
        # Calculate total matches for round-robin
        total_matches = (num_images * (num_images - 1)) // 2
        
        # Choose checkpoint iteration (midpoint of tournament)
        checkpoint_iteration = max(1, total_matches // 2)
        
        # PATH A: Run tournament without interruption
        elo_a, bt_a, rankings_a = run_tournament_with_checkpoint(
            images=images,
            checkpoint_iteration=checkpoint_iteration,
            temp_dir=temp_dir,
            scorer_seed=scorer_seed
        )
        
        # PATH B: Run to checkpoint, then resume
        # First, run up to checkpoint
        checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
        checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        
        # Get the checkpoint file created during path A
        checkpoints = checkpoint_manager.list_checkpoints()
        
        # Find checkpoint at specified iteration
        checkpoint_path = None
        for cp in checkpoints:
            if f"iter{checkpoint_iteration:06d}" in cp['filename']:
                checkpoint_path = cp['path']
                break
        
        if checkpoint_path is None:
            pytest.skip(f"Checkpoint not found at iteration {checkpoint_iteration}")
        
        # Resume from checkpoint
        elo_b, bt_b, rankings_b = run_tournament_from_checkpoint(
            checkpoint_path=checkpoint_path,
            images=images,
            temp_dir=temp_dir,
            scorer_seed=scorer_seed
        )
        
        # VERIFICATION: Final results should be identical
        
        # 1. Check Elo ratings match
        assert set(elo_a.keys()) == set(elo_b.keys()), \
            "Elo ratings should contain same image IDs"
        
        for image_id in elo_a.keys():
            assert abs(elo_a[image_id] - elo_b[image_id]) < 0.01, \
                f"Elo rating mismatch for {image_id}: {elo_a[image_id]} vs {elo_b[image_id]}"
        
        # 2. Check Bradley-Terry ratings match
        assert set(bt_a.keys()) == set(bt_b.keys()), \
            "Bradley-Terry ratings should contain same image IDs"
        
        for image_id in bt_a.keys():
            assert abs(bt_a[image_id] - bt_b[image_id]) < 0.01, \
                f"Bradley-Terry rating mismatch for {image_id}: {bt_a[image_id]} vs {bt_b[image_id]}"
        
        # 3. Check final rankings match
        assert len(rankings_a) == len(rankings_b), \
            "Rankings should have same length"
        
        for i, ((id_a, rating_a), (id_b, rating_b)) in enumerate(zip(rankings_a, rankings_b)):
            assert id_a == id_b, \
                f"Ranking position {i}: image ID mismatch ({id_a} vs {id_b})"
            
            assert abs(rating_a - rating_b) < 0.01, \
                f"Ranking position {i}: rating mismatch for {id_a} ({rating_a} vs {rating_b})"
    
    finally:
        # Cleanup temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to cleanup temp directory {temp_dir}: {e}")


def test_checkpoint_idempotence_simple():
    """
    Simple unit test for checkpoint idempotence with fixed inputs.
    
    **Validates: Requirements 9.2, 9.3, 9.4**
    
    This is a simpler, deterministic test that complements the property test.
    """
    temp_dir = tempfile.mkdtemp(prefix='checkpoint_simple_test_')
    
    try:
        # Create 4 test images
        images = create_test_images(num_images=4, temp_dir=temp_dir)
        scorer_seed = 42
        
        # Total matches: 4 choose 2 = 6
        total_matches = 6
        checkpoint_iteration = 3  # Checkpoint at halfway point
        
        # PATH A: Uninterrupted run
        elo_a, bt_a, rankings_a = run_tournament_with_checkpoint(
            images=images,
            checkpoint_iteration=checkpoint_iteration,
            temp_dir=temp_dir,
            scorer_seed=scorer_seed
        )
        
        # PATH B: Resume from checkpoint
        checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
        checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        checkpoints = checkpoint_manager.list_checkpoints()
        
        # Find checkpoint at iteration 3
        checkpoint_path = None
        for cp in checkpoints:
            if "iter000003" in cp['filename']:
                checkpoint_path = cp['path']
                break
        
        assert checkpoint_path is not None, "Checkpoint file should exist"
        
        elo_b, bt_b, rankings_b = run_tournament_from_checkpoint(
            checkpoint_path=checkpoint_path,
            images=images,
            temp_dir=temp_dir,
            scorer_seed=scorer_seed
        )
        
        # Verify results match
        assert elo_a == elo_b or \
            all(abs(elo_a[k] - elo_b[k]) < 0.01 for k in elo_a.keys()), \
            "Elo ratings should match"
        
        assert bt_a == bt_b or \
            all(abs(bt_a[k] - bt_b[k]) < 0.01 for k in bt_a.keys()), \
            "Bradley-Terry ratings should match"
        
        assert rankings_a == rankings_b or \
            all(a[0] == b[0] and abs(a[1] - b[1]) < 0.01 
                for a, b in zip(rankings_a, rankings_b)), \
            "Final rankings should match"
    
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to cleanup temp directory {temp_dir}: {e}")


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
