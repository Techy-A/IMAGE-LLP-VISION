"""
Unit tests for data models and type definitions.

Tests validate all data model validation rules including:
- Score bounds [0.0, 1.0]
- ID uniqueness and validity
- Path existence and format validation
- Proper data types and ranges
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import os
from typing import Dict, Any

from src.data_models import (
    ImageMetadata,
    MatchResult,
    TournamentResult,
    TournamentMetadata,
    CheckpointState,
    Config,
    ModelConfig,
    ScoringConfig,
    BradleyTerryConfig,
    TrainingConfig,
    DeviceConfig,
)


# Fixtures for creating temporary test images
@pytest.fixture
def temp_image_file():
    """Create a temporary image file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        # Write minimal PNG header
        f.write(b'\x89PNG\r\n\x1a\n')
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_image_files():
    """Create multiple temporary image files for testing."""
    files = []
    for suffix in ['.png', '.jpg', '.jpeg', '.webp']:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(b'\x89PNG\r\n\x1a\n')  # Minimal PNG header
            files.append(f.name)
    yield files
    # Cleanup
    for path in files:
        if os.path.exists(path):
            os.unlink(path)


# ModelConfig Tests
class TestModelConfig:
    """Tests for ModelConfig validation."""
    
    def test_valid_model_config(self):
        """Test creating valid ModelConfig."""
        config = ModelConfig(
            hf_id="Salesforce/instructblip-vicuna-7b",
            quantization="4bit"
        )
        assert config.hf_id == "Salesforce/instructblip-vicuna-7b"
        assert config.quantization == "4bit"
    
    def test_invalid_quantization(self):
        """Test that invalid quantization raises ValueError."""
        with pytest.raises(ValueError, match="Invalid quantization"):
            ModelConfig(
                hf_id="Salesforce/instructblip-vicuna-7b",
                quantization="2bit"  # Invalid
            )
    
    def test_invalid_hf_id_format(self):
        """Test that invalid HuggingFace ID format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid HuggingFace model ID"):
            ModelConfig(
                hf_id="invalid-model-id",  # Missing organization/
                quantization="4bit"
            )
    
    def test_empty_hf_id(self):
        """Test that empty HuggingFace ID raises ValueError."""
        with pytest.raises(ValueError, match="Invalid HuggingFace model ID"):
            ModelConfig(hf_id="", quantization="4bit")
    
    def test_all_quantization_options(self):
        """Test all valid quantization options."""
        for quant in ["4bit", "8bit", "fp16", "fp32"]:
            config = ModelConfig(
                hf_id="openai/clip-vit-large-patch14",
                quantization=quant
            )
            assert config.quantization == quant


# BradleyTerryConfig Tests
class TestBradleyTerryConfig:
    """Tests for BradleyTerryConfig validation."""
    
    def test_valid_bradley_terry_config(self):
        """Test creating valid BradleyTerryConfig."""
        config = BradleyTerryConfig(beta=1e-10)
        assert config.beta == 1e-10
    
    def test_negative_beta(self):
        """Test that negative beta raises ValueError."""
        with pytest.raises(ValueError, match="Beta regularization must be > 0"):
            BradleyTerryConfig(beta=-0.1)
    
    def test_zero_beta(self):
        """Test that zero beta raises ValueError."""
        with pytest.raises(ValueError, match="Beta regularization must be > 0"):
            BradleyTerryConfig(beta=0.0)


# ScoringConfig Tests
class TestScoringConfig:
    """Tests for ScoringConfig validation."""
    
    def test_valid_scoring_config(self):
        """Test creating valid ScoringConfig."""
        config = ScoringConfig(
            weights={"vqa": 0.3, "pickscore": 0.2, "image_reward": 0.2, "vlm_judge": 0.3},
            bt_mle=BradleyTerryConfig(beta=1e-10)
        )
        assert config.weights["vqa"] == 0.3
        assert config.bt_mle.beta == 1e-10
    
    def test_weights_sum_not_one(self):
        """Test that weights not summing to 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="Scorer weights must sum to 1.0"):
            ScoringConfig(
                weights={"vqa": 0.3, "pickscore": 0.3},  # Sum = 0.6
                bt_mle=BradleyTerryConfig(beta=1e-10)
            )
    
    def test_negative_weight(self):
        """Test that negative weight raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            ScoringConfig(
                weights={"vqa": 1.2, "pickscore": -0.2},  # Sum = 1.0 but negative
                bt_mle=BradleyTerryConfig(beta=1e-10)
            )
    
    def test_empty_weights(self):
        """Test that empty weights raises ValueError."""
        with pytest.raises(ValueError, match="Scoring weights cannot be empty"):
            ScoringConfig(
                weights={},
                bt_mle=BradleyTerryConfig(beta=1e-10)
            )
    
    def test_weights_sum_with_tolerance(self):
        """Test that weights sum close to 1.0 within tolerance is accepted."""
        # 1.0000001 should be within tolerance
        config = ScoringConfig(
            weights={"vqa": 0.25, "pickscore": 0.25, "image_reward": 0.25, "vlm_judge": 0.25000001},
            bt_mle=BradleyTerryConfig(beta=1e-10)
        )
        assert sum(config.weights.values()) > 0.999


# TrainingConfig Tests
class TestTrainingConfig:
    """Tests for TrainingConfig validation."""
    
    def test_valid_training_config(self):
        """Test creating valid TrainingConfig."""
        config = TrainingConfig(
            checkpoint_every=100,
            batch_size=8,
            learning_rate=1e-4
        )
        assert config.checkpoint_every == 100
        assert config.batch_size == 8
        assert config.learning_rate == 1e-4
    
    def test_negative_checkpoint_every(self):
        """Test that negative checkpoint_every raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_every must be > 0"):
            TrainingConfig(checkpoint_every=-10, batch_size=8, learning_rate=1e-4)
    
    def test_zero_batch_size(self):
        """Test that zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            TrainingConfig(checkpoint_every=100, batch_size=0, learning_rate=1e-4)
    
    def test_negative_learning_rate(self):
        """Test that negative learning_rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            TrainingConfig(checkpoint_every=100, batch_size=8, learning_rate=-0.001)


# DeviceConfig Tests
class TestDeviceConfig:
    """Tests for DeviceConfig validation."""
    
    def test_valid_device_config(self):
        """Test creating valid DeviceConfig."""
        config = DeviceConfig(auto_detect=True, fallback="cpu")
        assert config.auto_detect is True
        assert config.fallback == "cpu"
    
    def test_invalid_fallback_device(self):
        """Test that invalid fallback device raises ValueError."""
        with pytest.raises(ValueError, match="Invalid fallback device"):
            DeviceConfig(auto_detect=True, fallback="gpu")
    
    def test_all_valid_devices(self):
        """Test all valid fallback devices."""
        for device in ["cpu", "cuda", "mps"]:
            config = DeviceConfig(auto_detect=False, fallback=device)
            assert config.fallback == device


# Config Tests
class TestConfig:
    """Tests for main Config validation."""
    
    def test_valid_config(self):
        """Test creating valid Config."""
        config = Config(
            models={"clip": ModelConfig(hf_id="openai/clip-vit-large-patch14", quantization="fp16")},
            scoring=ScoringConfig(
                weights={"vqa": 1.0},
                bt_mle=BradleyTerryConfig(beta=1e-10)
            ),
            training=TrainingConfig(checkpoint_every=100, batch_size=8, learning_rate=1e-4),
            device=DeviceConfig(auto_detect=True, fallback="cpu")
        )
        assert "clip" in config.models
    
    def test_empty_models(self):
        """Test that empty models raises ValueError."""
        with pytest.raises(ValueError, match="At least one model must be configured"):
            Config(
                models={},
                scoring=ScoringConfig(
                    weights={"vqa": 1.0},
                    bt_mle=BradleyTerryConfig(beta=1e-10)
                ),
                training=TrainingConfig(checkpoint_every=100, batch_size=8, learning_rate=1e-4),
                device=DeviceConfig(auto_detect=True, fallback="cpu")
            )


# ImageMetadata Tests
class TestImageMetadata:
    """Tests for ImageMetadata validation."""
    
    def test_valid_image_metadata(self, temp_image_file):
        """Test creating valid ImageMetadata."""
        metadata = ImageMetadata(
            id="img_001",
            path=temp_image_file,
            prompt="A test image",
            model="dalle-3",
            attributes={"style": "realistic"}
        )
        assert metadata.id == "img_001"
        assert metadata.path == temp_image_file
        assert metadata.prompt == "A test image"
        assert metadata.model == "dalle-3"
        assert metadata.attributes["style"] == "realistic"
    
    def test_empty_id(self, temp_image_file):
        """Test that empty ID raises ValueError."""
        with pytest.raises(ValueError, match="Image ID cannot be empty"):
            ImageMetadata(id="", path=temp_image_file)
    
    def test_empty_path(self):
        """Test that empty path raises ValueError."""
        with pytest.raises(ValueError, match="Image path cannot be empty"):
            ImageMetadata(id="img_001", path="")
    
    def test_nonexistent_path(self):
        """Test that nonexistent path raises ValueError."""
        with pytest.raises(ValueError, match="Image path does not exist"):
            ImageMetadata(id="img_001", path="/nonexistent/path/image.png")
    
    def test_path_is_directory(self, temp_image_file):
        """Test that directory path raises ValueError."""
        temp_dir = os.path.dirname(temp_image_file)
        with pytest.raises(ValueError, match="Image path is not a file"):
            ImageMetadata(id="img_001", path=temp_dir)
    
    def test_unsupported_format(self):
        """Test that unsupported format raises ValueError."""
        # Create temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported image format"):
                ImageMetadata(id="img_001", path=temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_all_supported_formats(self, temp_image_files):
        """Test all supported image formats."""
        for i, path in enumerate(temp_image_files):
            metadata = ImageMetadata(id=f"img_{i:03d}", path=path)
            assert metadata.path == path
    
    def test_optional_fields(self, temp_image_file):
        """Test ImageMetadata with only required fields."""
        metadata = ImageMetadata(id="img_001", path=temp_image_file)
        assert metadata.prompt is None
        assert metadata.model is None
        assert metadata.attributes == {}


# MatchResult Tests
class TestMatchResult:
    """Tests for MatchResult validation."""
    
    def test_valid_match_result(self):
        """Test creating valid MatchResult."""
        result = MatchResult(
            image_a_id="img_001",
            image_b_id="img_002",
            scores_by_scorer={"vqa": 0.6, "pickscore": 0.7},
            ensemble_score=0.65,
            winner_id="img_001",
            timestamp=datetime.now()
        )
        assert result.winner_id == "img_001"
        assert result.ensemble_score == 0.65
    
    def test_score_below_range(self):
        """Test that score below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
            MatchResult(
                image_a_id="img_001",
                image_b_id="img_002",
                scores_by_scorer={"vqa": -0.1},
                ensemble_score=0.5,
                winner_id="img_001",
                timestamp=datetime.now()
            )
    
    def test_score_above_range(self):
        """Test that score above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
            MatchResult(
                image_a_id="img_001",
                image_b_id="img_002",
                scores_by_scorer={"vqa": 1.5},
                ensemble_score=0.5,
                winner_id="img_001",
                timestamp=datetime.now()
            )
    
    def test_ensemble_score_out_of_range(self):
        """Test that ensemble score out of range raises ValueError."""
        with pytest.raises(ValueError, match="Ensemble score must be in \\[0.0, 1.0\\]"):
            MatchResult(
                image_a_id="img_001",
                image_b_id="img_002",
                scores_by_scorer={"vqa": 0.6},
                ensemble_score=1.1,
                winner_id="img_001",
                timestamp=datetime.now()
            )
    
    def test_invalid_winner_id(self):
        """Test that invalid winner_id raises ValueError."""
        with pytest.raises(ValueError, match="winner_id .* must be either image_a_id"):
            MatchResult(
                image_a_id="img_001",
                image_b_id="img_002",
                scores_by_scorer={"vqa": 0.6},
                ensemble_score=0.6,
                winner_id="img_003",  # Invalid
                timestamp=datetime.now()
            )
    
    def test_same_image_ids(self):
        """Test that same image IDs raises ValueError."""
        with pytest.raises(ValueError, match="image_a_id and image_b_id must be different"):
            MatchResult(
                image_a_id="img_001",
                image_b_id="img_001",  # Same as image_a_id
                scores_by_scorer={"vqa": 0.6},
                ensemble_score=0.6,
                winner_id="img_001",
                timestamp=datetime.now()
            )
    
    def test_boundary_scores(self):
        """Test boundary scores 0.0 and 1.0 are accepted."""
        result = MatchResult(
            image_a_id="img_001",
            image_b_id="img_002",
            scores_by_scorer={"vqa": 0.0, "pickscore": 1.0},
            ensemble_score=0.5,
            winner_id="img_002",
            timestamp=datetime.now()
        )
        assert result.scores_by_scorer["vqa"] == 0.0
        assert result.scores_by_scorer["pickscore"] == 1.0


# TournamentMetadata Tests
class TestTournamentMetadata:
    """Tests for TournamentMetadata validation."""
    
    def test_valid_tournament_metadata(self):
        """Test creating valid TournamentMetadata."""
        metadata = TournamentMetadata(
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_images=10,
            total_matches=45,
            config_snapshot={"k_factor": 32},
            skipped_images=["img_bad"],
            failed_scorers={"vqa": 2}
        )
        assert metadata.total_images == 10
        assert metadata.total_matches == 45
    
    def test_negative_total_images(self):
        """Test that negative total_images raises ValueError."""
        with pytest.raises(ValueError, match="total_images must be non-negative"):
            TournamentMetadata(
                start_time=datetime.now(),
                total_images=-5
            )
    
    def test_negative_total_matches(self):
        """Test that negative total_matches raises ValueError."""
        with pytest.raises(ValueError, match="total_matches must be non-negative"):
            TournamentMetadata(
                start_time=datetime.now(),
                total_matches=-10
            )
    
    def test_end_time_before_start_time(self):
        """Test that end_time before start_time raises ValueError."""
        start = datetime(2024, 1, 2, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 0, 0)
        with pytest.raises(ValueError, match="end_time .* cannot be before start_time"):
            TournamentMetadata(
                start_time=start,
                end_time=end
            )
    
    def test_optional_fields(self):
        """Test TournamentMetadata with only required fields."""
        metadata = TournamentMetadata(start_time=datetime.now())
        assert metadata.end_time is None
        assert metadata.total_images == 0
        assert metadata.total_matches == 0
        assert metadata.config_snapshot == {}
        assert metadata.skipped_images == []
        assert metadata.failed_scorers == {}


# TournamentResult Tests
class TestTournamentResult:
    """Tests for TournamentResult validation."""
    
    def test_valid_tournament_result(self):
        """Test creating valid TournamentResult."""
        matches = [
            MatchResult(
                image_a_id="img_001",
                image_b_id="img_002",
                scores_by_scorer={"vqa": 0.6},
                ensemble_score=0.6,
                winner_id="img_001",
                timestamp=datetime.now()
            )
        ]
        result = TournamentResult(
            matches=matches,
            elo_ratings={"img_001": 1516, "img_002": 1484},
            bt_ratings={"img_001": 0.6, "img_002": 0.4},
            final_rankings=[("img_001", 1516), ("img_002", 1484)],
            metadata=TournamentMetadata(start_time=datetime.now())
        )
        assert len(result.matches) == 1
        assert result.elo_ratings["img_001"] == 1516
    
    def test_invalid_matches_type(self):
        """Test that non-list matches raises ValueError."""
        with pytest.raises(ValueError, match="matches must be a list"):
            TournamentResult(
                matches="not a list",
                elo_ratings={"img_001": 1500},
                bt_ratings={"img_001": 0.5},
                final_rankings=[("img_001", 1500)],
                metadata=TournamentMetadata(start_time=datetime.now())
            )
    
    def test_invalid_elo_ratings_type(self):
        """Test that non-dict elo_ratings raises ValueError."""
        with pytest.raises(ValueError, match="elo_ratings must be a dictionary"):
            TournamentResult(
                matches=[],
                elo_ratings="not a dict",
                bt_ratings={"img_001": 0.5},
                final_rankings=[],
                metadata=TournamentMetadata(start_time=datetime.now())
            )
    
    def test_non_numeric_elo_rating(self):
        """Test that non-numeric Elo rating raises ValueError."""
        with pytest.raises(ValueError, match="Elo rating .* must be numeric"):
            TournamentResult(
                matches=[],
                elo_ratings={"img_001": "high"},
                bt_ratings={"img_001": 0.5},
                final_rankings=[],
                metadata=TournamentMetadata(start_time=datetime.now())
            )
    
    def test_non_numeric_bt_rating(self):
        """Test that non-numeric Bradley-Terry rating raises ValueError."""
        with pytest.raises(ValueError, match="Bradley-Terry rating .* must be numeric"):
            TournamentResult(
                matches=[],
                elo_ratings={"img_001": 1500},
                bt_ratings={"img_001": "high"},
                final_rankings=[],
                metadata=TournamentMetadata(start_time=datetime.now())
            )
    
    def test_rankings_not_sorted_descending(self):
        """Test that rankings not sorted descending raises ValueError."""
        with pytest.raises(ValueError, match="final_rankings must be sorted descending"):
            TournamentResult(
                matches=[],
                elo_ratings={"img_001": 1484, "img_002": 1516},
                bt_ratings={"img_001": 0.4, "img_002": 0.6},
                final_rankings=[("img_001", 1484), ("img_002", 1516)],  # Wrong order
                metadata=TournamentMetadata(start_time=datetime.now())
            )
    
    def test_rankings_sorted_descending_with_equal_values(self):
        """Test that equal ranking values are accepted."""
        result = TournamentResult(
            matches=[],
            elo_ratings={"img_001": 1500, "img_002": 1500},
            bt_ratings={"img_001": 0.5, "img_002": 0.5},
            final_rankings=[("img_001", 1500), ("img_002", 1500)],  # Equal
            metadata=TournamentMetadata(start_time=datetime.now())
        )
        assert len(result.final_rankings) == 2


# CheckpointState Tests
class TestCheckpointState:
    """Tests for CheckpointState validation."""
    
    def test_valid_checkpoint_state(self):
        """Test creating valid CheckpointState."""
        matches = [
            MatchResult(
                image_a_id="img_001",
                image_b_id="img_002",
                scores_by_scorer={"vqa": 0.6},
                ensemble_score=0.6,
                winner_id="img_001",
                timestamp=datetime.now()
            )
        ]
        state = CheckpointState(
            iteration=1,
            completed_matches=matches,
            current_elo_ratings={"img_001": 1516, "img_002": 1484},
            pending_pairs=[("img_001", "img_003"), ("img_002", "img_003")],
            timestamp=datetime.now(),
            config_snapshot={"k_factor": 32}
        )
        assert state.iteration == 1
        assert len(state.completed_matches) == 1
        assert len(state.pending_pairs) == 2
    
    def test_negative_iteration(self):
        """Test that negative iteration raises ValueError."""
        with pytest.raises(ValueError, match="iteration must be non-negative"):
            CheckpointState(
                iteration=-1,
                completed_matches=[],
                current_elo_ratings={},
                pending_pairs=[],
                timestamp=datetime.now(),
                config_snapshot={}
            )
    
    def test_mismatched_iteration_and_matches_length(self):
        """Test that mismatched iteration and matches length raises ValueError."""
        with pytest.raises(ValueError, match="completed_matches length .* must equal iteration"):
            CheckpointState(
                iteration=5,
                completed_matches=[],  # Length 0, not 5
                current_elo_ratings={},
                pending_pairs=[],
                timestamp=datetime.now(),
                config_snapshot={}
            )
    
    def test_invalid_elo_ratings_type(self):
        """Test that non-dict current_elo_ratings raises ValueError."""
        with pytest.raises(ValueError, match="current_elo_ratings must be a dictionary"):
            CheckpointState(
                iteration=0,
                completed_matches=[],
                current_elo_ratings="not a dict",
                pending_pairs=[],
                timestamp=datetime.now(),
                config_snapshot={}
            )
    
    def test_non_numeric_elo_rating(self):
        """Test that non-numeric Elo rating raises ValueError."""
        with pytest.raises(ValueError, match="Elo rating .* must be numeric"):
            CheckpointState(
                iteration=0,
                completed_matches=[],
                current_elo_ratings={"img_001": "high"},
                pending_pairs=[],
                timestamp=datetime.now(),
                config_snapshot={}
            )
    
    def test_invalid_pending_pairs_type(self):
        """Test that non-list pending_pairs raises ValueError."""
        with pytest.raises(ValueError, match="pending_pairs must be a list"):
            CheckpointState(
                iteration=0,
                completed_matches=[],
                current_elo_ratings={},
                pending_pairs="not a list",
                timestamp=datetime.now(),
                config_snapshot={}
            )
    
    def test_invalid_pair_format(self):
        """Test that invalid pair format raises ValueError."""
        with pytest.raises(ValueError, match="Each pending pair must be a tuple of length 2"):
            CheckpointState(
                iteration=0,
                completed_matches=[],
                current_elo_ratings={},
                pending_pairs=[("img_001",)],  # Length 1, not 2
                timestamp=datetime.now(),
                config_snapshot={}
            )
    
    def test_valid_pending_pairs_format(self):
        """Test valid pending pairs format."""
        state = CheckpointState(
            iteration=0,
            completed_matches=[],
            current_elo_ratings={},
            pending_pairs=[("img_001", "img_002"), ("img_001", "img_003")],
            timestamp=datetime.now(),
            config_snapshot={}
        )
        assert len(state.pending_pairs) == 2
        assert state.pending_pairs[0] == ("img_001", "img_002")
