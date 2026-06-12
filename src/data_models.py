"""
Data models and type definitions for IMAGE-LLP-VISION system.

This module defines all core data structures used throughout the pairwise
image evaluation system, including metadata, results, configurations, and
checkpoint state.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math


@dataclass
class ModelConfig:
    """Configuration for a single vision model."""
    hf_id: str  # HuggingFace model identifier
    quantization: str  # "4bit", "8bit", "fp16", "fp32"
    
    def __post_init__(self):
        """Validate model configuration."""
        valid_quantizations = ["4bit", "8bit", "fp16", "fp32"]
        if self.quantization not in valid_quantizations:
            raise ValueError(
                f"Invalid quantization '{self.quantization}'. "
                f"Must be one of: {valid_quantizations}"
            )
        
        # Basic HuggingFace ID format validation
        if not self.hf_id or "/" not in self.hf_id:
            raise ValueError(
                f"Invalid HuggingFace model ID '{self.hf_id}'. "
                f"Expected format: 'organization/model-name'"
            )


@dataclass
class BradleyTerryConfig:
    """Configuration for Bradley-Terry MLE rating system."""
    beta: float  # Regularization for images with 0 wins
    
    def __post_init__(self):
        """Validate Bradley-Terry configuration."""
        if self.beta <= 0:
            raise ValueError(
                f"Beta regularization must be > 0, got {self.beta}"
            )


@dataclass
class ScoringConfig:
    """Configuration for scoring system."""
    weights: Dict[str, float]  # Scorer name -> weight
    bt_mle: BradleyTerryConfig
    
    def __post_init__(self):
        """Validate scoring configuration."""
        if not self.weights:
            raise ValueError("Scoring weights cannot be empty")
        
        # Validate all weights sum to 1.0 (with floating point tolerance)
        weight_sum = sum(self.weights.values())
        if not math.isclose(weight_sum, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Scorer weights must sum to 1.0, got {weight_sum}. "
                f"Weights: {self.weights}"
            )
        
        # Validate all weights are non-negative
        for scorer_name, weight in self.weights.items():
            if weight < 0:
                raise ValueError(
                    f"Weight for scorer '{scorer_name}' must be non-negative, "
                    f"got {weight}"
                )


@dataclass
class TrainingConfig:
    """Configuration for training/tournament execution."""
    checkpoint_every: int  # Save checkpoint every N matches
    batch_size: int
    learning_rate: float
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.checkpoint_every <= 0:
            raise ValueError(
                f"checkpoint_every must be > 0, got {self.checkpoint_every}"
            )
        
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be > 0, got {self.batch_size}"
            )
        
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be > 0, got {self.learning_rate}"
            )


@dataclass
class DeviceConfig:
    """Configuration for compute device selection."""
    auto_detect: bool  # Auto-detect best available device
    fallback: str  # Fallback device: "cpu", "cuda", "mps"
    
    def __post_init__(self):
        """Validate device configuration."""
        valid_devices = ["cpu", "cuda", "mps"]
        if self.fallback not in valid_devices:
            raise ValueError(
                f"Invalid fallback device '{self.fallback}'. "
                f"Must be one of: {valid_devices}"
            )


@dataclass
class Config:
    """Main system configuration."""
    models: Dict[str, ModelConfig]
    scoring: ScoringConfig
    training: TrainingConfig
    device: DeviceConfig
    
    def __post_init__(self):
        """Validate main configuration."""
        if not self.models:
            raise ValueError("At least one model must be configured")


@dataclass
class ImageMetadata:
    """
    Metadata for a single image participating in the evaluation tournament.

    Populated by CSVLoader or XLSXImageExtractor and used throughout
    the pipeline. The `group` field is the most important: it controls
    which other images this image will be compared against (pairs are
    only generated within the same group).
    """
    id: str                            # Unique identifier, e.g. "mermaid_metric_row3"
    path: str                          # Absolute or relative path to the image file
    prompt: Optional[str] = None       # Text prompt used to generate the image;
                                       # forwarded to CLIPScore and VQAScore so
                                       # those scorers can evaluate prompt fidelity
    model: Optional[str] = None        # Name of the generative model (from Column B
                                       # of the XLSX), used in CSV output for analysis
    group: Optional[str] = None        # Group name (= XLSX sheet name, or CSV prompt);
                                       # images in the same group share a scene/prompt
                                       # and will be compared against each other
    attributes: Dict[str, Any] = field(default_factory=dict)  # Extra metadata
                                       # (source_sheet, source_cell, extracted, etc.)

    def __post_init__(self):
        """Validate image metadata."""
        if not self.id:
            raise ValueError("Image ID cannot be empty")

        if not self.path:
            raise ValueError("Image path cannot be empty")

        # Validate path points to existing file
        path_obj = Path(self.path)
        if not path_obj.exists():
            raise ValueError(
                f"Image path does not exist: {self.path}"
            )

        if not path_obj.is_file():
            raise ValueError(
                f"Image path is not a file: {self.path}"
            )

        # Validate file format
        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        if path_obj.suffix.lower() not in valid_extensions:
            raise ValueError(
                f"Unsupported image format '{path_obj.suffix}'. "
                f"Supported formats: {valid_extensions}"
            )


@dataclass
class MatchResult:
    """
    Result from a single pairwise image comparison.

    All score fields use image_a's perspective: a value > 0.5 means image_a
    performed better. ResultsExporter inverts these when computing image_b's
    per-match contribution to that image's aggregate statistics.
    """
    image_a_id: str                          # ID of the first image
    image_b_id: str                          # ID of the second image
    scores_by_scorer: Dict[str, float]       # Per-scorer raw scores, image_a perspective
    ensemble_score: float                    # Weighted aggregate of all scorer scores
    winner_id: str                           # image_a_id if ensemble_score > 0.5, else image_b_id
    timestamp: datetime                      # When this match was processed
    
    def __post_init__(self):
        """Validate match result."""
        # Validate all scorer scores are in [0.0, 1.0]
        for scorer_name, score in self.scores_by_scorer.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(
                    f"Score for scorer '{scorer_name}' must be in [0.0, 1.0], "
                    f"got {score}"
                )
        
        # Validate ensemble score is in [0.0, 1.0]
        if not (0.0 <= self.ensemble_score <= 1.0):
            raise ValueError(
                f"Ensemble score must be in [0.0, 1.0], got {self.ensemble_score}"
            )
        
        # Validate winner_id matches one of the images
        if self.winner_id not in {self.image_a_id, self.image_b_id}:
            raise ValueError(
                f"winner_id '{self.winner_id}' must be either image_a_id "
                f"'{self.image_a_id}' or image_b_id '{self.image_b_id}'"
            )
        
        # Validate image IDs are different
        if self.image_a_id == self.image_b_id:
            raise ValueError(
                f"image_a_id and image_b_id must be different, "
                f"both are '{self.image_a_id}'"
            )


@dataclass
class TournamentMetadata:
    """
    Metadata about tournament execution.

    Populated by Arena as the tournament runs and attached to TournamentResult.
    Surfaced in the JSON output under the 'metadata' key so runs can be
    audited and compared.
    """
    start_time: datetime
    end_time: Optional[datetime] = None
    total_images: int = 0
    total_matches: int = 0
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    skipped_images: List[str] = field(default_factory=list)   # Images that failed to load
    failed_scorers: Dict[str, int] = field(default_factory=dict)  # Scorer → #failures
    
    def __post_init__(self):
        """Validate tournament metadata."""
        if self.total_images < 0:
            raise ValueError(
                f"total_images must be non-negative, got {self.total_images}"
            )
        
        if self.total_matches < 0:
            raise ValueError(
                f"total_matches must be non-negative, got {self.total_matches}"
            )
        
        # If both end_time and start_time are set, validate order
        if self.end_time is not None and self.end_time < self.start_time:
            raise ValueError(
                f"end_time {self.end_time} cannot be before start_time {self.start_time}"
            )


@dataclass
class TournamentResult:
    """
    Complete results from a tournament run.

    This is the root object returned by Arena.run_tournament() and
    serialised to disk by ResultsExporter. The `final_rankings` list is
    sorted by Bradley-Terry skill (descending) and drives both the CSV
    rank column and the visualize command's bar-chart output.
    """
    matches: List[MatchResult]                          # All completed pairwise comparisons
    elo_ratings: Dict[str, float]                       # Final Elo rating per image (per-group)
    bt_ratings: Dict[str, float]                        # Final BT-MLE skill per image (per-group)
    final_rankings: List[Tuple[str, float]]             # (image_id, bt_skill) sorted descending
    metadata: TournamentMetadata                        # Timing and diagnostics
    
    def __post_init__(self):
        """Validate tournament result."""
        # Validate matches list
        if not isinstance(self.matches, list):
            raise ValueError("matches must be a list")
        
        # Validate ratings are dictionaries with float values
        if not isinstance(self.elo_ratings, dict):
            raise ValueError("elo_ratings must be a dictionary")
        
        if not isinstance(self.bt_ratings, dict):
            raise ValueError("bt_ratings must be a dictionary")
        
        # Validate all rating values are floats
        for image_id, rating in self.elo_ratings.items():
            if not isinstance(rating, (int, float)):
                raise ValueError(
                    f"Elo rating for '{image_id}' must be numeric, "
                    f"got {type(rating)}"
                )
        
        for image_id, rating in self.bt_ratings.items():
            if not isinstance(rating, (int, float)):
                raise ValueError(
                    f"Bradley-Terry rating for '{image_id}' must be numeric, "
                    f"got {type(rating)}"
                )
        
        # Validate final_rankings is sorted descending
        if len(self.final_rankings) > 1:
            for i in range(len(self.final_rankings) - 1):
                if self.final_rankings[i][1] < self.final_rankings[i + 1][1]:
                    raise ValueError(
                        f"final_rankings must be sorted descending by rating. "
                        f"Found {self.final_rankings[i]} before {self.final_rankings[i + 1]}"
                    )


@dataclass
class CheckpointState:
    """Saved tournament state for resume capability."""
    iteration: int
    completed_matches: List[MatchResult]
    current_elo_ratings: Dict[str, float]
    pending_pairs: List[Tuple[str, str]]
    timestamp: datetime
    config_snapshot: Dict[str, Any]
    
    def __post_init__(self):
        """Validate checkpoint state."""
        # Validate iteration is non-negative
        if self.iteration < 0:
            raise ValueError(
                f"iteration must be non-negative, got {self.iteration}"
            )
        
        # Validate completed_matches length matches iteration
        if len(self.completed_matches) != self.iteration:
            raise ValueError(
                f"completed_matches length ({len(self.completed_matches)}) "
                f"must equal iteration ({self.iteration})"
            )
        
        # Validate ratings are dictionaries with float values
        if not isinstance(self.current_elo_ratings, dict):
            raise ValueError("current_elo_ratings must be a dictionary")
        
        for image_id, rating in self.current_elo_ratings.items():
            if not isinstance(rating, (int, float)):
                raise ValueError(
                    f"Elo rating for '{image_id}' must be numeric, "
                    f"got {type(rating)}"
                )
        
        # Validate pending_pairs format
        if not isinstance(self.pending_pairs, list):
            raise ValueError("pending_pairs must be a list")
        
        for pair in self.pending_pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(
                    f"Each pending pair must be a tuple of length 2, got {pair}"
                )
