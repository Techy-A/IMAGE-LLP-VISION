"""
PickScore metric for aesthetic image scoring.

PickScore uses a CLIP-based model fine-tuned on human preference data
to score image aesthetic quality. This implementation uses AutoProcessor
(NOT CLIPProcessor) as specified in the design.
"""

import logging
import math
from typing import List, Tuple

import torch
from transformers import AutoProcessor, AutoModel
from PIL.Image import Image as PILImage

from src.scoring.base_scorer import BaseScorer


logger = logging.getLogger(__name__)


class PickScore(BaseScorer):
    """
    PickScore aesthetic quality scorer.
    
    PickScore is a CLIP-based model trained on human preference data for
    image quality assessment. It provides scores that correlate with human
    aesthetic preferences.
    
    Paper: "Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation"
    Model: yuvalkirstain/PickScore_v1
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize PickScore scorer.
        
        Args:
            device: Device to run inference on ("auto", "cuda", "mps", "cpu")
        """
        self.model = None
        self.processor = None
        self.device = self._detect_device() if device == "auto" else device
        logger.info(f"PickScore initialized with device: {self.device}")
    
    def load(self, model_id: str = "yuvalkirstain/PickScore_v1") -> None:
        """
        Load PickScore model and processor.
        
        Args:
            model_id: HuggingFace model identifier
        
        Raises:
            RuntimeError: If loading fails
        """
        try:
            logger.info(f"Loading PickScore model: {model_id}")

            # PickScore's official usage specifies AutoProcessor (not CLIPProcessor)
            # even though PickScore is built on top of CLIP. Using CLIPProcessor
            # directly skips PickScore-specific pre-processing steps and produces
            # subtly different (wrong) embeddings.
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to(self.device)
            self.model.eval()

            logger.info("PickScore model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load PickScore model: {e}")
            raise RuntimeError(f"PickScore loading failed: {e}") from e
    
    def compare(self, image_a: PILImage, image_b: PILImage) -> float:
        """
        Compare two images using PickScore aesthetic scoring.
        
        Args:
            image_a: First image to compare
            image_b: Second image to compare
        
        Returns:
            Score in [0.0, 1.0] where >0.5 means image_a has better aesthetics
        
        Raises:
            RuntimeError: If comparison fails
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("PickScore model not loaded. Call load() first.")
        
        try:
            # Compute scores for both images
            score_a = self._score_image(image_a)
            score_b = self._score_image(image_b)
            
            # Compute difference
            score_diff = score_a - score_b
            
            # Apply sigmoid normalization to get value in [0.0, 1.0]
            normalized_score = self._sigmoid(score_diff)
            
            return float(normalized_score)
            
        except Exception as e:
            logger.error(f"PickScore comparison failed: {e}")
            raise RuntimeError(f"PickScore comparison failed: {e}") from e
    
    def batch_compare(
        self,
        pairs: List[Tuple[PILImage, PILImage]]
    ) -> List[float]:
        """
        Compare multiple image pairs using PickScore.
        
        Args:
            pairs: List of (image_a, image_b) tuples
        
        Returns:
            List of scores in [0.0, 1.0] for each pair
        
        Raises:
            RuntimeError: If batch comparison fails
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("PickScore model not loaded. Call load() first.")
        
        try:
            results = []
            
            # Process each pair
            # TODO: Could optimize with true batch processing
            for image_a, image_b in pairs:
                score = self.compare(image_a, image_b)
                results.append(score)
            
            return results
            
        except Exception as e:
            logger.error(f"PickScore batch comparison failed: {e}")
            raise RuntimeError(f"PickScore batch comparison failed: {e}") from e
    
    def _score_image(self, image: PILImage) -> float:
        """
        Compute PickScore for a single image.
        
        Args:
            image: Image to score
        
        Returns:
            Aesthetic quality score (higher is better)
        """
        # Process image — no text tokens needed for quality scoring
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            # PickScore uses the L2 norm (magnitude) of the image embedding
            # as its quality signal. A higher-norm embedding means the image
            # is more "confident" in feature space — correlating with higher
            # human preference scores as measured by the Pick-a-Pic dataset.
            image_features = self.model.get_image_features(**inputs)
            score = image_features.norm(dim=-1)

        return score.item()
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        """
        Apply sigmoid function to normalize score difference.
        
        Args:
            x: Score difference
        
        Returns:
            Normalized value in [0.0, 1.0]
        """
        return 1.0 / (1.0 + math.exp(-x))
    
    @staticmethod
    def _detect_device() -> str:
        """
        Auto-detect best available device.
        
        Returns:
            Device string ("cuda", "mps", or "cpu")
        """
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def unload(self) -> None:
        """
        Unload PickScore model from memory.
        """
        if self.model is not None:
            logger.info("Unloading PickScore model")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            
            # Clear GPU cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()
            
            logger.info("PickScore model unloaded")
