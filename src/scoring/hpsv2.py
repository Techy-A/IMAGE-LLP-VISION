"""
HPSv2 (Human Preference Score v2) metric for aesthetic image scoring.

HPSv2 is a model trained on human preference data to predict aesthetic
quality and alignment with human preferences.
"""

import logging
import math
from typing import List, Tuple

import torch
from transformers import AutoProcessor, AutoModel
from PIL.Image import Image as PILImage

from src.scoring.base_scorer import BaseScorer


logger = logging.getLogger(__name__)


class HPSv2(BaseScorer):
    """
    HPSv2 (Human Preference Score v2) aesthetic quality scorer.
    
    HPSv2 predicts human aesthetic preferences for images, providing
    scores that correlate with human judgments of image quality.
    
    Model: xswu/HPSv2
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize HPSv2 scorer.
        
        Args:
            device: Device to run inference on ("auto", "cuda", "mps", "cpu")
        """
        self.model = None
        self.processor = None
        self.device = self._detect_device() if device == "auto" else device
        logger.info(f"HPSv2 initialized with device: {self.device}")
    
    def load(self, model_id: str = "xswu/HPSv2") -> None:
        """
        Load HPSv2 model and processor.
        
        Args:
            model_id: HuggingFace model identifier
        
        Raises:
            RuntimeError: If loading fails
        """
        try:
            logger.info(f"Loading HPSv2 model: {model_id}")
            
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to(self.device)
            self.model.eval()
            
            logger.info("HPSv2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load HPSv2 model: {e}")
            raise RuntimeError(f"HPSv2 loading failed: {e}") from e
    
    def compare(self, image_a: PILImage, image_b: PILImage) -> float:
        """
        Compare two images using HPSv2 aesthetic scoring.
        
        Args:
            image_a: First image to compare
            image_b: Second image to compare
        
        Returns:
            Score in [0.0, 1.0] where >0.5 means image_a has better aesthetics
        
        Raises:
            RuntimeError: If comparison fails
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("HPSv2 model not loaded. Call load() first.")
        
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
            logger.error(f"HPSv2 comparison failed: {e}")
            raise RuntimeError(f"HPSv2 comparison failed: {e}") from e
    
    def batch_compare(
        self,
        pairs: List[Tuple[PILImage, PILImage]]
    ) -> List[float]:
        """
        Compare multiple image pairs using HPSv2.
        
        Args:
            pairs: List of (image_a, image_b) tuples
        
        Returns:
            List of scores in [0.0, 1.0] for each pair
        
        Raises:
            RuntimeError: If batch comparison fails
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("HPSv2 model not loaded. Call load() first.")
        
        try:
            results = []
            
            # Process each pair
            # TODO: Could optimize with true batch processing
            for image_a, image_b in pairs:
                score = self.compare(image_a, image_b)
                results.append(score)
            
            return results
            
        except Exception as e:
            logger.error(f"HPSv2 batch comparison failed: {e}")
            raise RuntimeError(f"HPSv2 batch comparison failed: {e}") from e
    
    def _score_image(self, image: PILImage) -> float:
        """
        Compute HPSv2 score for a single image.

        HPSv2 is built on CLIP, which is a dual-encoder (image + text). Calling
        the full forward pass with image-only inputs raises
        "You have to specify input_ids" because the text encoder never receives
        tokens. Use get_image_features() instead — it runs only the vision
        encoder and returns the image embedding, whose L2 norm we use as the
        quality score (higher norm = higher aesthetic score).

        Args:
            image: Image to score

        Returns:
            Aesthetic quality score (higher is better)
        """
        # Process image only — no text tokens needed
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            # image-only path: avoids "You have to specify input_ids"
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
        Unload HPSv2 model from memory.
        """
        if self.model is not None:
            logger.info("Unloading HPSv2 model")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            
            # Clear GPU cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()
            
            logger.info("HPSv2 model unloaded")
