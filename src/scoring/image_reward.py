"""
ImageReward metric for human preference alignment scoring.

ImageReward uses a model trained on human feedback to score images based
on how well they align with human preferences. This implementation tries
the native ImageReward library first, then falls back to HuggingFace.
"""

import logging
import math
from typing import List, Tuple, Optional

import torch
from PIL.Image import Image as PILImage

from src.scoring.base_scorer import BaseScorer


logger = logging.getLogger(__name__)


def _patch_transformers_compat() -> None:
    """
    Make the native ImageReward (v1.5) package import on modern transformers.

    ImageReward's bundled BLIP (`models/BLIP/med.py`) does:
        from transformers.modeling_utils import (
            PreTrainedModel, apply_chunking_to_forward,
            find_pruneable_heads_and_indices, prune_linear_layer)
    Newer transformers moved these to `transformers.pytorch_utils` (4.x) and
    removed `find_pruneable_heads_and_indices` entirely (5.x). This re-exposes
    them on `transformers.modeling_utils` so the import succeeds. No-op when the
    symbols are already present (e.g. transformers 4.30-era installs).
    """
    try:
        import transformers.modeling_utils as mu
    except Exception:
        return

    # apply_chunking_to_forward / prune_linear_layer: re-export from pytorch_utils
    for sym in ("apply_chunking_to_forward", "prune_linear_layer"):
        if not hasattr(mu, sym):
            try:
                from transformers import pytorch_utils as pu
                if hasattr(pu, sym):
                    setattr(mu, sym, getattr(pu, sym))
            except Exception:
                pass

    # find_pruneable_heads_and_indices: re-export if present, else provide it
    if not hasattr(mu, "find_pruneable_heads_and_indices"):
        impl = None
        try:
            from transformers.pytorch_utils import find_pruneable_heads_and_indices as impl
        except Exception:
            impl = None
        if impl is None:
            import torch

            def impl(heads, n_heads, head_size, already_pruned_heads):
                mask = torch.ones(n_heads, head_size)
                heads = set(heads) - already_pruned_heads
                for head in heads:
                    head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                    mask[head] = 0
                mask = mask.view(-1).contiguous().eq(1)
                index = torch.arange(len(mask))[mask].long()
                return heads, index

        mu.find_pruneable_heads_and_indices = impl


class ImageReward(BaseScorer):
    """
    ImageReward human preference alignment scorer.
    
    ImageReward predicts how well images align with human preferences,
    trained on human feedback data. Uses a fallback chain:
    1. Native ImageReward library (if available)
    2. HuggingFace transformers implementation
    
    Paper: "ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation"
    Model: THUDM/ImageReward
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize ImageReward scorer.
        
        Args:
            device: Device to run inference on ("auto", "cuda", "mps", "cpu")
        """
        self.model = None
        self.device = self._detect_device() if device == "auto" else device
        self.using_native = False
        logger.info(f"ImageReward initialized with device: {self.device}")
    
    def load(self, model_id: str = "THUDM/ImageReward") -> None:
        """
        Load ImageReward model.
        
        Tries native ImageReward library first, falls back to HuggingFace.
        
        Args:
            model_id: Model identifier
        
        Raises:
            RuntimeError: If loading fails with both methods
        """
        try:
            # Try native ImageReward library first
            logger.info("Attempting to load ImageReward using native library")
            self._load_native(model_id)
            self.using_native = True
            logger.info("ImageReward loaded successfully using native library")
            
        except Exception as native_error:
            logger.warning(
                f"Failed to load native ImageReward library: {native_error}. "
                f"Falling back to HuggingFace implementation."
            )
            
            try:
                # Fall back to HuggingFace
                logger.info("Attempting to load ImageReward using HuggingFace")
                self._load_huggingface(model_id)
                self.using_native = False
                logger.info("ImageReward loaded successfully using HuggingFace")
                
            except Exception as hf_error:
                logger.error(f"Failed to load ImageReward with both methods")
                raise RuntimeError(
                    f"ImageReward loading failed. "
                    f"Native error: {native_error}. "
                    f"HuggingFace error: {hf_error}"
                ) from hf_error
    
    def _load_native(self, model_id: str) -> None:
        """
        Load ImageReward using native library.
        
        Args:
            model_id: Model identifier
        
        Raises:
            ImportError: If ImageReward library not available
            RuntimeError: If loading fails
        """
        # ImageReward 1.5 bundles an old BLIP that imports symbols which newer
        # transformers relocated/removed. Re-expose them before importing so the
        # package works on transformers 4.x AND 5.x.
        _patch_transformers_compat()

        import ImageReward as RM

        # ImageReward's native loader only knows short model names (its
        # available models are ["ImageReward-v1.0"]); an HF repo id like
        # "THUDM/ImageReward" is NOT a valid native name. Map repo ids to the
        # canonical native model and pass bare names / local paths through as-is.
        native_name = "ImageReward-v1.0" if "/" in str(model_id) else model_id
        self.model = RM.load(native_name, device=self.device)
    
    def _load_huggingface(self, model_id: str) -> None:
        """
        Load ImageReward using HuggingFace transformers.
        
        Args:
            model_id: HuggingFace model identifier
        
        Raises:
            RuntimeError: If loading fails
        """
        from transformers import AutoProcessor, AutoModel
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
    
    def compare(self, image_a: PILImage, image_b: PILImage) -> float:
        """
        Compare two images using ImageReward scoring.
        
        Args:
            image_a: First image to compare
            image_b: Second image to compare
        
        Returns:
            Score in [0.0, 1.0] where >0.5 means image_a better aligns with human preferences
        
        Raises:
            RuntimeError: If comparison fails
        """
        if self.model is None:
            raise RuntimeError("ImageReward model not loaded. Call load() first.")
        
        try:
            # Compute scores for both images
            if self.using_native:
                score_a = self._score_image_native(image_a)
                score_b = self._score_image_native(image_b)
            else:
                score_a = self._score_image_huggingface(image_a)
                score_b = self._score_image_huggingface(image_b)
            
            # Compute difference
            score_diff = score_a - score_b
            
            # Apply sigmoid normalization to get value in [0.0, 1.0]
            normalized_score = self._sigmoid(score_diff)
            
            return float(normalized_score)
            
        except Exception as e:
            logger.error(f"ImageReward comparison failed: {e}")
            raise RuntimeError(f"ImageReward comparison failed: {e}") from e
    
    def batch_compare(
        self,
        pairs: List[Tuple[PILImage, PILImage]]
    ) -> List[float]:
        """
        Compare multiple image pairs using ImageReward.
        
        Args:
            pairs: List of (image_a, image_b) tuples
        
        Returns:
            List of scores in [0.0, 1.0] for each pair
        
        Raises:
            RuntimeError: If batch comparison fails
        """
        if self.model is None:
            raise RuntimeError("ImageReward model not loaded. Call load() first.")
        
        try:
            results = []
            
            # Process each pair
            # TODO: Could optimize with true batch processing
            for image_a, image_b in pairs:
                score = self.compare(image_a, image_b)
                results.append(score)
            
            return results
            
        except Exception as e:
            logger.error(f"ImageReward batch comparison failed: {e}")
            raise RuntimeError(f"ImageReward batch comparison failed: {e}") from e
    
    def _score_image_native(self, image: PILImage) -> float:
        """
        Compute ImageReward score using native library.
        
        Args:
            image: Image to score
        
        Returns:
            Preference alignment score (higher is better)
        """
        # Native library expects prompt, but for scoring we use empty prompt
        score = self.model.score("", image)
        return float(score)
    
    def _score_image_huggingface(self, image: PILImage) -> float:
        """
        Compute ImageReward score using HuggingFace.
        
        Args:
            image: Image to score
        
        Returns:
            Preference alignment score (higher is better)
        """
        # Process image
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Get score
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Extract reward score
            if hasattr(outputs, 'logits'):
                score = outputs.logits
            else:
                # Fall back to using image features
                image_features = outputs.last_hidden_state[:, 0]  # CLS token
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
        Unload ImageReward model from memory.
        """
        if self.model is not None:
            logger.info("Unloading ImageReward model")
            del self.model
            if hasattr(self, 'processor'):
                del self.processor
            self.model = None
            
            # Clear GPU cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()
            
            logger.info("ImageReward model unloaded")
