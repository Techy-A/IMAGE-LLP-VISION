"""
CLIPScore metric for prompt-image alignment scoring.

CLIPScore measures how well an image matches its text prompt by computing
the cosine similarity between CLIP image and text embeddings. Unlike other
scorers that assess aesthetic quality, this scorer evaluates prompt fidelity.
"""

import logging
import math
from typing import List, Tuple, Optional

import torch
from PIL.Image import Image as PILImage

from src.scoring.base_scorer import BaseScorer
from src.models.clip_adapter import CLIPAdapter


logger = logging.getLogger(__name__)


class CLIPScore(BaseScorer):
    """
    CLIPScore prompt-image alignment scorer.
    
    Uses CLIP to measure how well an image matches its text prompt by
    computing cosine similarity between image and text embeddings in
    CLIP's shared embedding space.
    
    This is the only scorer in the ensemble that evaluates prompt fidelity
    rather than pure aesthetic quality, making it a crucial signal for
    text-to-image generation evaluation.
    
    Model: openai/clip-vit-large-patch14 (fp16)
    Memory footprint: ~1GB
    """
    
    def __init__(self, clip_model: Optional[CLIPAdapter] = None, device: str = "auto"):
        """
        Initialize CLIPScore scorer.
        
        Args:
            clip_model: Pre-loaded CLIPAdapter instance (optional, for dependency injection)
            device: Device to run inference on ("auto", "cuda", "mps", "cpu")
        """
        self.clip_model = clip_model
        self.device = self._detect_device() if device == "auto" else device
        self._owns_model = clip_model is None  # Track if we own the model for cleanup
        logger.info(f"CLIPScore initialized with device: {self.device}")
    
    def load(self, model_id: str = "openai/clip-vit-large-patch14") -> None:
        """
        Load CLIP model for scoring.
        
        Args:
            model_id: HuggingFace model identifier or config dict
        
        Raises:
            RuntimeError: If loading fails
        """
        # If model was injected, don't reload
        if self.clip_model is not None and not self._owns_model:
            logger.info("CLIPScore using pre-loaded CLIP model (dependency injection)")
            return
        
        try:
            logger.info(f"Loading CLIP model for CLIPScore: {model_id}")
            
            # Create and load CLIP adapter
            self.clip_model = CLIPAdapter()
            
            # Build config
            config = {
                "hf_id": model_id,
                "quantization": "fp16"  # CLIP works best with fp16
            }
            
            self.clip_model.load(config)
            self._owns_model = True
            
            logger.info("CLIPScore model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model for CLIPScore: {e}")
            raise RuntimeError(f"CLIPScore loading failed: {e}") from e
    
    def compare(
        self,
        image_a: PILImage,
        image_b: PILImage,
        prompt_a: Optional[str] = None,
        prompt_b: Optional[str] = None
    ) -> float:
        """
        Compare two images using CLIP prompt-image alignment.
        
        Computes CLIP similarity between each image and its prompt, then
        returns a normalized score indicating which image better matches
        its prompt.
        
        Args:
            image_a: First image to compare
            image_b: Second image to compare
            prompt_a: Text prompt for image_a (required for meaningful scoring)
            prompt_b: Text prompt for image_b (required for meaningful scoring)
        
        Returns:
            Score in [0.0, 1.0] where >0.5 means image_a has better prompt alignment
        
        Raises:
            RuntimeError: If comparison fails
            ValueError: If prompts are not provided
        """
        if self.clip_model is None:
            raise RuntimeError("CLIP model not loaded. Call load() first.")
        
        if prompt_a is None or prompt_b is None:
            raise ValueError(
                "CLIPScore requires prompts for both images. "
                "Pass prompt_a and prompt_b parameters."
            )
        
        try:
            # Compute CLIP similarity for each image-prompt pair
            similarity_a = self._compute_similarity(image_a, prompt_a)
            similarity_b = self._compute_similarity(image_b, prompt_b)
            
            # Compute difference
            similarity_diff = similarity_a - similarity_b
            
            # Apply amplified sigmoid normalization
            # CLIP similarities are typically 0.20-0.35 with diffs of 0.01-0.03
            # Use ×20 amplification to create meaningful spread
            normalized_score = self._sigmoid(similarity_diff)
            
            # DEBUG: Log raw scores for first few matches
            logger.debug(
                f"CLIPScore RAW: sim_a={similarity_a:.4f}, sim_b={similarity_b:.4f}, "
                f"diff={similarity_diff:.4f}, sigmoid={normalized_score:.4f}"
            )
            
            return float(normalized_score)
            
        except Exception as e:
            logger.error(f"CLIPScore comparison failed: {e}")
            raise RuntimeError(f"CLIPScore comparison failed: {e}") from e
    
    def batch_compare(
        self,
        pairs: List[Tuple[PILImage, PILImage]],
        prompts: Optional[List[Tuple[str, str]]] = None
    ) -> List[float]:
        """
        Compare multiple image pairs using CLIPScore.
        
        Args:
            pairs: List of (image_a, image_b) tuples
            prompts: List of (prompt_a, prompt_b) tuples (required)
        
        Returns:
            List of scores in [0.0, 1.0] for each pair
        
        Raises:
            RuntimeError: If batch comparison fails
            ValueError: If prompts are not provided
        """
        if self.clip_model is None:
            raise RuntimeError("CLIP model not loaded. Call load() first.")
        
        if prompts is None or len(prompts) != len(pairs):
            raise ValueError(
                "CLIPScore requires prompts for all image pairs. "
                "Pass prompts list with same length as pairs."
            )
        
        try:
            results = []
            
            # Process each pair
            # TODO: Could optimize with true batch processing
            for (image_a, image_b), (prompt_a, prompt_b) in zip(pairs, prompts):
                score = self.compare(image_a, image_b, prompt_a, prompt_b)
                results.append(score)
            
            return results
            
        except Exception as e:
            logger.error(f"CLIPScore batch comparison failed: {e}")
            raise RuntimeError(f"CLIPScore batch comparison failed: {e}") from e
    
    def _compute_similarity(self, image: PILImage, text: str) -> float:
        """
        Compute CLIP cosine similarity between image and text.
        
        Args:
            image: PIL Image
            text: Text prompt
        
        Returns:
            Cosine similarity score (typically 0.20-0.35 for matching prompts)
        """
        try:
            # Get image embedding
            image_features = self.clip_model.encode_image(image)
            
            # Get text embedding
            # Access the underlying open_clip model for text encoding
            import open_clip
            tokenizer = open_clip.get_tokenizer(self.clip_model.model_name)
            text_tokens = tokenizer([text]).to(self.clip_model.device)
            
            with torch.no_grad():
                text_features = self.clip_model.model.encode_text(text_tokens)
                # Normalize text embedding
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            # Both features are already normalized, so dot product = cosine similarity
            similarity = (image_features @ text_features.squeeze(0).cpu()).item()
            
            return similarity
            
        except Exception as e:
            logger.error(f"Failed to compute CLIP similarity: {e}")
            raise RuntimeError(f"CLIP similarity computation failed: {e}") from e
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        """
        Apply amplified sigmoid function to normalize similarity difference.
        
        CLIP similarities for same-prompt images are typically in the
        0.20-0.35 range with small differences of 0.01-0.03. We use ×20
        amplification to create meaningful spread without saturation.
        
        Args:
            x: Similarity difference (typically in range [-0.1, 0.1])
        
        Returns:
            Normalized value in [0.0, 1.0]
        
        Note:
            Amplification factor: ×20
            - diff=0.01 → 0.5498 (weak preference)
            - diff=0.02 → 0.5987 (moderate preference)
            - diff=0.03 → 0.6457 (strong preference)
            - diff=0.05 → 0.7311 (decisive)
            
            This scaling provides good discrimination for typical CLIP
            similarity differences without oversaturating on noise.
        """
        # Amplify by 20x to preserve small differences in CLIP similarities
        scaled_x = x * 20.0
        return 1.0 / (1.0 + math.exp(-scaled_x))
    
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
        Unload CLIP model from memory.
        
        Only unloads if this scorer owns the model (not dependency-injected).
        """
        if self.clip_model is not None and self._owns_model:
            logger.info("Unloading CLIPScore model")
            self.clip_model.unload()
            self.clip_model = None
            logger.info("CLIPScore model unloaded")
        elif self.clip_model is not None:
            logger.info("CLIPScore using shared CLIP model, skipping unload")

