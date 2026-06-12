"""
CLIP model adapter for image encoding.

This module provides the CLIPAdapter class that wraps OpenAI's CLIP model
for generating image embeddings. CLIP uses FP16 quantization by default
and is memory-efficient (~1GB).
"""

import logging
from typing import Dict, Any

import torch
from torch import Tensor
import open_clip
from PIL.Image import Image as PILImage

from src.models.base_vision_model import BaseVisionModel


logger = logging.getLogger(__name__)


class CLIPAdapter(BaseVisionModel):
    """
    Adapter for OpenAI CLIP vision-language model.
    
    CLIP (Contrastive Language-Image Pre-training) is a lightweight model
    that encodes images and text into a shared embedding space. This adapter
    uses the open-clip-torch implementation for flexibility and performance.
    
    Memory footprint: ~1GB with FP16 quantization
    Recommended quantization: fp16 (default for CLIP)
    """
    
    def __init__(self):
        """Initialize CLIP adapter with common attributes."""
        super().__init__()
        self.model = None
        self.preprocess = None
        self.device = None
        self.model_name = None
        self.pretrained_dataset = None
    
    def load(self, config: Dict[str, Any]) -> None:
        """
        Load CLIP model with specified configuration.
        
        Args:
            config: Dictionary containing:
                - hf_id: HuggingFace model identifier (e.g., "openai/clip-vit-large-patch14")
                - quantization: Quantization mode (fp16 recommended for CLIP)
                - Additional optional parameters
        
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If model loading fails
        """
        # Extract configuration
        hf_id = config.get("hf_id")
        quantization = config.get("quantization", "fp16")
        
        if not hf_id:
            raise ValueError("hf_id is required in config")
            
        try:
            # Auto-detect device
            self.device = self.detect_device()
            logger.info(f"Loading CLIP model on device: {self.device}")
            
            # Parse model name from HuggingFace ID
            # Expected format: "openai/clip-vit-large-patch14" or similar
            # For open_clip, we need the model architecture name
            self.model_name = self._parse_model_name(hf_id)
            self.pretrained_dataset = config.get("pretrained", "openai")
            
            logger.info(f"Loading CLIP model: {self.model_name} (pretrained: {self.pretrained_dataset})")
            
            # Load model and preprocessing
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained_dataset,
                device=self.device
            )
            
            # Set quantization
            if quantization == "fp16":
                logger.info("Converting model to FP16")
                self.model = self.model.half()
            elif quantization == "fp32":
                logger.info("Using FP32 precision (no quantization)")
                self.model = self.model.float()
            else:
                logger.warning(
                    f"Quantization '{quantization}' not optimal for CLIP. "
                    f"Recommended: fp16 or fp32. Falling back to fp16."
                )
                self.model = self.model.half()
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info(f"CLIP model loaded successfully on {self.device}")
            logger.info(f"Approximate memory footprint: {self.get_memory_footprint() / (1024**3):.2f} GB")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise RuntimeError(f"CLIP model loading failed: {e}") from e
    
    def encode_image(self, image: PILImage) -> Tensor:
        """
        Generate embedding vector for an image using CLIP.
        
        Args:
            image: PIL Image object to encode
        
        Returns:
            Tensor containing the normalized image embedding
        
        Raises:
            RuntimeError: If encoding fails or model not loaded
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # Preprocess image
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Handle FP16
            if next(self.model.parameters()).dtype == torch.float16:
                image_input = image_input.half()
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                # Normalize embedding (CLIP uses cosine similarity)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.squeeze(0).cpu()
            
        except Exception as e:
            logger.error(f"Failed to encode image with CLIP: {e}")
            raise RuntimeError(f"CLIP image encoding failed: {e}") from e
    
    def unload(self) -> None:
        """
        Unload CLIP model from memory and free GPU resources.
        
        This method:
        - Deletes model from memory
        - Clears GPU cache
        - Resets internal state
        
        Raises:
            RuntimeError: If unloading fails
        """
        try:
            if self.model is not None:
                logger.info("Unloading CLIP model from memory")
                
                # Delete model
                del self.model
                del self.preprocess
                self.model = None
                self.preprocess = None
                
                # Clear GPU cache
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache")
                elif self.device == "mps":
                    torch.mps.empty_cache()
                    logger.info("Cleared MPS cache")
                
                # Reset state
                self.device = None
                self.model_name = None
                self.pretrained_dataset = None
                
                logger.info("CLIP model unloaded successfully")
            else:
                logger.warning("Attempted to unload CLIP model but model was not loaded")
                
        except Exception as e:
            logger.error(f"Failed to unload CLIP model: {e}")
            raise RuntimeError(f"CLIP model unloading failed: {e}") from e
    
    def get_memory_footprint(self) -> int:
        """
        Get approximate memory usage of the loaded CLIP model.
        
        Returns:
            Memory usage in bytes
        
        Raises:
            RuntimeError: If model not loaded or memory calculation fails
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            total_bytes = 0
            
            # Calculate parameter memory
            for param in self.model.parameters():
                param_bytes = param.numel() * param.element_size()
                total_bytes += param_bytes
            
            # Calculate buffer memory (batch norm running stats, etc.)
            for buffer in self.model.buffers():
                buffer_bytes = buffer.numel() * buffer.element_size()
                total_bytes += buffer_bytes
            
            # Add overhead estimate (20% for activations, intermediate tensors)
            total_bytes = int(total_bytes * 1.2)
            
            return total_bytes
            
        except Exception as e:
            logger.error(f"Failed to calculate CLIP memory footprint: {e}")
            raise RuntimeError(f"Memory footprint calculation failed: {e}") from e
    
    def _parse_model_name(self, hf_id: str) -> str:
        """
        Parse model architecture name from HuggingFace ID.
        
        Args:
            hf_id: HuggingFace model identifier
        
        Returns:
            Model architecture name for open_clip
        
        Examples:
            "openai/clip-vit-large-patch14" → "ViT-L-14"
            "openai/clip-vit-base-patch32" → "ViT-B-32"
        """
        # Extract model name after the slash
        if "/" in hf_id:
            model_part = hf_id.split("/")[-1]
        else:
            model_part = hf_id
        
        # Map common HuggingFace names to open_clip names
        mapping = {
            "clip-vit-large-patch14": "ViT-L-14",
            "clip-vit-large-patch14-336": "ViT-L-14-336",
            "clip-vit-base-patch32": "ViT-B-32",
            "clip-vit-base-patch16": "ViT-B-16",
            "clip-vit-huge-patch14": "ViT-H-14",
        }
        
        # Check if we have a direct mapping
        if model_part in mapping:
            return mapping[model_part]
        
        # If no mapping found, try to use the model_part as-is
        # (might already be in open_clip format)
        logger.warning(
            f"No mapping found for model '{model_part}'. "
            f"Attempting to use as-is. If loading fails, check model name format."
        )
        return model_part
