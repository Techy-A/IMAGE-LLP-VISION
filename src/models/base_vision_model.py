"""
Abstract base class for vision model adapters.

This module provides the BaseVisionModel abstract class that defines the
interface all vision model adapters must implement. It includes concrete
helper methods for quantization configuration and device detection.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional

import torch
from torch import Tensor
from transformers import BitsAndBytesConfig
from PIL.Image import Image as PILImage


logger = logging.getLogger(__name__)


class BaseVisionModel(ABC):
    """
    Abstract base class for vision model adapters.
    
    All vision model implementations (CLIP, BLIP, BLIP-2, InstructBLIP) must
    inherit from this class and implement the abstract methods.
    
    Provides concrete helper methods for:
    - Device auto-detection (CUDA → MPS → CPU)
    - Quantization configuration (4bit, 8bit, fp16, fp32)
    """
    
    def __init__(self):
        """Initialize base vision model with common attributes."""
        self.device: Optional[str] = None
        self.model: Optional[Any] = None
    
    @abstractmethod
    def load(self, config: Dict[str, Any]) -> None:
        """
        Load the vision model with specified configuration.
        
        Args:
            config: Dictionary containing model configuration including:
                - hf_id: HuggingFace model identifier
                - quantization: Quantization mode ("4bit", "8bit", "fp16", "fp32")
                - Additional model-specific parameters
        
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If model loading fails
        """
        pass
    
    @abstractmethod
    def encode_image(self, image: PILImage) -> Tensor:
        """
        Generate embedding vector for an image.
        
        Args:
            image: PIL Image object to encode
        
        Returns:
            Tensor containing the image embedding
        
        Raises:
            RuntimeError: If encoding fails or model not loaded
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """
        Unload model from memory and free GPU resources.
        
        This method should:
        - Delete model from memory
        - Clear GPU cache
        - Reset internal state
        
        Raises:
            RuntimeError: If unloading fails
        """
        pass
    
    @abstractmethod
    def get_memory_footprint(self) -> int:
        """
        Get approximate memory usage of the loaded model.
        
        Returns:
            Memory usage in bytes
        
        Raises:
            RuntimeError: If model not loaded or memory calculation fails
        """
        pass
    
    def detect_device(self) -> str:
        """
        Auto-detect the best available compute device.
        
        Detection order:
        1. CUDA (NVIDIA GPUs)
        2. MPS (Apple Silicon)
        3. CPU (fallback)
        
        Returns:
            Device string: "cuda", "mps", or "cpu"
        """
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("Detected device: cuda (NVIDIA GPU)")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Detected device: mps (Apple Silicon)")
        else:
            device = "cpu"
            logger.info("Detected device: cpu (no GPU acceleration)")
        
        return device
    
    def load_with_quantization(self, quantization: str) -> Optional[BitsAndBytesConfig]:
        """
        Create BitsAndBytesConfig for model quantization.
        
        Quantization reduces memory footprint at the cost of some precision:
        - 4bit: ~4x reduction, minimal quality loss (recommended for large models)
        - 8bit: ~2x reduction, negligible quality loss
        - fp16: ~2x reduction vs fp32, standard precision
        - fp32: No reduction, full precision
        
        Args:
            quantization: Quantization mode ("4bit", "8bit", "fp16", "fp32")
        
        Returns:
            BitsAndBytesConfig for 4bit/8bit, None for fp16/fp32
        
        Raises:
            ValueError: If quantization mode is invalid
        """
        valid_modes = ["4bit", "8bit", "fp16", "fp32"]
        if quantization not in valid_modes:
            raise ValueError(
                f"Invalid quantization mode '{quantization}'. "
                f"Must be one of: {valid_modes}"
            )
        
        if quantization == "4bit":
            logger.info("Configuring 4-bit quantization (NF4 with double quantization)")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",  # NormalFloat4 quantization
                bnb_4bit_use_double_quant=True  # Additional memory savings
            )
        elif quantization == "8bit":
            logger.info("Configuring 8-bit quantization")
            return BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            # fp16 and fp32 don't use BitsAndBytesConfig
            # Calling code should use torch_dtype directly
            logger.info(f"Using {quantization} precision (no quantization config needed)")
            return None
