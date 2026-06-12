"""
BLIP-2 model adapter for image encoding.

This module provides the BLIP2Adapter class that wraps Salesforce's BLIP-2
model with OPT-2.7B for generating image embeddings. BLIP-2 uses 4-bit
quantization for memory efficiency.
"""

import logging
from typing import Dict, Any

import torch
from torch import Tensor
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from PIL.Image import Image as PILImage

from src.models.base_vision_model import BaseVisionModel


logger = logging.getLogger(__name__)


class BLIP2Adapter(BaseVisionModel):
    """
    Adapter for Salesforce BLIP-2 vision-language model with OPT-2.7B.
    
    BLIP-2 bridges frozen image encoders and large language models using
    a lightweight Querying Transformer (Q-Former). This adapter uses BLIP-2
    with the OPT-2.7B language model component.
    
    Memory footprint: ~3-4GB with 4-bit quantization
    Recommended quantization: 4bit (for memory efficiency)
    """
    
    def __init__(self):
        """Initialize BLIP-2 adapter with common attributes."""
        super().__init__()
        self.model = None
        self.processor = None
        self.device = None
        self.hf_id = None
    
    def load(self, config: Dict[str, Any]) -> None:
        """
        Load BLIP-2 model with specified configuration.
        
        Args:
            config: Dictionary containing:
                - hf_id: HuggingFace model identifier (e.g., "Salesforce/blip2-opt-2.7b")
                - quantization: Quantization mode (4bit, 8bit, fp16, fp32)
        
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If model loading fails
        """
        try:
            # Extract configuration
            self.hf_id = config.get("hf_id")
            quantization = config.get("quantization", "4bit")
            
            if not self.hf_id:
                raise ValueError("hf_id is required in config")
            
            # Auto-detect device
            self.device = self.detect_device()
            logger.info(f"Loading BLIP-2 model on device: {self.device}")
            
            logger.info(f"Loading BLIP-2 model: {self.hf_id} with {quantization} quantization")
            
            # Configure quantization
            load_in_4bit = False
            load_in_8bit = False
            torch_dtype = torch.float32
            cuda_available = torch.cuda.is_available()

            if quantization == "4bit":
                if cuda_available:
                    load_in_4bit = True
                    torch_dtype = torch.float16
                    logger.info("Using 4-bit quantization for OPT-2.7B component")
                else:
                    torch_dtype = torch.float16
                    logger.warning(
                        "4-bit quantization requires CUDA which is not available "
                        f"on this device ({self.device}). Falling back to fp16."
                    )
            elif quantization == "8bit":
                if cuda_available:
                    load_in_8bit = True
                    torch_dtype = torch.float16
                    logger.info("Using 8-bit quantization")
                else:
                    torch_dtype = torch.float16
                    logger.warning(
                        "8-bit quantization requires CUDA which is not available "
                        f"on this device ({self.device}). Falling back to fp16."
                    )
            elif quantization == "fp16":
                torch_dtype = torch.float16
                logger.info("Using FP16 precision")
            elif quantization == "fp32":
                torch_dtype = torch.float32
                logger.info("Using FP32 precision (no quantization)")
            else:
                logger.warning(
                    f"Unknown quantization '{quantization}'. "
                    f"Falling back to fp16 for memory efficiency."
                )
                torch_dtype = torch.float16
            
            # Load processor
            logger.info(f"Loading BLIP-2 processor from {self.hf_id}")
            self.processor = Blip2Processor.from_pretrained(self.hf_id)
            
            # Load model with quantization
            logger.info(f"Loading BLIP-2 model with quantization settings")
            
            if load_in_4bit or load_in_8bit:
                # Use bitsandbytes for quantization
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.hf_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch_dtype
                )
            else:
                # Regular loading without quantization
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.hf_id,
                    torch_dtype=torch_dtype
                ).to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info(f"BLIP-2 model loaded successfully on {self.device}")
            logger.info(f"Approximate memory footprint: {self.get_memory_footprint() / (1024**3):.2f} GB")
            
        except Exception as e:
            logger.error(f"Failed to load BLIP-2 model: {e}")
            raise RuntimeError(f"BLIP-2 model loading failed: {e}") from e
    
    def encode_image(self, image: PILImage) -> Tensor:
        """
        Generate embedding vector for an image using BLIP-2.
        
        Args:
            image: PIL Image object to encode
        
        Returns:
            Tensor containing the normalized image embedding
        
        Raises:
            RuntimeError: If encoding fails or model not loaded
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # Preprocess image
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Handle dtype conversion for quantized models
            if next(self.model.parameters()).dtype in [torch.float16, torch.bfloat16]:
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(next(self.model.parameters()).dtype)
            
            # Generate embedding
            with torch.no_grad():
                # Get vision embeddings from BLIP-2 vision encoder
                vision_outputs = self.model.vision_model(
                    pixel_values=inputs["pixel_values"]
                )
                
                # Use pooled output (CLS token)
                image_features = vision_outputs.pooler_output
                
                # Pass through Q-Former to get query embeddings
                # Q-Former bridges vision and language modalities
                query_outputs = self.model.qformer(
                    query_embeds=self.model.query_tokens.expand(image_features.shape[0], -1, -1),
                    encoder_hidden_states=vision_outputs.last_hidden_state,
                    encoder_attention_mask=torch.ones(
                        vision_outputs.last_hidden_state.shape[:2],
                        dtype=torch.long,
                        device=self.device
                    )
                )
                
                # Average pool query outputs to get single embedding
                query_features = query_outputs.last_hidden_state.mean(dim=1)
                
                # Normalize embedding (for cosine similarity)
                query_features = query_features / query_features.norm(dim=-1, keepdim=True)
            
            return query_features.squeeze(0).cpu()
            
        except Exception as e:
            logger.error(f"Failed to encode image with BLIP-2: {e}")
            raise RuntimeError(f"BLIP-2 image encoding failed: {e}") from e
    
    def unload(self) -> None:
        """
        Unload BLIP-2 model from memory and free GPU resources.
        
        This method:
        - Deletes model and processor from memory
        - Clears GPU cache
        - Resets internal state
        
        Raises:
            RuntimeError: If unloading fails
        """
        try:
            if self.model is not None:
                logger.info("Unloading BLIP-2 model from memory")
                
                # Delete model and processor
                del self.model
                del self.processor
                self.model = None
                self.processor = None
                
                # Clear GPU cache
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache")
                elif self.device == "mps":
                    torch.mps.empty_cache()
                    logger.info("Cleared MPS cache")
                
                # Reset state
                self.device = None
                self.hf_id = None
                
                logger.info("BLIP-2 model unloaded successfully")
            else:
                logger.warning("Attempted to unload BLIP-2 model but model was not loaded")
                
        except Exception as e:
            logger.error(f"Failed to unload BLIP-2 model: {e}")
            raise RuntimeError(f"BLIP-2 model unloading failed: {e}") from e
    
    def get_memory_footprint(self) -> int:
        """
        Get approximate memory usage of the loaded BLIP-2 model.
        
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
            logger.error(f"Failed to calculate BLIP-2 memory footprint: {e}")
            raise RuntimeError(f"Memory footprint calculation failed: {e}") from e
