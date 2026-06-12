"""
InstructBLIP model adapter for image encoding and visual question answering.

This module provides the InstructBLIPAdapter class that wraps Salesforce's
InstructBLIP model with Vicuna-7B for generating image embeddings and
answering visual questions. InstructBLIP uses 4-bit quantization for memory
efficiency.
"""

import logging
from typing import Dict, Any, Optional

import torch
from torch import Tensor
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
from PIL.Image import Image as PILImage

from src.models.base_vision_model import BaseVisionModel


logger = logging.getLogger(__name__)


class InstructBLIPAdapter(BaseVisionModel):
    """
    Adapter for Salesforce InstructBLIP vision-language model with Vicuna-7B.
    
    InstructBLIP is an instruction-tuned BLIP-2 model that follows natural
    language instructions for vision-language tasks. This adapter supports
    both image encoding and text-conditioned visual question answering.
    
    This instance is designed to be shared by VQAScore and VLMJudge scorers
    to avoid duplicate model loading.
    
    Memory footprint: ~7-8GB with 4-bit quantization
    Recommended quantization: 4bit (required for Vicuna-7B on consumer GPUs)
    """
    
    def __init__(self):
        """Initialize InstructBLIP adapter with common attributes."""
        super().__init__()
        self.model = None
        self.processor = None
        self.device = None
        self.hf_id = None
    
    def load(self, config: Dict[str, Any]) -> None:
        """
        Load InstructBLIP model with specified configuration.
        
        Args:
            config: Dictionary containing:
                - hf_id: HuggingFace model identifier (e.g., "Salesforce/instructblip-vicuna-7b")
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
            logger.info(f"Loading InstructBLIP model on device: {self.device}")
            
            logger.info(f"Loading InstructBLIP model: {self.hf_id} with {quantization} quantization")
            
            # Configure quantization
            load_in_4bit = False
            load_in_8bit = False
            torch_dtype = torch.float32

            # bitsandbytes 4-bit/8-bit quantization requires CUDA.
            # On Mac (MPS) or CPU-only systems, fall back to fp16 automatically.
            cuda_available = torch.cuda.is_available()

            if quantization == "4bit":
                if cuda_available:
                    load_in_4bit = True
                    torch_dtype = torch.float16
                    logger.info("Using 4-bit quantization for Vicuna-7B component")
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
                logger.warning(
                    "FP32 for InstructBLIP (7B params) requires ~28GB RAM. "
                    "Consider using 4-bit quantization on a CUDA GPU."
                )
            else:
                logger.warning(
                    f"Unknown quantization '{quantization}'. "
                    f"Falling back to fp16 for this device."
                )
                torch_dtype = torch.float16

            # Load processor
            logger.info(f"Loading InstructBLIP processor from {self.hf_id}")
            self.processor = InstructBlipProcessor.from_pretrained(self.hf_id)

            # Load model
            logger.info(f"Loading InstructBLIP model with quantization settings")

            if load_in_4bit or load_in_8bit:
                # Use bitsandbytes for quantization (CUDA only)
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

                self.model = InstructBlipForConditionalGeneration.from_pretrained(
                    self.hf_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch_dtype
                )
            else:
                # Regular fp16/fp32 loading — works on MPS and CPU
                # Use device_map="mps" or explicit .to(device) depending on availability
                if self.device == "mps":
                    self.model = InstructBlipForConditionalGeneration.from_pretrained(
                        self.hf_id,
                        torch_dtype=torch_dtype,
                        device_map={"": self.device}
                    )
                else:
                    self.model = InstructBlipForConditionalGeneration.from_pretrained(
                        self.hf_id,
                        torch_dtype=torch_dtype
                    ).to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info(f"InstructBLIP model loaded successfully on {self.device}")
            logger.info(f"Approximate memory footprint: {self.get_memory_footprint() / (1024**3):.2f} GB")
            
        except Exception as e:
            logger.error(f"Failed to load InstructBLIP model: {e}")
            raise RuntimeError(f"InstructBLIP model loading failed: {e}") from e
    
    def encode_image(self, image: PILImage) -> Tensor:
        """
        Generate embedding vector for an image using InstructBLIP.
        
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
            # Preprocess image (resize, normalize, convert to tensor)
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)

            # Handle dtype conversion for quantized models
            # bitsandbytes quantisation changes parameter dtype to fp16;
            # the processor always returns fp32 pixel_values, so we cast
            # them to match before forwarding to avoid dtype mismatch errors.
            if next(self.model.parameters()).dtype in [torch.float16, torch.bfloat16]:
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(next(self.model.parameters()).dtype)

            with torch.no_grad():
                # ── InstructBLIP image encoding path ─────────────────────
                # InstructBLIP architecture:
                #   Image → ViT encoder → Q-Former → language model (Vicuna-7B)
                #
                # For embedding purposes we stop at the Q-Former output.
                # Step 1: Run the ViT vision encoder to get patch embeddings.
                vision_outputs = self.model.vision_model(
                    pixel_values=inputs["pixel_values"]
                )

                # Step 2: Feed the ViT patch grid into the Q-Former.
                # Q-Former has 32 learnable "query tokens" that cross-attend
                # to all ViT patches and extract the semantically richest
                # representation. pooler_output (CLS token) provides a
                # compact summary of the full patch grid.
                image_features = vision_outputs.pooler_output

                query_outputs = self.model.qformer(
                    query_embeds=self.model.query_tokens.expand(image_features.shape[0], -1, -1),
                    encoder_hidden_states=vision_outputs.last_hidden_state,
                    encoder_attention_mask=torch.ones(
                        vision_outputs.last_hidden_state.shape[:2],
                        dtype=torch.long,
                        device=self.device
                    )
                )

                # Step 3: Average-pool the 32 query token outputs down to
                # a single vector. This collapses the sequence dimension and
                # gives a fixed-size image representation compatible with
                # cosine similarity comparisons.
                query_features = query_outputs.last_hidden_state.mean(dim=1)

                # L2-normalise so cosine similarity == dot product (and
                # avoids scale ambiguity when comparing different images).
                query_features = query_features / query_features.norm(dim=-1, keepdim=True)

            return query_features.squeeze(0).cpu()

        except Exception as e:
            logger.error(f"Failed to encode image with InstructBLIP: {e}")
            raise RuntimeError(f"InstructBLIP image encoding failed: {e}") from e
    
    def query_image(self, image: PILImage, prompt: str, max_new_tokens: int = 50) -> str:
        """
        Query the image with a text prompt using InstructBLIP.
        
        This method enables text-conditioned visual question answering,
        which is used by VQAScore and VLMJudge scorers.
        
        Args:
            image: PIL Image object to query
            prompt: Text prompt/question about the image
            max_new_tokens: Maximum number of tokens to generate
        
        Returns:
            Generated text response from the model
        
        Raises:
            RuntimeError: If query fails or model not loaded
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # Preprocess image and prompt together
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            # Handle dtype conversion for quantized models (same reason as encode_image)
            if next(self.model.parameters()).dtype in [torch.float16, torch.bfloat16]:
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(next(self.model.parameters()).dtype)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,   # Greedy / beam search — deterministic output
                                       # so the same image+prompt always returns the
                                       # same text (important for reproducibility).
                    num_beams=5,       # Beam search width (quality vs speed trade-off)
                    early_stopping=True
                )

            # Decode and strip leading/trailing whitespace
            response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

            return response.strip()

        except Exception as e:
            logger.error(f"Failed to query image with InstructBLIP: {e}")
            raise RuntimeError(f"InstructBLIP query failed: {e}") from e
    
    def unload(self) -> None:
        """
        Unload InstructBLIP model from memory and free GPU resources.
        
        This method:
        - Deletes model and processor from memory
        - Clears GPU cache
        - Resets internal state
        
        Raises:
            RuntimeError: If unloading fails
        """
        try:
            if self.model is not None:
                logger.info("Unloading InstructBLIP model from memory")
                
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
                
                logger.info("InstructBLIP model unloaded successfully")
            else:
                logger.warning("Attempted to unload InstructBLIP model but model was not loaded")
                
        except Exception as e:
            logger.error(f"Failed to unload InstructBLIP model: {e}")
            raise RuntimeError(f"InstructBLIP model unloading failed: {e}") from e
    
    def get_memory_footprint(self) -> int:
        """
        Get approximate memory usage of the loaded InstructBLIP model.
        
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
            logger.error(f"Failed to calculate InstructBLIP memory footprint: {e}")
            raise RuntimeError(f"Memory footprint calculation failed: {e}") from e
