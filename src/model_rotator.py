"""
Model rotation manager for memory-efficient sequential model loading.

This module implements the ModelRotator pattern that ensures only one large
vision-language model is loaded in memory at any given time, preventing VRAM
exhaustion during tournament execution.
"""

import logging
from typing import List, Optional, Dict, Any

from src.models.base_vision_model import BaseVisionModel


logger = logging.getLogger(__name__)


class ModelRotator:
    """
    Manages sequential loading and unloading of vision-language models.
    
    The ModelRotator ensures that at most one large model is loaded in GPU
    memory at any time by:
    1. Unloading the currently loaded model before loading the next
    2. Tracking memory usage against a configurable threshold
    3. Providing round-robin access to a list of models
    
    This pattern is critical for running multiple large models (e.g., InstructBLIP
    with Vicuna-7B, BLIP-2 with OPT-2.7B) on consumer GPUs with limited VRAM.
    
    Example:
        >>> models = [clip_adapter, blip_adapter, instructblip_adapter]
        >>> configs = {...}
        >>> rotator = ModelRotator(models, configs, max_memory=16 * 1024**3)
        >>> 
        >>> # First call loads CLIP
        >>> model = rotator.get_next_model()
        >>> embedding = model.encode_image(image)
        >>> 
        >>> # Second call unloads CLIP and loads BLIP
        >>> model = rotator.get_next_model()
        >>> embedding = model.encode_image(image)
    """
    
    def __init__(
        self,
        models: List[BaseVisionModel],
        model_configs: Dict[str, Dict[str, Any]],
        max_memory: Optional[int] = None
    ):
        """
        Initialize ModelRotator with models and memory constraints.
        
        Args:
            models: List of BaseVisionModel instances to rotate through
            model_configs: Dictionary mapping model class names to their configs
            max_memory: Maximum memory threshold in bytes (None = no limit)
        
        Raises:
            ValueError: If models list is empty or configs are invalid
        """
        if not models:
            raise ValueError("models list cannot be empty")
        
        if not model_configs:
            raise ValueError("model_configs cannot be empty")
        
        self.models = models
        self.model_configs = model_configs
        self.max_memory = max_memory
        self.current_model: Optional[BaseVisionModel] = None
        self.current_index: int = -1
        
        logger.info(
            f"ModelRotator initialized with {len(models)} models. "
            f"Max memory: {max_memory / (1024**3):.2f} GB" if max_memory else "No memory limit"
        )
    
    def get_next_model(self) -> BaseVisionModel:
        """
        Get the next model in rotation, unloading the current model first.
        
        This method implements the core rotation logic:
        1. Unload currently loaded model (if any)
        2. Advance to next model in round-robin order
        3. Load the next model with its configuration
        4. Check memory threshold (if configured)
        5. Return the loaded model
        
        Returns:
            The next loaded BaseVisionModel instance
        
        Raises:
            RuntimeError: If model loading fails or memory threshold exceeded
        """
        try:
            # Unload current model if one is loaded
            if self.current_model is not None:
                logger.info(
                    f"Unloading current model: {self.current_model.__class__.__name__}"
                )
                self.current_model.unload()
                self.current_model = None
            
            # Advance to next model in round-robin order
            self.current_index = (self.current_index + 1) % len(self.models)
            next_model = self.models[self.current_index]
            
            # Get model configuration
            model_name = next_model.__class__.__name__
            if model_name not in self.model_configs:
                raise ValueError(
                    f"No configuration found for model '{model_name}'. "
                    f"Available configs: {list(self.model_configs.keys())}"
                )
            
            config = self.model_configs[model_name]
            
            # Load the model
            logger.info(
                f"Loading model {self.current_index + 1}/{len(self.models)}: "
                f"{model_name}"
            )
            next_model.load(config)
            
            # Check memory threshold
            if self.max_memory is not None:
                footprint = next_model.get_memory_footprint()
                if footprint > self.max_memory:
                    # Unload the model immediately
                    next_model.unload()
                    raise RuntimeError(
                        f"Model '{model_name}' memory footprint "
                        f"({footprint / (1024**3):.2f} GB) exceeds "
                        f"maximum threshold ({self.max_memory / (1024**3):.2f} GB)"
                    )
                
                logger.info(
                    f"Model loaded successfully. "
                    f"Memory: {footprint / (1024**3):.2f} GB / "
                    f"{self.max_memory / (1024**3):.2f} GB"
                )
            
            # Set as current model
            self.current_model = next_model
            
            return self.current_model
            
        except Exception as e:
            logger.error(f"Failed to get next model: {e}")
            # Ensure we don't leave a partially loaded model
            if self.current_model is not None:
                try:
                    self.current_model.unload()
                except Exception as unload_error:
                    logger.error(f"Failed to unload model after error: {unload_error}")
                self.current_model = None
            raise RuntimeError(f"Model rotation failed: {e}") from e
    
    def get_current_model(self) -> Optional[BaseVisionModel]:
        """
        Get the currently loaded model without rotation.
        
        Returns:
            Currently loaded model, or None if no model is loaded
        """
        return self.current_model
    
    def unload_current(self) -> None:
        """
        Unload the currently loaded model.
        
        This is useful for explicitly freeing memory when no more
        models are needed.
        
        Raises:
            RuntimeError: If unloading fails
        """
        if self.current_model is not None:
            try:
                logger.info(
                    f"Explicitly unloading current model: "
                    f"{self.current_model.__class__.__name__}"
                )
                self.current_model.unload()
                self.current_model = None
            except Exception as e:
                logger.error(f"Failed to unload current model: {e}")
                raise RuntimeError(f"Model unload failed: {e}") from e
        else:
            logger.info("No model currently loaded")
    
    def reset(self) -> None:
        """
        Reset rotation state to beginning.
        
        Unloads any currently loaded model and resets the rotation index
        to the start of the model list.
        
        Raises:
            RuntimeError: If unloading fails
        """
        try:
            logger.info("Resetting ModelRotator")
            
            # Unload current model
            self.unload_current()
            
            # Reset index
            self.current_index = -1
            
            logger.info("ModelRotator reset successfully")
            
        except Exception as e:
            logger.error(f"Failed to reset ModelRotator: {e}")
            raise RuntimeError(f"ModelRotator reset failed: {e}") from e
    
    def get_loaded_model_name(self) -> Optional[str]:
        """
        Get the name of the currently loaded model.
        
        Returns:
            Name of the currently loaded model, or None if no model is loaded
        """
        if self.current_model is not None:
            return self.current_model.__class__.__name__
        return None
    
    def get_rotation_progress(self) -> Dict[str, Any]:
        """
        Get information about rotation progress.
        
        Returns:
            Dictionary containing:
                - current_index: Index of currently loaded model (or None)
                - total_models: Total number of models in rotation
                - current_model_name: Name of currently loaded model (or None)
                - models_remaining: Number of models left in current cycle
        """
        return {
            "current_index": self.current_index if self.current_index >= 0 else None,
            "total_models": len(self.models),
            "current_model_name": self.get_loaded_model_name(),
            "models_remaining": (
                len(self.models) - self.current_index - 1
                if self.current_index >= 0
                else len(self.models)
            )
        }
