"""
Embedding cache management for IMAGE-LLP-VISION system.

This module handles caching of image embeddings to avoid redundant model inference.
Embeddings are indexed by (image_id, model_name) tuples to prevent key collisions.
"""

from typing import Dict, Optional, Tuple
import logging

import torch

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages embedding cache for image-model pairs.
    
    Stores embeddings indexed by (image_id, model_name) tuples to prevent
    key collisions between different models processing the same image.
    Provides methods for storing, retrieving, and clearing cached embeddings.
    """
    
    def __init__(self):
        """Initialize CacheManager with empty cache."""
        # Cache structure: {(image_id, model_name): torch.Tensor}
        self._cache: Dict[Tuple[str, str], torch.Tensor] = {}
        self._hit_count: int = 0
        self._miss_count: int = 0
        logger.debug("Initialized CacheManager")
    
    def get_embedding(self, image_id: str, model_name: str) -> Optional[torch.Tensor]:
        """
        Retrieve cached embedding for an image-model pair.
        
        Args:
            image_id: Unique identifier for the image
            model_name: Name of the model that generated the embedding
            
        Returns:
            Cached embedding tensor if available, None otherwise
        """
        cache_key = (image_id, model_name)
        
        if cache_key in self._cache:
            self._hit_count += 1
            logger.debug(
                f"Cache hit for image_id='{image_id}', model_name='{model_name}' "
                f"(hit_rate: {self.get_hit_rate():.2%})"
            )
            return self._cache[cache_key]
        else:
            self._miss_count += 1
            logger.debug(
                f"Cache miss for image_id='{image_id}', model_name='{model_name}' "
                f"(hit_rate: {self.get_hit_rate():.2%})"
            )
            return None
    
    def store_embedding(
        self,
        image_id: str,
        model_name: str,
        embedding: torch.Tensor
    ) -> None:
        """
        Store an embedding in the cache.
        
        Args:
            image_id: Unique identifier for the image
            model_name: Name of the model that generated the embedding
            embedding: Embedding tensor to cache
        """
        cache_key = (image_id, model_name)
        
        # Check if we're overwriting an existing entry
        if cache_key in self._cache:
            logger.warning(
                f"Overwriting existing cache entry for image_id='{image_id}', "
                f"model_name='{model_name}'"
            )
        
        self._cache[cache_key] = embedding
        logger.debug(
            f"Stored embedding for image_id='{image_id}', model_name='{model_name}' "
            f"(cache_size: {len(self._cache)})"
        )
    
    def clear_cache(self) -> None:
        """
        Clear all cached embeddings.
        
        Also resets hit/miss statistics.
        """
        cache_size = len(self._cache)
        self._cache.clear()
        
        # Reset statistics
        old_hit_rate = self.get_hit_rate()
        self._hit_count = 0
        self._miss_count = 0
        
        logger.info(
            f"Cleared cache: removed {cache_size} entries "
            f"(final hit_rate: {old_hit_rate:.2%})"
        )
    
    def get_cache_size(self) -> int:
        """
        Get the number of cached embeddings.
        
        Returns:
            Number of entries in cache
        """
        return len(self._cache)
    
    def get_hit_rate(self) -> float:
        """
        Calculate cache hit rate.
        
        Returns:
            Hit rate as a float in [0.0, 1.0], or 0.0 if no accesses
        """
        total_accesses = self._hit_count + self._miss_count
        if total_accesses == 0:
            return 0.0
        return self._hit_count / total_accesses
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache_size, hit_count, miss_count, and hit_rate
        """
        return {
            'cache_size': len(self._cache),
            'hit_count': self._hit_count,
            'miss_count': self._miss_count,
            'total_accesses': self._hit_count + self._miss_count,
            'hit_rate': self.get_hit_rate()
        }
    
    def has_embedding(self, image_id: str, model_name: str) -> bool:
        """
        Check if an embedding exists in cache without retrieving it.
        
        Args:
            image_id: Unique identifier for the image
            model_name: Name of the model
            
        Returns:
            True if embedding is cached, False otherwise
        """
        cache_key = (image_id, model_name)
        return cache_key in self._cache
    
    def remove_embedding(self, image_id: str, model_name: str) -> bool:
        """
        Remove a specific embedding from cache.
        
        Args:
            image_id: Unique identifier for the image
            model_name: Name of the model
            
        Returns:
            True if embedding was removed, False if it didn't exist
        """
        cache_key = (image_id, model_name)
        
        if cache_key in self._cache:
            del self._cache[cache_key]
            logger.debug(
                f"Removed embedding for image_id='{image_id}', model_name='{model_name}' "
                f"(cache_size: {len(self._cache)})"
            )
            return True
        else:
            logger.debug(
                f"Cannot remove non-existent embedding for image_id='{image_id}', "
                f"model_name='{model_name}'"
            )
            return False
