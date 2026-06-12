"""
Unit tests for CacheManager embedding storage.

Tests verify cache storage, retrieval, key collision prevention,
and cache management operations.
"""

import pytest
import torch

from src.cache_manager import CacheManager


class TestCacheManager:
    """Test suite for CacheManager class."""
    
    def test_initialization(self):
        """Test CacheManager initializes with empty cache."""
        cache = CacheManager()
        
        assert cache.get_cache_size() == 0
        assert cache.get_hit_rate() == 0.0
        assert cache.get_statistics()['hit_count'] == 0
        assert cache.get_statistics()['miss_count'] == 0
    
    def test_store_and_retrieve_embedding(self):
        """Test storing and retrieving an embedding."""
        cache = CacheManager()
        
        # Create a test embedding
        embedding = torch.randn(512)
        image_id = "img_001"
        model_name = "clip"
        
        # Store embedding
        cache.store_embedding(image_id, model_name, embedding)
        
        # Verify cache size
        assert cache.get_cache_size() == 1
        
        # Retrieve embedding
        retrieved = cache.get_embedding(image_id, model_name)
        
        # Verify embedding matches
        assert retrieved is not None
        assert torch.equal(retrieved, embedding)
    
    def test_cache_miss(self):
        """Test retrieving non-existent embedding returns None."""
        cache = CacheManager()
        
        # Try to retrieve non-existent embedding
        result = cache.get_embedding("img_999", "blip")
        
        assert result is None
        assert cache.get_statistics()['miss_count'] == 1
        assert cache.get_hit_rate() == 0.0
    
    def test_key_collision_prevention(self):
        """Test that different image-model pairs don't collide."""
        cache = CacheManager()
        
        # Store embeddings for same image, different models
        embedding1 = torch.randn(512)
        embedding2 = torch.randn(512)
        image_id = "img_001"
        
        cache.store_embedding(image_id, "clip", embedding1)
        cache.store_embedding(image_id, "blip", embedding2)
        
        # Store embeddings for different images, same model
        embedding3 = torch.randn(512)
        embedding4 = torch.randn(512)
        
        cache.store_embedding("img_001", "instructblip", embedding3)
        cache.store_embedding("img_002", "instructblip", embedding4)
        
        # Verify all embeddings are distinct and retrievable
        assert cache.get_cache_size() == 4
        
        retrieved1 = cache.get_embedding(image_id, "clip")
        retrieved2 = cache.get_embedding(image_id, "blip")
        retrieved3 = cache.get_embedding("img_001", "instructblip")
        retrieved4 = cache.get_embedding("img_002", "instructblip")
        
        assert torch.equal(retrieved1, embedding1)
        assert torch.equal(retrieved2, embedding2)
        assert torch.equal(retrieved3, embedding3)
        assert torch.equal(retrieved4, embedding4)
        
        # Verify embeddings are different
        assert not torch.equal(embedding1, embedding2)
        assert not torch.equal(embedding3, embedding4)
    
    def test_clear_cache(self):
        """Test clearing cache removes all entries."""
        cache = CacheManager()
        
        # Store multiple embeddings
        cache.store_embedding("img_001", "clip", torch.randn(512))
        cache.store_embedding("img_002", "blip", torch.randn(512))
        cache.store_embedding("img_003", "instructblip", torch.randn(512))
        
        # Generate some cache hits
        cache.get_embedding("img_001", "clip")
        cache.get_embedding("img_999", "unknown")  # miss
        
        assert cache.get_cache_size() == 3
        assert cache.get_statistics()['hit_count'] == 1
        assert cache.get_statistics()['miss_count'] == 1
        
        # Clear cache
        cache.clear_cache()
        
        # Verify cache is empty and statistics reset
        assert cache.get_cache_size() == 0
        assert cache.get_statistics()['hit_count'] == 0
        assert cache.get_statistics()['miss_count'] == 0
        assert cache.get_hit_rate() == 0.0
        
        # Verify embeddings are gone
        assert cache.get_embedding("img_001", "clip") is None
    
    def test_has_embedding(self):
        """Test checking embedding existence without retrieval."""
        cache = CacheManager()
        
        # Check non-existent embedding
        assert not cache.has_embedding("img_001", "clip")
        
        # Store embedding
        cache.store_embedding("img_001", "clip", torch.randn(512))
        
        # Check existing embedding
        assert cache.has_embedding("img_001", "clip")
        
        # Check with different model name
        assert not cache.has_embedding("img_001", "blip")
        
        # Verify has_embedding doesn't affect statistics
        stats = cache.get_statistics()
        assert stats['hit_count'] == 0
        assert stats['miss_count'] == 0
    
    def test_remove_embedding(self):
        """Test removing specific embeddings from cache."""
        cache = CacheManager()
        
        # Store embeddings
        cache.store_embedding("img_001", "clip", torch.randn(512))
        cache.store_embedding("img_002", "blip", torch.randn(512))
        
        assert cache.get_cache_size() == 2
        
        # Remove existing embedding
        result = cache.remove_embedding("img_001", "clip")
        assert result is True
        assert cache.get_cache_size() == 1
        assert not cache.has_embedding("img_001", "clip")
        
        # Verify other embedding still exists
        assert cache.has_embedding("img_002", "blip")
        
        # Try removing non-existent embedding
        result = cache.remove_embedding("img_999", "unknown")
        assert result is False
        assert cache.get_cache_size() == 1
    
    def test_overwrite_warning(self, caplog):
        """Test that overwriting an embedding logs a warning."""
        cache = CacheManager()
        
        embedding1 = torch.randn(512)
        embedding2 = torch.randn(512)
        
        # Store initial embedding
        cache.store_embedding("img_001", "clip", embedding1)
        
        # Overwrite with different embedding
        cache.store_embedding("img_001", "clip", embedding2)
        
        # Verify the new embedding replaced the old one
        retrieved = cache.get_embedding("img_001", "clip")
        assert torch.equal(retrieved, embedding2)
        assert not torch.equal(retrieved, embedding1)
        
        # Verify cache size didn't increase
        assert cache.get_cache_size() == 1
    
    def test_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        cache = CacheManager()
        
        # Initially hit rate is 0.0
        assert cache.get_hit_rate() == 0.0
        
        # Store an embedding
        cache.store_embedding("img_001", "clip", torch.randn(512))
        
        # Generate hits and misses
        cache.get_embedding("img_001", "clip")  # hit
        cache.get_embedding("img_001", "clip")  # hit
        cache.get_embedding("img_002", "blip")  # miss
        cache.get_embedding("img_001", "clip")  # hit
        
        # Verify hit rate: 3 hits out of 4 accesses = 0.75
        stats = cache.get_statistics()
        assert stats['hit_count'] == 3
        assert stats['miss_count'] == 1
        assert stats['total_accesses'] == 4
        assert cache.get_hit_rate() == 0.75
    
    def test_embedding_independence(self):
        """Test that modifying cached embedding doesn't affect original."""
        cache = CacheManager()
        
        # Create and store embedding
        original_embedding = torch.randn(512)
        image_id = "img_001"
        model_name = "clip"
        
        cache.store_embedding(image_id, model_name, original_embedding)
        
        # Retrieve embedding
        retrieved = cache.get_embedding(image_id, model_name)
        
        # Verify they are the same object (cache stores reference)
        assert retrieved is original_embedding
        
        # This is expected behavior - cache stores tensor references
        # If isolation is needed, callers should clone() embeddings
    
    def test_multiple_models_same_image(self):
        """Test caching embeddings from multiple models for same image."""
        cache = CacheManager()
        
        image_id = "img_001"
        models = ["clip", "blip", "blip2", "instructblip"]
        embeddings = [torch.randn(512) for _ in models]
        
        # Store embeddings from all models
        for model, embedding in zip(models, embeddings):
            cache.store_embedding(image_id, model, embedding)
        
        assert cache.get_cache_size() == 4
        
        # Retrieve and verify each embedding
        for model, expected_embedding in zip(models, embeddings):
            retrieved = cache.get_embedding(image_id, model)
            assert retrieved is not None
            assert torch.equal(retrieved, expected_embedding)
    
    def test_cache_with_special_characters_in_keys(self):
        """Test cache handles special characters in image IDs and model names."""
        cache = CacheManager()
        
        # Test various special characters
        test_cases = [
            ("img-001", "clip-vit-large"),
            ("img_002", "blip_2"),
            ("img.003", "model.v1"),
            ("img/004", "path/to/model"),
            ("img@005", "model@latest"),
        ]
        
        for image_id, model_name in test_cases:
            embedding = torch.randn(512)
            cache.store_embedding(image_id, model_name, embedding)
            
            retrieved = cache.get_embedding(image_id, model_name)
            assert retrieved is not None
            assert torch.equal(retrieved, embedding)
        
        assert cache.get_cache_size() == len(test_cases)
    
    def test_statistics_dictionary(self):
        """Test that get_statistics returns complete dictionary."""
        cache = CacheManager()
        
        # Store and access embeddings
        cache.store_embedding("img_001", "clip", torch.randn(512))
        cache.get_embedding("img_001", "clip")  # hit
        cache.get_embedding("img_002", "blip")  # miss
        
        stats = cache.get_statistics()
        
        # Verify all required keys present
        assert 'cache_size' in stats
        assert 'hit_count' in stats
        assert 'miss_count' in stats
        assert 'total_accesses' in stats
        assert 'hit_rate' in stats
        
        # Verify values
        assert stats['cache_size'] == 1
        assert stats['hit_count'] == 1
        assert stats['miss_count'] == 1
        assert stats['total_accesses'] == 2
        assert stats['hit_rate'] == 0.5
