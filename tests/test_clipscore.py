"""
Unit tests for CLIPScore prompt-image alignment scorer.

Tests validate model loading, comparison scoring with prompts,
error handling, and sigmoid amplification functionality.
"""

import pytest
import tempfile
import os
import math
from unittest.mock import Mock, patch, MagicMock
import sys

from PIL import Image

# Mock torch before importing CLIPScore if torch is not available
try:
    import torch
except ImportError:
    torch = MagicMock()
    sys.modules['torch'] = torch
    sys.modules['torch.cuda'] = MagicMock()
    sys.modules['torch.backends'] = MagicMock()
    sys.modules['torch.backends.mps'] = MagicMock()

from src.scoring.clipscore import CLIPScore


# Fixtures for creating temporary test images
@pytest.fixture
def temp_test_image_a():
    """Create a temporary test image A."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img = Image.new('RGB', (224, 224), color='red')
        img.save(f.name, 'PNG')
        image_path = f.name
    yield img
    # Cleanup
    if os.path.exists(image_path):
        os.unlink(image_path)


@pytest.fixture
def temp_test_image_b():
    """Create a temporary test image B."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img = Image.new('RGB', (224, 224), color='blue')
        img.save(f.name, 'PNG')
        image_path = f.name
    yield img
    if os.path.exists(image_path):
        os.unlink(image_path)


@pytest.fixture
def mock_clip_adapter():
    """Mock CLIPAdapter for testing."""
    mock_adapter = MagicMock()
    
    # Mock encode_image to return normalized embeddings
    def mock_encode_image(image):
        # Return a mock normalized embedding
        return torch.randn(512) / torch.randn(512).norm()
    
    mock_adapter.encode_image = mock_encode_image
    mock_adapter.model = MagicMock()
    mock_adapter.device = "cpu"
    mock_adapter.model_name = "ViT-L-14"
    
    # Mock text encoding
    def mock_encode_text(tokens):
        return torch.randn(1, 512)
    
    mock_adapter.model.encode_text = mock_encode_text
    
    return mock_adapter


class TestCLIPScore:
    """Tests for CLIPScore scorer functionality."""
    
    def test_initialization_auto_device(self):
        """Test CLIPScore initialization with auto device detection."""
        scorer = CLIPScore(device="auto")
        
        assert scorer.clip_model is None
        assert scorer.device in ["cuda", "mps", "cpu"]
        assert scorer._owns_model is True
    
    def test_initialization_with_injected_model(self):
        """Test CLIPScore initialization with dependency-injected model."""
        mock_clip = MagicMock()
        scorer = CLIPScore(clip_model=mock_clip, device="cpu")
        
        assert scorer.clip_model is mock_clip
        assert scorer.device == "cpu"
        assert scorer._owns_model is False
    
    def test_detect_device_cuda(self):
        """Test device detection prefers CUDA when available."""
        with patch('torch.cuda.is_available', return_value=True):
            device = CLIPScore._detect_device()
            assert device == "cuda"
    
    def test_detect_device_mps(self):
        """Test device detection uses MPS when CUDA unavailable."""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=True):
            device = CLIPScore._detect_device()
            assert device == "mps"
    
    def test_detect_device_cpu_fallback(self):
        """Test device detection falls back to CPU."""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            device = CLIPScore._detect_device()
            assert device == "cpu"
    
    def test_sigmoid_amplification(self):
        """Test sigmoid amplification with typical CLIP similarity differences."""
        # Test that ×20 amplification provides good spread
        assert abs(CLIPScore._sigmoid(0.01) - 0.5498) < 0.001
        assert abs(CLIPScore._sigmoid(0.02) - 0.5987) < 0.001
        assert abs(CLIPScore._sigmoid(0.03) - 0.6457) < 0.001
        assert abs(CLIPScore._sigmoid(0.05) - 0.7311) < 0.001
        
        # Test symmetry
        assert abs(CLIPScore._sigmoid(-0.02) - (1.0 - CLIPScore._sigmoid(0.02))) < 0.001
    
    def test_sigmoid_zero_diff(self):
        """Test sigmoid returns 0.5 for zero difference."""
        assert CLIPScore._sigmoid(0.0) == 0.5
    
    def test_compare_requires_model(self, temp_test_image_a, temp_test_image_b):
        """Test compare raises error when model not loaded."""
        scorer = CLIPScore(device="cpu")
        
        with pytest.raises(RuntimeError, match="model not loaded"):
            scorer.compare(temp_test_image_a, temp_test_image_b, "prompt a", "prompt b")
    
    def test_compare_requires_prompts(self, temp_test_image_a, temp_test_image_b):
        """Test compare raises error when prompts not provided."""
        scorer = CLIPScore(clip_model=MagicMock(), device="cpu")
        
        with pytest.raises(ValueError, match="requires prompts"):
            scorer.compare(temp_test_image_a, temp_test_image_b)
    
    def test_compare_with_mock_model(self, temp_test_image_a, temp_test_image_b, mock_clip_adapter):
        """Test compare with mocked CLIP adapter."""
        scorer = CLIPScore(clip_model=mock_clip_adapter, device="cpu")
        
        # Mock text tokenizer
        with patch('open_clip.get_tokenizer') as mock_tokenizer:
            mock_tokens = MagicMock()
            mock_tokens.to.return_value = mock_tokens
            mock_tokenizer.return_value = lambda x: mock_tokens
            
            score = scorer.compare(
                temp_test_image_a,
                temp_test_image_b,
                prompt_a="a red image",
                prompt_b="a blue image"
            )
            
            # Score should be a float in [0, 1]
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
    
    def test_batch_compare_requires_prompts(self, temp_test_image_a, temp_test_image_b):
        """Test batch compare requires prompts list."""
        scorer = CLIPScore(clip_model=MagicMock(), device="cpu")
        pairs = [(temp_test_image_a, temp_test_image_b)]
        
        with pytest.raises(ValueError, match="requires prompts"):
            scorer.batch_compare(pairs)
    
    def test_batch_compare_prompts_length_mismatch(self, temp_test_image_a, temp_test_image_b):
        """Test batch compare validates prompts list length."""
        scorer = CLIPScore(clip_model=MagicMock(), device="cpu")
        pairs = [(temp_test_image_a, temp_test_image_b)]
        prompts = []  # Empty list, length mismatch
        
        with pytest.raises(ValueError, match="same length"):
            scorer.batch_compare(pairs, prompts)
    
    def test_load_with_injected_model_skips_loading(self):
        """Test load() doesn't reload when model is injected."""
        mock_clip = MagicMock()
        scorer = CLIPScore(clip_model=mock_clip, device="cpu")
        
        # Should not raise, should not try to create new adapter
        scorer.load("openai/clip-vit-large-patch14")
        
        # Model should still be the injected one
        assert scorer.clip_model is mock_clip
    
    @patch('src.scoring.clipscore.CLIPAdapter')
    def test_load_creates_adapter_when_none(self, mock_adapter_class):
        """Test load() creates CLIPAdapter when none provided."""
        mock_adapter_instance = MagicMock()
        mock_adapter_class.return_value = mock_adapter_instance
        
        scorer = CLIPScore(device="cpu")
        scorer.load("openai/clip-vit-large-patch14")
        
        # Should have created adapter
        mock_adapter_class.assert_called_once()
        mock_adapter_instance.load.assert_called_once()
        assert scorer.clip_model is mock_adapter_instance
        assert scorer._owns_model is True
    
    def test_unload_owned_model(self):
        """Test unload releases owned model."""
        mock_clip = MagicMock()
        scorer = CLIPScore(clip_model=mock_clip, device="cpu")
        scorer._owns_model = True
        
        scorer.unload()
        
        mock_clip.unload.assert_called_once()
        assert scorer.clip_model is None
    
    def test_unload_shared_model_does_not_unload(self):
        """Test unload doesn't release shared/injected model."""
        mock_clip = MagicMock()
        scorer = CLIPScore(clip_model=mock_clip, device="cpu")
        scorer._owns_model = False
        
        scorer.unload()
        
        # Should not unload the shared model
        mock_clip.unload.assert_not_called()
        # Model reference should still exist
        assert scorer.clip_model is mock_clip
    
    def test_compute_similarity_typical_range(self, temp_test_image_a, mock_clip_adapter):
        """Test _compute_similarity returns values in typical CLIP range."""
        scorer = CLIPScore(clip_model=mock_clip_adapter, device="cpu")
        
        with patch('open_clip.get_tokenizer') as mock_tokenizer:
            mock_tokens = MagicMock()
            mock_tokens.to.return_value = mock_tokens
            mock_tokenizer.return_value = lambda x: mock_tokens
            
            similarity = scorer._compute_similarity(temp_test_image_a, "a red image")
            
            # CLIP similarities for matching prompts are typically 0.20-0.35
            # With random mocks, just verify it's a reasonable float
            assert isinstance(similarity, float)
            assert -1.0 <= similarity <= 1.0  # Cosine similarity range


class TestCLIPScoreIntegration:
    """Integration tests for CLIPScore (requires mocking but tests full flow)."""
    
    def test_compare_higher_similarity_wins(self, temp_test_image_a, temp_test_image_b):
        """Test that image with higher prompt similarity gets higher score."""
        # Create scorer with mock that returns predictable similarities
        mock_clip = MagicMock()
        
        call_count = [0]
        
        def mock_encode_image(image):
            # First call (image_a): higher similarity
            # Second call (image_b): lower similarity
            call_count[0] += 1
            if call_count[0] == 1:
                return torch.tensor([1.0, 0.0])  # Will have high similarity with prompt
            else:
                return torch.tensor([0.5, 0.5])  # Will have lower similarity
        
        mock_clip.encode_image = mock_encode_image
        mock_clip.model = MagicMock()
        mock_clip.device = "cpu"
        mock_clip.model_name = "ViT-L-14"
        
        def mock_encode_text(tokens):
            return torch.tensor([[1.0, 0.0]])  # Matches first image better
        
        mock_clip.model.encode_text = mock_encode_text
        
        scorer = CLIPScore(clip_model=mock_clip, device="cpu")
        
        with patch('open_clip.get_tokenizer') as mock_tokenizer:
            mock_tokens = MagicMock()
            mock_tokens.to.return_value = mock_tokens
            mock_tokenizer.return_value = lambda x: mock_tokens
            
            score = scorer.compare(
                temp_test_image_a,
                temp_test_image_b,
                prompt_a="test prompt",
                prompt_b="test prompt"
            )
            
            # Image A should win (score > 0.5)
            assert score > 0.5
