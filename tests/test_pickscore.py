"""
Unit tests for PickScore aesthetic scorer.

Tests validate model loading, comparison scoring, batch processing,
error handling, and score normalization functionality.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys

from PIL import Image

# Mock torch before importing PickScore if torch is not available
try:
    import torch
except ImportError:
    torch = MagicMock()
    sys.modules['torch'] = torch
    sys.modules['torch.cuda'] = MagicMock()
    sys.modules['torch.backends'] = MagicMock()
    sys.modules['torch.backends.mps'] = MagicMock()

from src.scoring.pickscore import PickScore


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
def mock_pickscore_model():
    """Mock PickScore model and processor."""
    mock_model = MagicMock()
    mock_processor = MagicMock()
    
    # Mock processor output
    mock_inputs = {
        'pixel_values': torch.randn(1, 3, 224, 224)
    }
    mock_processor.return_value = mock_inputs
    
    # Mock model output
    mock_features = torch.randn(1, 512)
    mock_model.get_image_features.return_value = mock_features
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    
    return mock_model, mock_processor


class TestPickScore:
    """Tests for PickScore scorer functionality."""
    
    def test_initialization_auto_device(self):
        """Test PickScore initialization with auto device detection."""
        scorer = PickScore(device="auto")
        
        assert scorer.model is None
        assert scorer.processor is None
        assert scorer.device in ["cuda", "mps", "cpu"]
    
    def test_initialization_explicit_device(self):
        """Test PickScore initialization with explicit device."""
        scorer = PickScore(device="cpu")
        
        assert scorer.model is None
        assert scorer.processor is None
        assert scorer.device == "cpu"
    
    def test_detect_device_cuda(self):
        """Test device detection prefers CUDA when available."""
        with patch('torch.cuda.is_available', return_value=True):
            device = PickScore._detect_device()
            assert device == "cuda"
    
    def test_detect_device_mps(self):
        """Test device detection uses MPS when CUDA unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                device = PickScore._detect_device()
                assert device == "mps"
    
    def test_detect_device_cpu_fallback(self):
        """Test device detection falls back to CPU."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                device = PickScore._detect_device()
                assert device == "cpu"
    
    @patch('src.scoring.pickscore.AutoModel')
    @patch('src.scoring.pickscore.AutoProcessor')
    def test_load_model_success(self, mock_processor_cls, mock_model_cls):
        """Test successful model loading."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        scorer = PickScore(device="cpu")
        scorer.load()
        
        assert scorer.model is not None
        assert scorer.processor is not None
        mock_model_cls.from_pretrained.assert_called_once()
        mock_processor_cls.from_pretrained.assert_called_once()
        mock_model.eval.assert_called_once()
    
    @patch('src.scoring.pickscore.AutoModel')
    @patch('src.scoring.pickscore.AutoProcessor')
    def test_load_model_custom_id(self, mock_processor_cls, mock_model_cls):
        """Test loading with custom model ID."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        custom_id = "custom/pickscore-model"
        scorer = PickScore(device="cpu")
        scorer.load(model_id=custom_id)
        
        mock_model_cls.from_pretrained.assert_called_with(custom_id)
        mock_processor_cls.from_pretrained.assert_called_with(custom_id)
    
    @patch('src.scoring.pickscore.AutoModel')
    @patch('src.scoring.pickscore.AutoProcessor')
    def test_load_model_failure(self, mock_processor_cls, mock_model_cls):
        """Test model loading failure raises RuntimeError."""
        mock_model_cls.from_pretrained.side_effect = Exception("Model not found")
        
        scorer = PickScore(device="cpu")
        
        with pytest.raises(RuntimeError, match="PickScore loading failed"):
            scorer.load()
    
    def test_compare_without_loading(self, temp_test_image_a, temp_test_image_b):
        """Test that compare() raises error if model not loaded."""
        scorer = PickScore(device="cpu")
        
        with pytest.raises(RuntimeError, match="model not loaded"):
            scorer.compare(temp_test_image_a, temp_test_image_b)
    
    @patch('src.scoring.pickscore.AutoModel')
    @patch('src.scoring.pickscore.AutoProcessor')
    def test_compare_returns_valid_range(
        self,
        mock_processor_cls,
        mock_model_cls,
        temp_test_image_a,
        temp_test_image_b
    ):
        """Test that compare() returns score in [0.0, 1.0]."""
        # Setup mocks
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        # Mock processor to return proper structure
        mock_inputs = {'pixel_values': torch.randn(1, 3, 224, 224)}
        mock_processor.return_value = MagicMock(to=lambda x: mock_inputs)
        
        # Mock model to return features with different norms
        feature_a = torch.randn(1, 512)
        feature_b = torch.randn(1, 512)
        mock_model.get_image_features.side_effect = [feature_a, feature_b]
        
        scorer = PickScore(device="cpu")
        scorer.load()
        
        score = scorer.compare(temp_test_image_a, temp_test_image_b)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    @patch('src.scoring.pickscore.AutoModel')
    @patch('src.scoring.pickscore.AutoProcessor')
    def test_compare_image_a_preference(
        self,
        mock_processor_cls,
        mock_model_cls,
        temp_test_image_a,
        temp_test_image_b
    ):
        """Test that higher score for image_a produces score > 0.5."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        # Mock processor
        mock_inputs = {'pixel_values': torch.randn(1, 3, 224, 224)}
        mock_processor.return_value = MagicMock(to=lambda x: mock_inputs)
        
        # Image A has higher feature norm (better quality)
        feature_a = torch.ones(1, 512) * 2.0  # norm = 32
        feature_b = torch.ones(1, 512) * 1.0  # norm = 16
        mock_model.get_image_features.side_effect = [feature_a, feature_b]
        
        scorer = PickScore(device="cpu")
        scorer.load()
        
        score = scorer.compare(temp_test_image_a, temp_test_image_b)
        
        # Score should favor image A
        assert score > 0.5
    
    @patch('src.scoring.pickscore.AutoModel')
    @patch('src.scoring.pickscore.AutoProcessor')
    def test_compare_image_b_preference(
        self,
        mock_processor_cls,
        mock_model_cls,
        temp_test_image_a,
        temp_test_image_b
    ):
        """Test that higher score for image_b produces score < 0.5."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        # Mock processor
        mock_inputs = {'pixel_values': torch.randn(1, 3, 224, 224)}
        mock_processor.return_value = MagicMock(to=lambda x: mock_inputs)
        
        # Image B has higher feature norm (better quality)
        feature_a = torch.ones(1, 512) * 1.0  # norm = 16
        feature_b = torch.ones(1, 512) * 2.0  # norm = 32
        mock_model.get_image_features.side_effect = [feature_a, feature_b]
        
        scorer = PickScore(device="cpu")
        scorer.load()
        
        score = scorer.compare(temp_test_image_a, temp_test_image_b)
        
        # Score should favor image B
        assert score < 0.5
    
    @patch('src.scoring.pickscore.AutoModel')
    @patch('src.scoring.pickscore.AutoProcessor')
    def test_compare_equal_scores(
        self,
        mock_processor_cls,
        mock_model_cls,
        temp_test_image_a,
        temp_test_image_b
    ):
        """Test that equal scores produce score near 0.5."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        # Mock processor
        mock_inputs = {'pixel_values': torch.randn(1, 3, 224, 224)}
        mock_processor.return_value = MagicMock(to=lambda x: mock_inputs)
        
        # Both images have equal feature norms
        feature = torch.ones(1, 512) * 1.0
        mock_model.get_image_features.side_effect = [feature.clone(), feature.clone()]
        
        scorer = PickScore(device="cpu")
        scorer.load()
        
        score = scorer.compare(temp_test_image_a, temp_test_image_b)
        
        # Score should be near 0.5 (neutral)
        assert abs(score - 0.5) < 0.01
    
    def test_batch_compare_without_loading(self, temp_test_image_a, temp_test_image_b):
        """Test that batch_compare() raises error if model not loaded."""
        scorer = PickScore(device="cpu")
        pairs = [(temp_test_image_a, temp_test_image_b)]
        
        with pytest.raises(RuntimeError, match="model not loaded"):
            scorer.batch_compare(pairs)
    
    @patch('src.scoring.pickscore.AutoModel')
    @patch('src.scoring.pickscore.AutoProcessor')
    def test_batch_compare_multiple_pairs(
        self,
        mock_processor_cls,
        mock_model_cls,
        temp_test_image_a,
        temp_test_image_b
    ):
        """Test batch comparison with multiple pairs."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        # Mock processor
        mock_inputs = {'pixel_values': torch.randn(1, 3, 224, 224)}
        mock_processor.return_value = MagicMock(to=lambda x: mock_inputs)
        
        # Mock features
        features = [torch.randn(1, 512) for _ in range(6)]
        mock_model.get_image_features.side_effect = features
        
        scorer = PickScore(device="cpu")
        scorer.load()
        
        pairs = [
            (temp_test_image_a, temp_test_image_b),
            (temp_test_image_b, temp_test_image_a),
            (temp_test_image_a, temp_test_image_a),
        ]
        
        scores = scorer.batch_compare(pairs)
        
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
        assert all(0.0 <= s <= 1.0 for s in scores)
    
    @patch('src.scoring.pickscore.AutoModel')
    @patch('src.scoring.pickscore.AutoProcessor')
    def test_batch_compare_empty_list(self, mock_processor_cls, mock_model_cls):
        """Test batch comparison with empty list."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        scorer = PickScore(device="cpu")
        scorer.load()
        
        scores = scorer.batch_compare([])
        
        assert scores == []
    
    def test_sigmoid_normalization(self):
        """Test sigmoid function produces correct normalization."""
        # Test zero difference
        assert abs(PickScore._sigmoid(0.0) - 0.5) < 1e-6
        
        # Test positive difference
        assert PickScore._sigmoid(1.0) > 0.5
        assert PickScore._sigmoid(10.0) > 0.99
        
        # Test negative difference
        assert PickScore._sigmoid(-1.0) < 0.5
        assert PickScore._sigmoid(-10.0) < 0.01
        
        # Test bounds (sigmoid(100.0) saturates to 1.0 in float64)
        assert 0.0 < PickScore._sigmoid(-100.0) < 1.0
        assert 0.0 < PickScore._sigmoid(100.0) <= 1.0
    
    @patch('src.scoring.pickscore.AutoModel')
    @patch('src.scoring.pickscore.AutoProcessor')
    def test_unload_clears_model(self, mock_processor_cls, mock_model_cls):
        """Test that unload() clears model and processor."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        scorer = PickScore(device="cpu")
        scorer.load()
        
        assert scorer.model is not None
        assert scorer.processor is not None
        
        scorer.unload()
        
        assert scorer.model is None
        assert scorer.processor is None
    
    @patch('src.scoring.pickscore.AutoModel')
    @patch('src.scoring.pickscore.AutoProcessor')
    @patch('torch.cuda.empty_cache')
    def test_unload_clears_cuda_cache(
        self,
        mock_cuda_cache,
        mock_processor_cls,
        mock_model_cls
    ):
        """Test that unload() clears CUDA cache when on CUDA device."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        scorer = PickScore(device="cuda")
        scorer.load()
        scorer.unload()
        
        mock_cuda_cache.assert_called_once()
    
    @patch('src.scoring.pickscore.AutoModel')
    @patch('src.scoring.pickscore.AutoProcessor')
    @patch('torch.mps.empty_cache')
    def test_unload_clears_mps_cache(
        self,
        mock_mps_cache,
        mock_processor_cls,
        mock_model_cls
    ):
        """Test that unload() clears MPS cache when on MPS device."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        scorer = PickScore(device="mps")
        scorer.load()
        scorer.unload()
        
        mock_mps_cache.assert_called_once()
    
    def test_unload_without_loading(self):
        """Test that unload() works even if model never loaded."""
        scorer = PickScore(device="cpu")
        
        # Should not raise error
        scorer.unload()
        
        assert scorer.model is None
        assert scorer.processor is None
    
    @patch('src.scoring.pickscore.AutoModel')
    @patch('src.scoring.pickscore.AutoProcessor')
    def test_compare_inference_failure(
        self,
        mock_processor_cls,
        mock_model_cls,
        temp_test_image_a,
        temp_test_image_b
    ):
        """Test that compare() raises RuntimeError on inference failure."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        # Mock processor to raise exception
        mock_processor.side_effect = Exception("Inference failed")
        
        scorer = PickScore(device="cpu")
        scorer.load()
        
        with pytest.raises(RuntimeError, match="comparison failed"):
            scorer.compare(temp_test_image_a, temp_test_image_b)
    
    def test_get_name(self):
        """Test that scorer returns correct name."""
        scorer = PickScore(device="cpu")
        assert scorer.get_name() == "PickScore"
