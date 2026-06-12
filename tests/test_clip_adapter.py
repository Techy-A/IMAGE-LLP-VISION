"""
Unit tests for CLIPAdapter.

Tests cover:
- Model loading with different quantization settings
- Image encoding and embedding generation
- Memory management and unloading
- Device detection
- Error handling
"""

import pytest
import torch
from unittest.mock import patch, MagicMock, call
from PIL import Image

from src.models.clip_adapter import CLIPAdapter


class TestCLIPAdapterInitialization:
    """Test CLIPAdapter initialization."""
    
    def test_initialization(self):
        """Should initialize with None values."""
        adapter = CLIPAdapter()
        
        assert adapter.model is None
        assert adapter.preprocess is None
        assert adapter.device is None
        assert adapter.model_name is None
        assert adapter.pretrained_dataset is None


class TestCLIPAdapterLoading:
    """Test CLIP model loading functionality."""
    
    @patch('src.models.clip_adapter.open_clip.create_model_and_transforms')
    @patch.object(CLIPAdapter, 'detect_device')
    def test_load_with_fp16(self, mock_detect_device, mock_create_model):
        """Should load CLIP model with FP16 quantization."""
        mock_detect_device.return_value = "cuda"
        
        # Create mock model
        mock_model = MagicMock()
        mock_model.half.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_preprocess = MagicMock()
        mock_create_model.return_value = (mock_model, None, mock_preprocess)
        
        adapter = CLIPAdapter()
        config = {
            "hf_id": "openai/clip-vit-large-patch14",
            "quantization": "fp16"
        }
        adapter.load(config)
        
        # Verify model loaded
        assert adapter.model is not None
        assert adapter.preprocess is not None
        assert adapter.device == "cuda"
        assert adapter.model_name == "ViT-L-14"
        
        # Verify FP16 conversion
        mock_model.half.assert_called_once()
        mock_model.eval.assert_called_once()
    
    @patch('src.models.clip_adapter.open_clip.create_model_and_transforms')
    @patch.object(CLIPAdapter, 'detect_device')
    def test_load_with_fp32(self, mock_detect_device, mock_create_model):
        """Should load CLIP model with FP32 precision."""
        mock_detect_device.return_value = "cpu"
        
        mock_model = MagicMock()
        mock_model.float.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_preprocess = MagicMock()
        mock_create_model.return_value = (mock_model, None, mock_preprocess)
        
        adapter = CLIPAdapter()
        config = {
            "hf_id": "openai/clip-vit-base-patch32",
            "quantization": "fp32"
        }
        adapter.load(config)
        
        # Verify FP32 used
        mock_model.float.assert_called_once()
        mock_model.half.assert_not_called()
    
    @patch('src.models.clip_adapter.open_clip.create_model_and_transforms')
    @patch.object(CLIPAdapter, 'detect_device')
    def test_load_with_unsupported_quantization_falls_back_to_fp16(
        self, mock_detect_device, mock_create_model
    ):
        """Should fall back to FP16 for unsupported quantization modes."""
        mock_detect_device.return_value = "cuda"
        
        mock_model = MagicMock()
        mock_model.half.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_preprocess = MagicMock()
        mock_create_model.return_value = (mock_model, None, mock_preprocess)
        
        adapter = CLIPAdapter()
        config = {
            "hf_id": "openai/clip-vit-large-patch14",
            "quantization": "4bit"  # Not optimal for CLIP
        }
        adapter.load(config)
        
        # Should fall back to FP16
        mock_model.half.assert_called_once()
    
    @patch('src.models.clip_adapter.open_clip.create_model_and_transforms')
    @patch.object(CLIPAdapter, 'detect_device')
    def test_load_with_custom_pretrained_dataset(
        self, mock_detect_device, mock_create_model
    ):
        """Should support custom pretrained dataset."""
        mock_detect_device.return_value = "cuda"
        
        mock_model = MagicMock()
        mock_model.half.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_preprocess = MagicMock()
        mock_create_model.return_value = (mock_model, None, mock_preprocess)
        
        adapter = CLIPAdapter()
        config = {
            "hf_id": "openai/clip-vit-large-patch14",
            "quantization": "fp16",
            "pretrained": "laion2b_s32b_b82k"
        }
        adapter.load(config)
        
        # Verify custom pretrained dataset used
        assert adapter.pretrained_dataset == "laion2b_s32b_b82k"
        mock_create_model.assert_called_once_with(
            "ViT-L-14",
            pretrained="laion2b_s32b_b82k",
            device="cuda"
        )
    
    def test_load_without_hf_id_raises_error(self):
        """Should raise ValueError if hf_id is missing."""
        adapter = CLIPAdapter()
        config = {"quantization": "fp16"}
        
        with pytest.raises(ValueError, match="hf_id is required"):
            adapter.load(config)
    
    @patch('src.models.clip_adapter.open_clip.create_model_and_transforms')
    @patch.object(CLIPAdapter, 'detect_device')
    def test_load_failure_raises_runtime_error(
        self, mock_detect_device, mock_create_model
    ):
        """Should raise RuntimeError if model loading fails."""
        mock_detect_device.return_value = "cuda"
        mock_create_model.side_effect = Exception("Model not found")
        
        adapter = CLIPAdapter()
        config = {
            "hf_id": "invalid/model",
            "quantization": "fp16"
        }
        
        with pytest.raises(RuntimeError, match="CLIP model loading failed"):
            adapter.load(config)


class TestCLIPAdapterEncoding:
    """Test image encoding functionality."""
    
    @patch('src.models.clip_adapter.open_clip.create_model_and_transforms')
    @patch.object(CLIPAdapter, 'detect_device')
    def test_encode_image_returns_normalized_tensor(
        self, mock_detect_device, mock_create_model
    ):
        """Should encode image and return normalized embedding."""
        mock_detect_device.return_value = "cpu"
        
        # Setup mock model
        mock_model = MagicMock()
        mock_model.half.return_value = mock_model
        mock_model.float.return_value = mock_model  # keep same mock after .float() call
        mock_model.eval.return_value = mock_model
        
        # Mock encode_image to return a real tensor
        mock_embedding = torch.randn(1, 512)
        mock_model.encode_image.return_value = mock_embedding
        # parameters() must return an iterator (not a list) for next() to work
        fp32_param = torch.nn.Parameter(torch.randn(1, dtype=torch.float32))
        mock_model.parameters.side_effect = lambda: iter([fp32_param])
        
        mock_preprocess = MagicMock()
        mock_preprocess.return_value = MagicMock()  # avoid real .to(device) on CPU
        
        mock_create_model.return_value = (mock_model, None, mock_preprocess)
        
        # Load and encode
        adapter = CLIPAdapter()
        adapter.load({"hf_id": "openai/clip-vit-base-patch32", "quantization": "fp32"})
        
        # Create test image
        test_image = Image.new('RGB', (224, 224))
        result = adapter.encode_image(test_image)
        
        # Verify result
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 1  # Should be squeezed
        mock_preprocess.assert_called_once_with(test_image)
        mock_model.encode_image.assert_called_once()
    
    def test_encode_image_without_loaded_model_raises_error(self):
        """Should raise RuntimeError if model not loaded."""
        adapter = CLIPAdapter()
        test_image = Image.new('RGB', (224, 224))
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode_image(test_image)
    
    @patch('src.models.clip_adapter.open_clip.create_model_and_transforms')
    @patch.object(CLIPAdapter, 'detect_device')
    def test_encode_image_handles_fp16_conversion(
        self, mock_detect_device, mock_create_model
    ):
        """Should convert input to FP16 when model is in FP16."""
        mock_detect_device.return_value = "cuda"
        
        # Setup mock model with FP16
        mock_model = MagicMock()
        mock_model.half.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        # Mock parameters to return FP16 via iterator
        mock_param = torch.nn.Parameter(torch.randn(1, dtype=torch.float16))
        mock_model.parameters.side_effect = lambda: iter([mock_param])
        
        mock_embedding = torch.randn(1, 512, dtype=torch.float16)
        mock_model.encode_image.return_value = mock_embedding
        
        mock_preprocess = MagicMock()
        mock_preprocess.return_value = MagicMock()  # avoid real .to("cuda") on CPU
        
        mock_create_model.return_value = (mock_model, None, mock_preprocess)
        
        # Load and encode
        adapter = CLIPAdapter()
        adapter.load({"hf_id": "openai/clip-vit-large-patch14", "quantization": "fp16"})
        
        test_image = Image.new('RGB', (224, 224))
        result = adapter.encode_image(test_image)
        
        # Verify result is returned
        assert isinstance(result, torch.Tensor)


class TestCLIPAdapterUnloading:
    """Test model unloading and memory management."""
    
    @patch('src.models.clip_adapter.torch.cuda.empty_cache')
    @patch('src.models.clip_adapter.open_clip.create_model_and_transforms')
    @patch.object(CLIPAdapter, 'detect_device')
    def test_unload_clears_model_and_cache(
        self, mock_detect_device, mock_create_model, mock_empty_cache
    ):
        """Should unload model and clear GPU cache."""
        mock_detect_device.return_value = "cuda"
        
        mock_model = MagicMock()
        mock_model.half.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_preprocess = MagicMock()
        mock_create_model.return_value = (mock_model, None, mock_preprocess)
        
        adapter = CLIPAdapter()
        adapter.load({"hf_id": "openai/clip-vit-large-patch14", "quantization": "fp16"})
        
        # Verify loaded
        assert adapter.model is not None
        assert adapter.device == "cuda"
        
        # Unload
        adapter.unload()
        
        # Verify cleared
        assert adapter.model is None
        assert adapter.preprocess is None
        assert adapter.device is None
        assert adapter.model_name is None
        mock_empty_cache.assert_called_once()
    
    @patch('src.models.clip_adapter.torch.mps.empty_cache')
    @patch('src.models.clip_adapter.open_clip.create_model_and_transforms')
    @patch.object(CLIPAdapter, 'detect_device')
    def test_unload_clears_mps_cache(
        self, mock_detect_device, mock_create_model, mock_empty_cache
    ):
        """Should clear MPS cache when device is MPS."""
        mock_detect_device.return_value = "mps"
        
        mock_model = MagicMock()
        mock_model.half.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_preprocess = MagicMock()
        mock_create_model.return_value = (mock_model, None, mock_preprocess)
        
        adapter = CLIPAdapter()
        adapter.load({"hf_id": "openai/clip-vit-large-patch14", "quantization": "fp16"})
        adapter.unload()
        
        mock_empty_cache.assert_called_once()
    
    def test_unload_without_loaded_model_logs_warning(self):
        """Should log warning when unloading non-existent model."""
        adapter = CLIPAdapter()
        
        # Should not raise error
        adapter.unload()
        
        assert adapter.model is None


class TestCLIPAdapterMemoryFootprint:
    """Test memory footprint calculation."""
    
    @patch('src.models.clip_adapter.open_clip.create_model_and_transforms')
    @patch.object(CLIPAdapter, 'detect_device')
    def test_get_memory_footprint_returns_positive_int(
        self, mock_detect_device, mock_create_model
    ):
        """Should calculate and return memory footprint in bytes."""
        mock_detect_device.return_value = "cpu"
        
        # Create mock model with known parameters
        mock_model = MagicMock()
        mock_model.float.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        # Mock parameters (simulate 100M parameters in FP32 = 400MB)
        mock_params = [
            torch.nn.Parameter(torch.randn(50_000_000)),  # 50M params
            torch.nn.Parameter(torch.randn(50_000_000))   # 50M params
        ]
        mock_model.parameters.return_value = mock_params
        mock_model.buffers.return_value = []
        
        mock_preprocess = MagicMock()
        mock_create_model.return_value = (mock_model, None, mock_preprocess)
        
        adapter = CLIPAdapter()
        adapter.load({"hf_id": "openai/clip-vit-base-patch32", "quantization": "fp32"})
        
        footprint = adapter.get_memory_footprint()
        
        # Verify footprint
        assert isinstance(footprint, int)
        assert footprint > 0
        # Should be ~480MB (400MB params + 20% overhead)
        expected_bytes = 100_000_000 * 4 * 1.2  # 100M params * 4 bytes * 1.2 overhead
        assert abs(footprint - expected_bytes) < 1_000_000  # Within 1MB tolerance
    
    def test_get_memory_footprint_without_loaded_model_raises_error(self):
        """Should raise RuntimeError if model not loaded."""
        adapter = CLIPAdapter()
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.get_memory_footprint()


class TestModelNameParsing:
    """Test HuggingFace ID to open_clip name parsing."""
    
    def test_parse_vit_large_patch14(self):
        """Should parse ViT-L-14 correctly."""
        adapter = CLIPAdapter()
        result = adapter._parse_model_name("openai/clip-vit-large-patch14")
        assert result == "ViT-L-14"
    
    def test_parse_vit_base_patch32(self):
        """Should parse ViT-B-32 correctly."""
        adapter = CLIPAdapter()
        result = adapter._parse_model_name("openai/clip-vit-base-patch32")
        assert result == "ViT-B-32"
    
    def test_parse_vit_base_patch16(self):
        """Should parse ViT-B-16 correctly."""
        adapter = CLIPAdapter()
        result = adapter._parse_model_name("openai/clip-vit-base-patch16")
        assert result == "ViT-B-16"
    
    def test_parse_without_org_prefix(self):
        """Should handle model names without organization prefix."""
        adapter = CLIPAdapter()
        result = adapter._parse_model_name("clip-vit-large-patch14")
        assert result == "ViT-L-14"
    
    def test_parse_unknown_model_returns_as_is(self):
        """Should return unknown models as-is with warning."""
        adapter = CLIPAdapter()
        result = adapter._parse_model_name("custom/unknown-model")
        assert result == "unknown-model"
