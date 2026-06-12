"""
Unit tests for BaseVisionModel abstract base class.

Tests cover:
- Abstract method enforcement
- Device auto-detection
- Quantization configuration
- Type hints and validation
"""

import pytest
import torch
from unittest.mock import patch, MagicMock
from transformers import BitsAndBytesConfig

from src.models.base_vision_model import BaseVisionModel


class ConcreteVisionModel(BaseVisionModel):
    """Concrete implementation for testing abstract base class."""
    
    def load(self, config):
        """Test implementation of load."""
        self.device = self.detect_device()
        self.model = MagicMock()
    
    def encode_image(self, image):
        """Test implementation of encode_image."""
        return torch.randn(1, 512)  # Mock embedding
    
    def unload(self):
        """Test implementation of unload."""
        self.model = None
        self.device = None
    
    def get_memory_footprint(self) -> int:
        """Test implementation of get_memory_footprint."""
        return 1024 * 1024 * 1024  # 1GB mock


class TestBaseVisionModelAbstraction:
    """Test that BaseVisionModel enforces abstract method implementation."""
    
    def test_cannot_instantiate_base_class(self):
        """BaseVisionModel cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseVisionModel()
    
    def test_concrete_subclass_can_be_instantiated(self):
        """Concrete subclass with all methods implemented can be instantiated."""
        model = ConcreteVisionModel()
        assert model is not None
        assert isinstance(model, BaseVisionModel)
    
    def test_incomplete_subclass_cannot_be_instantiated(self):
        """Subclass missing abstract methods cannot be instantiated."""
        
        class IncompleteModel(BaseVisionModel):
            def load(self, config):
                pass
            # Missing: encode_image, unload, get_memory_footprint
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteModel()


class TestDeviceDetection:
    """Test device auto-detection functionality."""
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_detect_cuda_when_available(self, mock_mps, mock_cuda):
        """Should detect CUDA when available (highest priority)."""
        mock_cuda.return_value = True
        mock_mps.return_value = False
        
        model = ConcreteVisionModel()
        device = model.detect_device()
        
        assert device == "cuda"
        mock_cuda.assert_called_once()
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_detect_mps_when_cuda_unavailable(self, mock_mps, mock_cuda):
        """Should detect MPS when CUDA unavailable but MPS available."""
        mock_cuda.return_value = False
        mock_mps.return_value = True
        
        model = ConcreteVisionModel()
        device = model.detect_device()
        
        assert device == "mps"
        mock_cuda.assert_called_once()
        mock_mps.assert_called_once()
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_fallback_to_cpu(self, mock_mps, mock_cuda):
        """Should fallback to CPU when no GPU available."""
        mock_cuda.return_value = False
        mock_mps.return_value = False
        
        model = ConcreteVisionModel()
        device = model.detect_device()
        
        assert device == "cpu"
        mock_cuda.assert_called_once()
        mock_mps.assert_called_once()
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_cuda_takes_precedence_over_mps(self, mock_mps, mock_cuda):
        """Should prefer CUDA over MPS when both available."""
        mock_cuda.return_value = True
        mock_mps.return_value = True
        
        model = ConcreteVisionModel()
        device = model.detect_device()
        
        assert device == "cuda"
        # MPS check should not be called when CUDA is available
        mock_mps.assert_not_called()


class TestQuantizationConfiguration:
    """Test quantization configuration creation."""
    
    @patch('src.models.base_vision_model.BitsAndBytesConfig')
    def test_4bit_quantization_config(self, mock_config_class):
        """Should create correct BitsAndBytesConfig for 4-bit quantization."""
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config
        
        model = ConcreteVisionModel()
        config = model.load_with_quantization("4bit")
        
        # Verify BitsAndBytesConfig was called with correct parameters
        mock_config_class.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        assert config == mock_config
    
    @patch('src.models.base_vision_model.BitsAndBytesConfig')
    def test_8bit_quantization_config(self, mock_config_class):
        """Should create correct BitsAndBytesConfig for 8-bit quantization."""
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config
        
        model = ConcreteVisionModel()
        config = model.load_with_quantization("8bit")
        
        # Verify BitsAndBytesConfig was called with correct parameters
        mock_config_class.assert_called_once_with(
            load_in_8bit=True
        )
        assert config == mock_config
    
    def test_fp16_returns_none(self):
        """Should return None for fp16 (no BitsAndBytesConfig needed)."""
        model = ConcreteVisionModel()
        config = model.load_with_quantization("fp16")
        
        assert config is None
    
    def test_fp32_returns_none(self):
        """Should return None for fp32 (no BitsAndBytesConfig needed)."""
        model = ConcreteVisionModel()
        config = model.load_with_quantization("fp32")
        
        assert config is None
    
    def test_invalid_quantization_mode_raises_error(self):
        """Should raise ValueError for invalid quantization mode."""
        model = ConcreteVisionModel()
        
        with pytest.raises(ValueError, match="Invalid quantization mode"):
            model.load_with_quantization("16bit")
        
        with pytest.raises(ValueError, match="Invalid quantization mode"):
            model.load_with_quantization("invalid")
        
        with pytest.raises(ValueError, match="Invalid quantization mode"):
            model.load_with_quantization("")


class TestConcreteImplementation:
    """Test concrete implementation behaviors."""
    
    def test_initialization(self):
        """Should initialize with None device and model."""
        model = ConcreteVisionModel()
        
        assert model.device is None
        assert model.model is None
    
    def test_load_sets_device(self):
        """Load should set device via detect_device."""
        model = ConcreteVisionModel()
        model.load({})
        
        assert model.device is not None
        assert model.device in ["cuda", "mps", "cpu"]
        assert model.model is not None
    
    def test_encode_image_returns_tensor(self):
        """encode_image should return a Tensor."""
        model = ConcreteVisionModel()
        image = MagicMock()  # Mock PIL Image
        
        result = model.encode_image(image)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 512)
    
    def test_unload_clears_state(self):
        """unload should clear model and device."""
        model = ConcreteVisionModel()
        model.load({})
        
        assert model.model is not None
        assert model.device is not None
        
        model.unload()
        
        assert model.model is None
        assert model.device is None
    
    def test_get_memory_footprint_returns_int(self):
        """get_memory_footprint should return integer bytes."""
        model = ConcreteVisionModel()
        
        footprint = model.get_memory_footprint()
        
        assert isinstance(footprint, int)
        assert footprint > 0


class TestTypeHints:
    """Verify type hints are properly defined."""
    
    def test_abstract_methods_have_type_hints(self):
        """All abstract methods should have type hints."""
        import inspect
        
        # Get abstract methods
        abstract_methods = [
            method for method in dir(BaseVisionModel)
            if hasattr(getattr(BaseVisionModel, method), '__isabstractmethod__')
            and getattr(getattr(BaseVisionModel, method), '__isabstractmethod__')
        ]
        
        # Should have 4 abstract methods
        assert len(abstract_methods) == 4
        assert 'load' in abstract_methods
        assert 'encode_image' in abstract_methods
        assert 'unload' in abstract_methods
        assert 'get_memory_footprint' in abstract_methods
    
    def test_concrete_methods_have_type_hints(self):
        """Concrete helper methods should have type hints."""
        import inspect
        
        # Check detect_device signature
        sig = inspect.signature(BaseVisionModel.detect_device)
        assert sig.return_annotation == str
        
        # Check load_with_quantization signature
        sig = inspect.signature(BaseVisionModel.load_with_quantization)
        assert 'quantization' in sig.parameters
        assert sig.parameters['quantization'].annotation == str
