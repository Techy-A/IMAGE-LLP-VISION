"""
Unit tests for ImageLoader functionality.

Tests validate image loading, format support, batch loading,
error handling, and logging functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path

from PIL import Image

from src.image_loader import ImageLoader


# Fixtures for creating temporary test images
@pytest.fixture
def temp_png_image():
    """Create a temporary valid PNG image."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        # Create a small 10x10 red image
        img = Image.new('RGB', (10, 10), color='red')
        img.save(f.name, 'PNG')
        image_path = f.name
    yield image_path
    # Cleanup
    if os.path.exists(image_path):
        os.unlink(image_path)


@pytest.fixture
def temp_jpg_image():
    """Create a temporary valid JPG image."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img = Image.new('RGB', (10, 10), color='blue')
        img.save(f.name, 'JPEG')
        image_path = f.name
    yield image_path
    if os.path.exists(image_path):
        os.unlink(image_path)


@pytest.fixture
def temp_jpeg_image():
    """Create a temporary valid JPEG image."""
    with tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False) as f:
        img = Image.new('RGB', (10, 10), color='green')
        img.save(f.name, 'JPEG')
        image_path = f.name
    yield image_path
    if os.path.exists(image_path):
        os.unlink(image_path)


@pytest.fixture
def temp_webp_image():
    """Create a temporary valid WEBP image."""
    with tempfile.NamedTemporaryFile(suffix='.webp', delete=False) as f:
        img = Image.new('RGB', (10, 10), color='yellow')
        img.save(f.name, 'WEBP')
        image_path = f.name
    yield image_path
    if os.path.exists(image_path):
        os.unlink(image_path)


@pytest.fixture
def temp_rgba_image():
    """Create a temporary RGBA PNG image (needs conversion to RGB)."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img = Image.new('RGBA', (10, 10), color=(255, 0, 0, 128))
        img.save(f.name, 'PNG')
        image_path = f.name
    yield image_path
    if os.path.exists(image_path):
        os.unlink(image_path)


@pytest.fixture
def temp_grayscale_image():
    """Create a temporary grayscale image (needs conversion to RGB)."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img = Image.new('L', (10, 10), color=128)
        img.save(f.name, 'PNG')
        image_path = f.name
    yield image_path
    if os.path.exists(image_path):
        os.unlink(image_path)


@pytest.fixture
def temp_corrupted_image():
    """Create a corrupted image file (truncated PNG)."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        # Write incomplete PNG header
        f.write(b'\x89PNG\r\n')
        image_path = f.name
    yield image_path
    if os.path.exists(image_path):
        os.unlink(image_path)


@pytest.fixture
def temp_unsupported_format():
    """Create a file with unsupported format (.txt)."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b'This is not an image')
        file_path = f.name
    yield file_path
    if os.path.exists(file_path):
        os.unlink(file_path)


@pytest.fixture
def temp_multiple_images():
    """Create multiple temporary images for batch testing."""
    images = []
    for i, color in enumerate(['red', 'green', 'blue']):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (10, 10), color=color)
            img.save(f.name, 'PNG')
            images.append(f.name)
    yield images
    # Cleanup
    for path in images:
        if os.path.exists(path):
            os.unlink(path)


# ImageLoader Tests
class TestImageLoader:
    """Tests for ImageLoader functionality."""
    
    def test_load_png_image(self, temp_png_image):
        """Test loading a valid PNG image."""
        loader = ImageLoader()
        image = loader.load_image(temp_png_image)
        
        assert image is not None
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
        assert image.size == (10, 10)
        assert len(loader.get_load_errors()) == 0
    
    def test_load_jpg_image(self, temp_jpg_image):
        """Test loading a valid JPG image."""
        loader = ImageLoader()
        image = loader.load_image(temp_jpg_image)
        
        assert image is not None
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
        assert image.size == (10, 10)
        assert len(loader.get_load_errors()) == 0
    
    def test_load_jpeg_image(self, temp_jpeg_image):
        """Test loading a valid JPEG image."""
        loader = ImageLoader()
        image = loader.load_image(temp_jpeg_image)
        
        assert image is not None
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
        assert image.size == (10, 10)
        assert len(loader.get_load_errors()) == 0
    
    def test_load_webp_image(self, temp_webp_image):
        """Test loading a valid WEBP image."""
        loader = ImageLoader()
        image = loader.load_image(temp_webp_image)
        
        assert image is not None
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
        assert image.size == (10, 10)
        assert len(loader.get_load_errors()) == 0
    
    def test_load_rgba_conversion(self, temp_rgba_image):
        """Test that RGBA images are converted to RGB."""
        loader = ImageLoader()
        image = loader.load_image(temp_rgba_image)
        
        assert image is not None
        assert image.mode == 'RGB'  # Should be converted
        assert len(loader.get_load_errors()) == 0
    
    def test_load_grayscale_conversion(self, temp_grayscale_image):
        """Test that grayscale images are converted to RGB."""
        loader = ImageLoader()
        image = loader.load_image(temp_grayscale_image)
        
        assert image is not None
        assert image.mode == 'RGB'  # Should be converted
        assert len(loader.get_load_errors()) == 0
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        loader = ImageLoader()
        image = loader.load_image('/nonexistent/path/image.png')
        
        assert image is None
        assert len(loader.get_load_errors()) == 1
        assert 'does not exist' in loader.get_load_errors()[0][1]
    
    def test_load_directory_path(self, temp_png_image):
        """Test loading with a directory path instead of file."""
        loader = ImageLoader()
        dir_path = os.path.dirname(temp_png_image)
        image = loader.load_image(dir_path)
        
        assert image is None
        assert len(loader.get_load_errors()) == 1
        assert 'not a file' in loader.get_load_errors()[0][1]
    
    def test_load_unsupported_format(self, temp_unsupported_format):
        """Test loading a file with unsupported format."""
        loader = ImageLoader()
        image = loader.load_image(temp_unsupported_format)
        
        assert image is None
        assert len(loader.get_load_errors()) == 1
        assert 'Unsupported image format' in loader.get_load_errors()[0][1]
        assert '.txt' in loader.get_load_errors()[0][1]
    
    def test_load_corrupted_image(self, temp_corrupted_image):
        """Test loading a corrupted image file."""
        loader = ImageLoader()
        image = loader.load_image(temp_corrupted_image)
        
        assert image is None
        assert len(loader.get_load_errors()) == 1
        # Should detect corruption
        error_msg = loader.get_load_errors()[0][1].lower()
        assert 'corrupted' in error_msg or 'error' in error_msg
    
    def test_batch_load_all_valid(self, temp_multiple_images):
        """Test batch loading with all valid images."""
        loader = ImageLoader()
        images = loader.batch_load(temp_multiple_images)
        
        assert len(images) == 3
        assert all(img is not None for img in images)
        assert all(isinstance(img, Image.Image) for img in images)
        assert all(img.mode == 'RGB' for img in images)
        assert len(loader.get_load_errors()) == 0
    
    def test_batch_load_mixed_valid_invalid(self, temp_multiple_images):
        """Test batch loading with mix of valid and invalid paths."""
        loader = ImageLoader()
        paths = [
            temp_multiple_images[0],
            '/nonexistent/image.png',
            temp_multiple_images[1],
            '/another/missing.jpg',
            temp_multiple_images[2]
        ]
        images = loader.batch_load(paths)
        
        assert len(images) == 5
        assert images[0] is not None
        assert images[1] is None
        assert images[2] is not None
        assert images[3] is None
        assert images[4] is not None
        
        # Should have 2 errors
        assert len(loader.get_load_errors()) == 2
    
    def test_batch_load_empty_list(self):
        """Test batch loading with empty list."""
        loader = ImageLoader()
        images = loader.batch_load([])
        
        assert images == []
        assert len(loader.get_load_errors()) == 0
    
    def test_batch_load_all_invalid(self):
        """Test batch loading with all invalid paths."""
        loader = ImageLoader()
        paths = [
            '/nonexistent/image1.png',
            '/nonexistent/image2.jpg',
            '/nonexistent/image3.webp'
        ]
        images = loader.batch_load(paths)
        
        assert len(images) == 3
        assert all(img is None for img in images)
        assert len(loader.get_load_errors()) == 3
    
    def test_error_tracking_multiple_loads(self, temp_png_image):
        """Test that errors accumulate across multiple loads."""
        loader = ImageLoader()
        
        # First load - error
        loader.load_image('/nonexistent/image1.png')
        assert len(loader.get_load_errors()) == 1
        
        # Second load - success (no new errors)
        loader.load_image(temp_png_image)
        assert len(loader.get_load_errors()) == 1
        
        # Third load - error
        loader.load_image('/nonexistent/image2.png')
        assert len(loader.get_load_errors()) == 2
    
    def test_get_load_errors(self, temp_png_image):
        """Test get_load_errors returns a copy."""
        loader = ImageLoader()
        loader.load_image('/nonexistent/image.png')
        
        errors1 = loader.get_load_errors()
        errors2 = loader.get_load_errors()
        
        # Should be equal but not the same object
        assert errors1 == errors2
        assert errors1 is not errors2
    
    def test_clear_errors(self):
        """Test clearing accumulated errors."""
        loader = ImageLoader()
        loader.load_image('/nonexistent/image1.png')
        loader.load_image('/nonexistent/image2.png')
        
        assert len(loader.get_load_errors()) == 2
        
        loader.clear_errors()
        assert len(loader.get_load_errors()) == 0
    
    def test_error_tuple_structure(self):
        """Test that errors are properly structured tuples."""
        loader = ImageLoader()
        loader.load_image('/nonexistent/image.png')
        
        errors = loader.get_load_errors()
        assert len(errors) == 1
        assert isinstance(errors[0], tuple)
        assert len(errors[0]) == 2
        assert isinstance(errors[0][0], str)  # Path
        assert isinstance(errors[0][1], str)  # Error message
    
    def test_is_supported_format_valid(self):
        """Test is_supported_format with valid formats."""
        assert ImageLoader.is_supported_format('image.png') is True
        assert ImageLoader.is_supported_format('image.jpg') is True
        assert ImageLoader.is_supported_format('image.jpeg') is True
        assert ImageLoader.is_supported_format('image.webp') is True
        
        # Case insensitive
        assert ImageLoader.is_supported_format('IMAGE.PNG') is True
        assert ImageLoader.is_supported_format('image.JPG') is True
    
    def test_is_supported_format_invalid(self):
        """Test is_supported_format with invalid formats."""
        assert ImageLoader.is_supported_format('file.txt') is False
        assert ImageLoader.is_supported_format('file.pdf') is False
        assert ImageLoader.is_supported_format('file.gif') is False
        assert ImageLoader.is_supported_format('file.bmp') is False
        assert ImageLoader.is_supported_format('file') is False
    
    def test_is_supported_format_with_path(self):
        """Test is_supported_format with full paths."""
        assert ImageLoader.is_supported_format('/path/to/image.png') is True
        assert ImageLoader.is_supported_format('/path/to/file.txt') is False
        assert ImageLoader.is_supported_format('../relative/path/image.jpg') is True
    
    def test_supported_formats_constant(self):
        """Test that SUPPORTED_FORMATS constant contains expected formats."""
        expected_formats = {'.png', '.jpg', '.jpeg', '.webp'}
        assert ImageLoader.SUPPORTED_FORMATS == expected_formats
    
    def test_load_image_preserves_size(self, temp_png_image):
        """Test that loading preserves image dimensions."""
        # Create image with specific size
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (100, 200), color='red')
            img.save(f.name, 'PNG')
            large_image_path = f.name
        
        try:
            loader = ImageLoader()
            loaded_image = loader.load_image(large_image_path)
            
            assert loaded_image is not None
            assert loaded_image.size == (100, 200)
        finally:
            if os.path.exists(large_image_path):
                os.unlink(large_image_path)
    
    def test_batch_load_maintains_order(self, temp_multiple_images):
        """Test that batch_load returns images in same order as input paths."""
        loader = ImageLoader()
        
        # Create images with distinct colors
        paths = temp_multiple_images
        images = loader.batch_load(paths)
        
        # Each image should correspond to its path
        assert len(images) == len(paths)
        for i, img in enumerate(images):
            assert img is not None
            # Verify image loaded from correct path
            img_direct = loader.load_image(paths[i])
            assert img.size == img_direct.size
    
    def test_case_insensitive_extension(self):
        """Test that file extensions are case-insensitive."""
        # Create images with uppercase extensions
        with tempfile.NamedTemporaryFile(suffix='.PNG', delete=False) as f:
            img = Image.new('RGB', (10, 10), color='red')
            img.save(f.name, 'PNG')
            upper_png_path = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.JPG', delete=False) as f:
            img = Image.new('RGB', (10, 10), color='blue')
            img.save(f.name, 'JPEG')
            upper_jpg_path = f.name
        
        try:
            loader = ImageLoader()
            
            png_img = loader.load_image(upper_png_path)
            jpg_img = loader.load_image(upper_jpg_path)
            
            assert png_img is not None
            assert jpg_img is not None
            assert len(loader.get_load_errors()) == 0
        finally:
            if os.path.exists(upper_png_path):
                os.unlink(upper_png_path)
            if os.path.exists(upper_jpg_path):
                os.unlink(upper_jpg_path)
    
    def test_multiple_loaders_independent(self):
        """Test that multiple ImageLoader instances are independent."""
        loader1 = ImageLoader()
        loader2 = ImageLoader()
        
        loader1.load_image('/nonexistent/image1.png')
        loader2.load_image('/nonexistent/image2.png')
        
        assert len(loader1.get_load_errors()) == 1
        assert len(loader2.get_load_errors()) == 1
        
        # Errors should be different
        assert loader1.get_load_errors()[0][0] != loader2.get_load_errors()[0][0]
