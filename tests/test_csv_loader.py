"""
Unit tests for CSV metadata loading.

Tests validate CSV parsing, metadata extraction, path validation,
error handling, and logging functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
import csv

from src.csv_loader import CSVLoader
from src.data_models import ImageMetadata


# Fixtures for creating temporary test data
@pytest.fixture
def temp_image_files():
    """Create multiple temporary image files for testing."""
    files = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            # Write minimal PNG header
            f.write(b'\x89PNG\r\n\x1a\n')
            files.append(f.name)
    yield files
    # Cleanup
    for path in files:
        if os.path.exists(path):
            os.unlink(path)


@pytest.fixture
def temp_csv_valid(temp_image_files):
    """Create a valid CSV file with image metadata."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'path', 'prompt', 'model', 'style'])
        writer.writerow(['img_001', temp_image_files[0], 'A sunset', 'dalle-3', 'realistic'])
        writer.writerow(['img_002', temp_image_files[1], 'A mountain', 'midjourney', 'artistic'])
        writer.writerow(['img_003', temp_image_files[2], 'A river', 'stable-diffusion', 'photographic'])
        csv_path = f.name
    yield csv_path
    # Cleanup
    if os.path.exists(csv_path):
        os.unlink(csv_path)


@pytest.fixture
def temp_csv_minimal(temp_image_files):
    """Create a CSV with only required columns."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'path'])
        writer.writerow(['img_001', temp_image_files[0]])
        writer.writerow(['img_002', temp_image_files[1]])
        csv_path = f.name
    yield csv_path
    # Cleanup
    if os.path.exists(csv_path):
        os.unlink(csv_path)


@pytest.fixture
def temp_csv_missing_files():
    """Create a CSV with references to non-existent files."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'path'])
        writer.writerow(['img_001', '/nonexistent/path/image1.png'])
        writer.writerow(['img_002', '/nonexistent/path/image2.png'])
        csv_path = f.name
    yield csv_path
    # Cleanup
    if os.path.exists(csv_path):
        os.unlink(csv_path)


@pytest.fixture
def temp_csv_mixed(temp_image_files):
    """Create a CSV with mix of valid and invalid entries."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'path', 'prompt'])
        writer.writerow(['img_001', temp_image_files[0], 'Valid image'])
        writer.writerow(['img_002', '/nonexistent/image.png', 'Missing file'])
        writer.writerow(['img_003', temp_image_files[1], 'Another valid'])
        csv_path = f.name
    yield csv_path
    # Cleanup
    if os.path.exists(csv_path):
        os.unlink(csv_path)


@pytest.fixture
def temp_csv_unsupported_format():
    """Create a CSV with unsupported image format."""
    # Create a text file (unsupported format)
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b'not an image')
        txt_path = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'path'])
        writer.writerow(['img_001', txt_path])
        csv_path = f.name
    
    yield csv_path
    
    # Cleanup
    if os.path.exists(csv_path):
        os.unlink(csv_path)
    if os.path.exists(txt_path):
        os.unlink(txt_path)


# CSVLoader Tests
class TestCSVLoader:
    """Tests for CSVLoader functionality."""
    
    def test_load_valid_csv(self, temp_csv_valid):
        """Test loading a valid CSV with all columns."""
        loader = CSVLoader()
        metadata_list = loader.load_metadata(temp_csv_valid)
        
        assert len(metadata_list) == 3
        assert metadata_list[0].id == 'img_001'
        assert metadata_list[0].prompt == 'A sunset'
        assert metadata_list[0].model == 'dalle-3'
        assert metadata_list[0].attributes['style'] == 'realistic'
        
        assert metadata_list[1].id == 'img_002'
        assert metadata_list[1].prompt == 'A mountain'
        assert metadata_list[1].model == 'midjourney'
        
        assert metadata_list[2].id == 'img_003'
        assert len(loader.get_load_errors()) == 0
    
    def test_load_minimal_csv(self, temp_csv_minimal):
        """Test loading CSV with only required columns."""
        loader = CSVLoader()
        metadata_list = loader.load_metadata(temp_csv_minimal)
        
        assert len(metadata_list) == 2
        assert metadata_list[0].id == 'img_001'
        assert metadata_list[0].prompt is None
        assert metadata_list[0].model is None
        assert metadata_list[0].attributes == {}
        
        assert metadata_list[1].id == 'img_002'
        assert len(loader.get_load_errors()) == 0
    
    def test_load_nonexistent_csv(self):
        """Test loading non-existent CSV file."""
        loader = CSVLoader()
        metadata_list = loader.load_metadata('/nonexistent/file.csv')
        
        assert len(metadata_list) == 0
        assert len(loader.get_load_errors()) == 1
        assert 'does not exist' in loader.get_load_errors()[0][1]
    
    def test_load_csv_missing_required_columns(self, temp_image_files):
        """Test CSV missing required columns."""
        # Create CSV without 'id' column
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'prompt'])  # Missing 'id'
            writer.writerow([temp_image_files[0], 'A test'])
            csv_path = f.name
        
        try:
            loader = CSVLoader()
            metadata_list = loader.load_metadata(csv_path)
            
            assert len(metadata_list) == 0
            assert len(loader.get_load_errors()) == 1
            assert 'missing required columns' in loader.get_load_errors()[0][1].lower()
            assert 'id' in loader.get_load_errors()[0][1]
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
    
    def test_load_csv_with_missing_files(self, temp_csv_missing_files):
        """Test CSV with references to non-existent image files."""
        loader = CSVLoader()
        metadata_list = loader.load_metadata(temp_csv_missing_files)
        
        # All entries should fail validation
        assert len(metadata_list) == 0
        assert len(loader.get_load_errors()) == 2
        
        # Check that errors were logged for missing files
        errors = loader.get_load_errors()
        assert any('img_001' in err[0] for err in errors)
        assert any('img_002' in err[0] for err in errors)
    
    def test_load_csv_mixed_valid_invalid(self, temp_csv_mixed):
        """Test CSV with mix of valid and invalid entries."""
        loader = CSVLoader()
        metadata_list = loader.load_metadata(temp_csv_mixed)
        
        # Only 2 valid entries
        assert len(metadata_list) == 2
        assert metadata_list[0].id == 'img_001'
        assert metadata_list[1].id == 'img_003'
        
        # One error for missing file
        assert len(loader.get_load_errors()) == 1
        assert 'img_002' in loader.get_load_errors()[0][0]
    
    def test_load_csv_unsupported_format(self, temp_csv_unsupported_format):
        """Test CSV with unsupported image format."""
        loader = CSVLoader()
        metadata_list = loader.load_metadata(temp_csv_unsupported_format)
        
        assert len(metadata_list) == 0
        assert len(loader.get_load_errors()) == 1
        assert 'Unsupported image format' in loader.get_load_errors()[0][1]
    
    def test_error_tracking(self, temp_csv_missing_files):
        """Test error tracking functionality."""
        loader = CSVLoader()
        loader.load_metadata(temp_csv_missing_files)
        
        errors = loader.get_load_errors()
        assert len(errors) == 2
        
        # Each error should be a tuple (identifier, message)
        assert all(isinstance(err, tuple) and len(err) == 2 for err in errors)
        assert all(isinstance(err[0], str) and isinstance(err[1], str) for err in errors)
    
    def test_clear_errors(self, temp_csv_missing_files):
        """Test clearing accumulated errors."""
        loader = CSVLoader()
        loader.load_metadata(temp_csv_missing_files)
        
        assert len(loader.get_load_errors()) > 0
        
        loader.clear_errors()
        assert len(loader.get_load_errors()) == 0
    
    def test_multiple_loads_accumulate_errors(self, temp_csv_missing_files, temp_csv_valid):
        """Test that multiple loads without clearing accumulate errors."""
        loader = CSVLoader()
        
        # First load with errors
        loader.load_metadata(temp_csv_missing_files)
        error_count_1 = len(loader.get_load_errors())
        assert error_count_1 == 2
        
        # Second load with no errors
        loader.load_metadata(temp_csv_valid)
        error_count_2 = len(loader.get_load_errors())
        assert error_count_2 == error_count_1  # Still 2 errors from first load
    
    def test_empty_csv(self):
        """Test loading an empty CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'path'])  # Header only, no data
            csv_path = f.name
        
        try:
            loader = CSVLoader()
            metadata_list = loader.load_metadata(csv_path)
            
            assert len(metadata_list) == 0
            assert len(loader.get_load_errors()) == 0
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
    
    def test_csv_with_empty_values(self, temp_image_files):
        """Test CSV with empty ID or path values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'path'])
            writer.writerow(['', temp_image_files[0]])  # Empty ID
            writer.writerow(['img_002', ''])  # Empty path
            writer.writerow(['img_003', temp_image_files[1]])  # Valid
            csv_path = f.name
        
        try:
            loader = CSVLoader()
            metadata_list = loader.load_metadata(csv_path)
            
            # Only 1 valid entry
            assert len(metadata_list) == 1
            assert metadata_list[0].id == 'img_003'
            
            # 2 errors for empty values
            assert len(loader.get_load_errors()) == 2
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
    
    def test_csv_with_whitespace(self, temp_image_files):
        """Test CSV with whitespace in values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'path', 'prompt'])
            writer.writerow(['  img_001  ', f'  {temp_image_files[0]}  ', '  Test prompt  '])
            csv_path = f.name
        
        try:
            loader = CSVLoader()
            metadata_list = loader.load_metadata(csv_path)
            
            assert len(metadata_list) == 1
            # Whitespace should be stripped
            assert metadata_list[0].id == 'img_001'
            assert metadata_list[0].path == temp_image_files[0]
            assert metadata_list[0].prompt == 'Test prompt'
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
    
    def test_csv_with_null_optional_fields(self, temp_image_files):
        """Test CSV with null/NaN optional fields."""
        import pandas as pd
        import numpy as np
        
        # Create CSV with pandas to control null values
        df = pd.DataFrame({
            'id': ['img_001', 'img_002'],
            'path': [temp_image_files[0], temp_image_files[1]],
            'prompt': ['Valid prompt', np.nan],
            'model': [np.nan, 'dalle-3']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            csv_path = f.name
        
        df.to_csv(csv_path, index=False)
        
        try:
            loader = CSVLoader()
            metadata_list = loader.load_metadata(csv_path)
            
            assert len(metadata_list) == 2
            assert metadata_list[0].prompt == 'Valid prompt'
            assert metadata_list[0].model is None
            assert metadata_list[1].prompt is None
            assert metadata_list[1].model == 'dalle-3'
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
    
    def test_validate_csv_format_valid(self, temp_csv_valid):
        """Test static validation method with valid CSV."""
        is_valid, error_msg = CSVLoader.validate_csv_format(temp_csv_valid)
        
        assert is_valid is True
        assert error_msg == ""
    
    def test_validate_csv_format_missing_columns(self, temp_image_files):
        """Test static validation method with missing columns."""
        # Create CSV without 'path' column
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'prompt'])
            writer.writerow(['img_001', 'Test'])
            csv_path = f.name
        
        try:
            is_valid, error_msg = CSVLoader.validate_csv_format(csv_path)
            
            assert is_valid is False
            assert 'missing required columns' in error_msg.lower()
            assert 'path' in error_msg
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
    
    def test_validate_csv_format_nonexistent(self):
        """Test static validation method with non-existent file."""
        is_valid, error_msg = CSVLoader.validate_csv_format('/nonexistent/file.csv')
        
        assert is_valid is False
        assert 'does not exist' in error_msg.lower()
    
    def test_additional_attributes(self, temp_image_files):
        """Test that additional CSV columns are stored in attributes."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'path', 'style', 'resolution', 'seed'])
            writer.writerow(['img_001', temp_image_files[0], 'realistic', '1024x1024', '12345'])
            csv_path = f.name
        
        try:
            loader = CSVLoader()
            metadata_list = loader.load_metadata(csv_path)
            
            assert len(metadata_list) == 1
            assert metadata_list[0].attributes['style'] == 'realistic'
            assert metadata_list[0].attributes['resolution'] == '1024x1024'
            assert metadata_list[0].attributes['seed'] == '12345'
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
    
    def test_malformed_csv(self):
        """Test handling of malformed CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("This is not a valid CSV\n")
            f.write("It has no structure\n")
            csv_path = f.name
        
        try:
            loader = CSVLoader()
            metadata_list = loader.load_metadata(csv_path)
            
            # Should handle gracefully
            assert len(metadata_list) == 0
            errors = loader.get_load_errors()
            assert len(errors) > 0
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
