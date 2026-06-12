"""
Unit tests for CSVLoader multi-sheet XLSX functionality.

Tests the ability to load image paths from specific cell ranges
across multiple sheets in an Excel workbook.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.csv_loader import CSVLoader
from src.data_models import ImageMetadata


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_xlsx_multisheet(temp_dir):
    """
    Create a sample multi-sheet XLSX file with image paths in column E.
    
    Structure:
    - 3 sheets: Sheet1, Sheet2, Sheet3
    - Column E contains image paths in rows 2-6 (5 images per sheet)
    - Row 1 is header
    """
    xlsx_path = Path(temp_dir) / "test_multisheet.xlsx"
    
    # Create test images
    image_dir = Path(temp_dir) / "images"
    image_dir.mkdir()
    
    image_paths = []
    for i in range(15):  # 3 sheets × 5 images
        image_path = image_dir / f"test_image_{i:02d}.png"
        # Create minimal PNG file (1x1 pixel)
        with open(image_path, 'wb') as f:
            # PNG magic number + minimal IHDR chunk for 1x1 pixel
            f.write(b'\x89PNG\r\n\x1a\n'
                   b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
                   b'\x08\x02\x00\x00\x00\x90wS\xde'
                   b'\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01'
                   b'\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')
        image_paths.append(str(image_path))
    
    # Create Excel file with multiple sheets
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        for sheet_num in range(1, 4):  # Sheet1, Sheet2, Sheet3
            # Create DataFrame with image paths in column E (index 4)
            # Rows: Header + 5 data rows
            data = {
                'A': ['HeaderA'] + [f'a{i}' for i in range(1, 6)],
                'B': ['HeaderB'] + [f'b{i}' for i in range(1, 6)],
                'C': ['HeaderC'] + [f'c{i}' for i in range(1, 6)],
                'D': ['HeaderD'] + [f'd{i}' for i in range(1, 6)],
                'E': ['ImagePath'] + image_paths[(sheet_num-1)*5:sheet_num*5]
            }
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=f'Sheet{sheet_num}', index=False, header=False)
    
    return xlsx_path, image_paths


def test_load_multisheet_basic(sample_xlsx_multisheet):
    """Test basic multi-sheet loading."""
    xlsx_path, expected_paths = sample_xlsx_multisheet
    
    loader = CSVLoader()
    metadata = loader.load_metadata(
        str(xlsx_path),
        sheet_range="Sheet1:Sheet3",
        cell_range="E2:E6"
    )
    
    # Should load 3 sheets × 5 rows = 15 images
    assert len(metadata) == 15
    assert len(loader.get_load_errors()) == 0
    
    # Verify all are ImageMetadata
    for meta in metadata:
        assert isinstance(meta, ImageMetadata)
    
    # Verify paths match expected
    loaded_paths = [m.path for m in metadata]
    assert set(loaded_paths) == set(expected_paths)


def test_multisheet_id_generation(sample_xlsx_multisheet):
    """Test that unique IDs are generated for each image."""
    xlsx_path, _ = sample_xlsx_multisheet
    
    loader = CSVLoader()
    metadata = loader.load_metadata(
        str(xlsx_path),
        sheet_range="Sheet1:Sheet3",
        cell_range="E2:E6"
    )
    
    # All IDs should be unique
    ids = [m.id for m in metadata]
    assert len(ids) == len(set(ids))
    
    # IDs should follow pattern: sheetN_rowM
    for meta in metadata:
        assert '_row' in meta.id
        # Examples: sheet1_row2, sheet2_row3, etc.
        parts = meta.id.split('_row')
        assert len(parts) == 2
        assert parts[0] in ['sheet1', 'sheet2', 'sheet3']
        assert parts[1].isdigit()


def test_multisheet_attributes(sample_xlsx_multisheet):
    """Test that source information is stored in attributes."""
    xlsx_path, _ = sample_xlsx_multisheet
    
    loader = CSVLoader()
    metadata = loader.load_metadata(
        str(xlsx_path),
        sheet_range="Sheet1:Sheet3",
        cell_range="E2:E6"
    )
    
    for meta in metadata:
        # Check required attributes
        assert 'source_sheet' in meta.attributes
        assert 'source_cell' in meta.attributes
        assert 'source_file' in meta.attributes
        
        # Validate values
        assert meta.attributes['source_sheet'] in ['Sheet1', 'Sheet2', 'Sheet3']
        assert meta.attributes['source_cell'].startswith('E')
        assert meta.attributes['source_file'] == 'test_multisheet.xlsx'
        
        # Verify prompt and model parsing
        # Row number in ID format: sheetN_rowM
        row_num = int(meta.id.split('_row')[1])
        # Index in the original image paths list is (row_num - 2) within each sheet's block
        expected_idx = row_num - 1  # 1-based index in the data list (c1 for row 2, etc.)
        assert meta.prompt == f"c{expected_idx}"
        assert meta.model == f"b{expected_idx}"


def test_multisheet_single_sheet(sample_xlsx_multisheet):
    """Test loading from a single sheet in multi-sheet mode."""
    xlsx_path, expected_paths = sample_xlsx_multisheet
    
    loader = CSVLoader()
    metadata = loader.load_metadata(
        str(xlsx_path),
        sheet_range="Sheet2:Sheet2",  # Only Sheet2
        cell_range="E2:E6"
    )
    
    # Should load 1 sheet × 5 rows = 5 images
    assert len(metadata) == 5
    
    # All should be from Sheet2
    for meta in metadata:
        assert meta.attributes['source_sheet'] == 'Sheet2'
        assert meta.id.startswith('sheet2_')


def test_multisheet_invalid_sheet_range(sample_xlsx_multisheet):
    """Test error handling for invalid sheet range."""
    xlsx_path, _ = sample_xlsx_multisheet
    
    loader = CSVLoader()
    metadata = loader.load_metadata(
        str(xlsx_path),
        sheet_range="Sheet1:Sheet99",  # Sheet99 doesn't exist
        cell_range="E2:E6"
    )
    
    # Should fail gracefully
    assert len(metadata) == 0
    assert len(loader.get_load_errors()) > 0


def test_multisheet_invalid_cell_range_format(sample_xlsx_multisheet):
    """Test error handling for invalid cell range format."""
    xlsx_path, _ = sample_xlsx_multisheet
    
    loader = CSVLoader()
    
    # Invalid format: missing colon
    metadata = loader.load_metadata(
        str(xlsx_path),
        sheet_range="Sheet1:Sheet3",
        cell_range="E2"  # Should be "E2:E6"
    )
    
    assert len(metadata) == 0
    assert len(loader.get_load_errors()) > 0


def test_multisheet_multi_column_range(sample_xlsx_multisheet):
    """Test that multi-column ranges are rejected."""
    xlsx_path, _ = sample_xlsx_multisheet
    
    loader = CSVLoader()
    metadata = loader.load_metadata(
        str(xlsx_path),
        sheet_range="Sheet1:Sheet3",
        cell_range="D2:E6"  # Multi-column not supported
    )
    
    assert len(metadata) == 0
    errors = loader.get_load_errors()
    assert len(errors) > 0
    # Check error message mentions multi-column
    assert any('multi-column' in err[1].lower() for err in errors)


def test_multisheet_with_empty_cells(temp_dir):
    """Test handling of empty cells in the range."""
    xlsx_path = Path(temp_dir) / "test_empty_cells.xlsx"
    
    # Create test images
    image_dir = Path(temp_dir) / "images"
    image_dir.mkdir()
    
    image_paths = []
    for i in range(3):
        image_path = image_dir / f"test_image_{i}.png"
        with open(image_path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n'
                   b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
                   b'\x08\x02\x00\x00\x00\x90wS\xde'
                   b'\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01'
                   b'\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')
        image_paths.append(str(image_path))
    
    # Create Excel with some empty cells
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        data = {
            'A': ['A1', 'A2', 'A3', 'A4', 'A5'],
            'B': ['', '', '', '', ''],
            'C': ['', '', '', '', ''],
            'D': ['', '', '', '', ''],
            'E': ['Header', image_paths[0], '', image_paths[1], image_paths[2]]  # Empty at row 3
        }
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='Sheet1', index=False, header=False)
    
    loader = CSVLoader()
    metadata = loader.load_metadata(
        str(xlsx_path),
        sheet_range="Sheet1:Sheet1",
        cell_range="E2:E5"
    )
    
    # Should load 3 images (skipping the empty cell)
    assert len(metadata) == 3
    loaded_paths = [m.path for m in metadata]
    assert set(loaded_paths) == set(image_paths)


def test_multisheet_standard_mode_fallback(sample_xlsx_multisheet):
    """Test that standard mode works when sheet_range/cell_range not provided."""
    xlsx_path, _ = sample_xlsx_multisheet
    
    # Standard mode should fail because no 'id' and 'path' columns
    loader = CSVLoader()
    metadata = loader.load_metadata(str(xlsx_path))
    
    # Should fail with missing columns error
    assert len(metadata) == 0
    errors = loader.get_load_errors()
    assert len(errors) > 0
    assert any('missing required columns' in err[1].lower() for err in errors)


def test_multisheet_sheet_names_with_spaces(temp_dir):
    """Test handling of sheet names with spaces."""
    xlsx_path = Path(temp_dir) / "test_spaces.xlsx"
    
    # Create test image
    image_dir = Path(temp_dir) / "images"
    image_dir.mkdir()
    image_path = image_dir / "test.png"
    with open(image_path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n'
               b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
               b'\x08\x02\x00\x00\x00\x90wS\xde'
               b'\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01'
               b'\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')
    
    # Create Excel with space in sheet name
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        data = {
            'A': ['', ''],
            'B': ['', ''],
            'C': ['', ''],
            'D': ['', ''],
            'E': ['Header', str(image_path)]
        }
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='Sheet 1', index=False, header=False)
    
    loader = CSVLoader()
    metadata = loader.load_metadata(
        str(xlsx_path),
        sheet_range="Sheet 1:Sheet 1",
        cell_range="E2:E2"
    )
    
    # Should handle sheet names with spaces
    assert len(metadata) == 1
    assert metadata[0].attributes['source_sheet'] == 'Sheet 1'
    # ID should have spaces replaced with underscores
    assert metadata[0].id == 'sheet_1_row2'


def test_multisheet_embedded_images_fallback(monkeypatch, temp_dir):
    """Test that when text cell range is empty, loader falls back to XLSXImageExtractor."""
    # Create an Excel file where column E is empty
    xlsx_path = Path(temp_dir) / "empty_cells.xlsx"
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        data = {
            'A': ['HeaderA', 'a1'],
            'B': ['HeaderB', 'b1'],
            'C': ['HeaderC', 'c1'],
            'D': ['HeaderD', 'd1'],
            'E': ['HeaderE', '']
        }
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='Sheet1', index=False, header=False)
        
    dummy_path = Path(temp_dir) / "dummy.png"
    with open(dummy_path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n'
               b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
               b'\x08\x02\x00\x00\x00\x90wS\xde'
               b'\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01'
               b'\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')
               
    # Mock XLSXImageExtractor
    class MockExtractor:
        def __init__(self, *args, **kwargs):
            self.load_errors = [("test", "error message")]
        def extract_images_from_sheets(self, xlsx_path, sheet_range, cell_range):
            return [
                ImageMetadata(
                    id="sheet1_row2",
                    path=str(dummy_path),
                    prompt="dummy prompt",
                    model="dummy model"
                )
            ]
        def get_load_errors(self):
            return self.load_errors
            
    monkeypatch.setattr("src.xlsx_image_extractor.XLSXImageExtractor", MockExtractor)
    
    loader = CSVLoader()
    metadata = loader.load_metadata(
        str(xlsx_path),
        sheet_range="Sheet1:Sheet1",
        cell_range="E2:E2"
    )
    
    assert len(metadata) == 1
    assert metadata[0].id == "sheet1_row2"
    assert metadata[0].prompt == "dummy prompt"
    assert metadata[0].model == "dummy model"
    assert loader.get_load_errors() == [("test", "error message")]

