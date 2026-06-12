"""
XLSX embedded image extractor for IMAGE-LLP-VISION system.

This module handles extracting images that are embedded directly in Excel cells,
not just text paths. It extracts the images and saves them to disk, then
creates ImageMetadata records pointing to the saved files.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import logging
from zipfile import ZipFile
import shutil

from openpyxl import load_workbook
from openpyxl.drawing.image import Image as OpenpyxlImage
from PIL import Image

from src.data_models import ImageMetadata

logger = logging.getLogger(__name__)


class XLSXImageExtractor:
    """
    Extracts embedded images from Excel files.
    
    Handles images that are inserted directly into Excel cells rather than
    just text paths. Extracts images, saves them to disk, and creates
    ImageMetadata records.
    """
    
    def __init__(self, output_dir: str = "extracted_images"):
        """
        Initialize XLSXImageExtractor.
        
        Args:
            output_dir: Directory where extracted images will be saved
        """
        self.output_dir = Path(output_dir)
        self.load_errors: List[Tuple[str, str]] = []
    
    def extract_images_from_sheets(
        self, 
        xlsx_path: str, 
        sheet_range: str,
        cell_range: str
    ) -> List[ImageMetadata]:
        """
        Extract embedded images from specific cell range across multiple sheets.
        
        Args:
            xlsx_path: Path to XLSX file
            sheet_range: Sheet range (e.g., "Mermaid Metric:Underwatercity Metric")
            cell_range: Cell range (e.g., "E2:E11")
            
        Returns:
            List of ImageMetadata records for extracted images
        """
        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Parse sheet range
            sheet_start, sheet_end = sheet_range.split(':')
            sheet_start = sheet_start.strip()
            sheet_end = sheet_end.strip()
            
            # Parse cell range
            import re
            match = re.match(r'^([A-Z]+)(\d+):([A-Z]+)(\d+)$', cell_range)
            if not match:
                error_msg = f"Invalid cell range format: {cell_range}"
                logger.error(error_msg)
                self.load_errors.append((xlsx_path, error_msg))
                return []
            
            col_start, row_start, col_end, row_end = match.groups()
            
            if col_start != col_end:
                error_msg = f"Multi-column ranges not supported: {cell_range}"
                logger.error(error_msg)
                self.load_errors.append((xlsx_path, error_msg))
                return []
            
            column = col_start
            start_row = int(row_start)
            end_row = int(row_end)
            
            # Load workbook
            wb = load_workbook(xlsx_path)
            sheet_names = wb.sheetnames
            
            # Find sheet indices
            try:
                start_idx = sheet_names.index(sheet_start)
                end_idx = sheet_names.index(sheet_end)
            except ValueError:
                available_sheets = ', '.join(sheet_names)
                error_msg = (
                    f"Sheet not found. Requested: '{sheet_start}' to '{sheet_end}'. "
                    f"Available: {available_sheets}"
                )
                logger.error(error_msg)
                self.load_errors.append((xlsx_path, error_msg))
                return []
            
            if start_idx > end_idx:
                error_msg = f"Invalid sheet range: '{sheet_start}' comes after '{sheet_end}'"
                logger.error(error_msg)
                self.load_errors.append((xlsx_path, error_msg))
                return []
            
            # Process sheets
            metadata_records = []
            sheets_to_process = sheet_names[start_idx:end_idx + 1]
            
            logger.info(
                f"Extracting embedded images from {len(sheets_to_process)} sheets, "
                f"cells {cell_range}"
            )
            
            for sheet_name in sheets_to_process:
                sheet = wb[sheet_name]
                
                # Get all images in the sheet
                if not hasattr(sheet, '_images') or not sheet._images:
                    logger.warning(f"No embedded images found in sheet '{sheet_name}'")
                    continue
                
                # Iterate over all DrawingML image objects attached to this sheet
                for img_idx, image in enumerate(sheet._images):
                    # ── Resolve anchor type ───────────────────────────────────
                    # Excel stores image positions as DrawingML anchors.
                    # There are two anchor types:
                    #   TwoCellAnchor (_from/_to): image spans two cells.
                    #   OneCellAnchor (col/row):   image anchored to one cell.
                    # We only care about the top-left corner (_from for two-cell;
                    # col/row for one-cell) to determine which cell the image
                    # "lives in" for the purpose of column/row matching.
                    #
                    # IMPORTANT: openpyxl anchor coordinates are 0-indexed
                    # (col 0 = A, row 0 = row 1 in Excel notation). We add 1
                    # to row so that cell_row matches Excel's 1-based numbering
                    # and can be compared against start_row/end_row from the
                    # user-supplied cell_range string (e.g. "E2:E11").
                    anchor = image.anchor

                    if hasattr(anchor, '_from'):
                        # Two-cell anchor: use the "from" corner
                        cell_col = anchor._from.col
                        cell_row = anchor._from.row + 1  # 0-indexed → 1-indexed
                    elif hasattr(anchor, 'col') and hasattr(anchor, 'row'):
                        # One-cell anchor
                        cell_col = anchor.col
                        cell_row = anchor.row + 1         # 0-indexed → 1-indexed
                    else:
                        logger.warning(f"Unknown anchor type in sheet '{sheet_name}'")
                        continue
                    
                    # Convert column letter to index (A=0, B=1, E=4)
                    target_col_idx = ord(column) - ord('A')
                    
                    # Check if image is in our target column and row range
                    if cell_col == target_col_idx and start_row <= cell_row <= end_row:
                        # Extract and save image
                        try:
                            image_data = image._data()
                            
                            # Create filename
                            safe_sheet_name = sheet_name.replace(' ', '_').lower()
                            filename = f"{safe_sheet_name}_row{cell_row}_img{img_idx}.png"
                            output_path = self.output_dir / filename
                            
                            # Save image
                            with open(output_path, 'wb') as f:
                                f.write(image_data)
                            
                            # Verify it's a valid image
                            try:
                                with Image.open(output_path) as img:
                                    img.verify()
                            except Exception as e:
                                error_msg = f"Invalid image at {sheet_name}!{column}{cell_row}: {e}"
                                logger.warning(error_msg)
                                self.load_errors.append((f"{sheet_name}!{column}{cell_row}", error_msg))
                                output_path.unlink()  # Delete invalid file
                                continue
                            
                            # ── Upward scan for merged prompt cell (Column C) ────
                            # Same logic as in CSVLoader: Column C values are
                            # stored only on the first row of a merged region.
                            # Scan from cell_row upward to find the nearest
                            # non-empty value in Column C (column index 3).
                            prompt = None
                            for r in range(cell_row, 1, -1):
                                val = sheet.cell(row=r, column=3).value
                                if val is not None and str(val).strip() != '':
                                    prompt = str(val).strip()
                                    break
                            
                            # Get model from Column B (2)
                            model = None
                            val = sheet.cell(row=cell_row, column=2).value
                            if val is not None and str(val).strip() != '':
                                model = str(val).strip()

                            # Create ImageMetadata
                            image_id = f"{safe_sheet_name}_row{cell_row}"
                            metadata = ImageMetadata(
                                id=image_id,
                                path=str(output_path),
                                prompt=prompt,
                                model=model,
                                group=sheet_name,  # Use sheet name as group identifier
                                attributes={
                                    'source_sheet': sheet_name,
                                    'source_cell': f"{column}{cell_row}",
                                    'source_file': Path(xlsx_path).name,
                                    'extracted': True
                                }
                            )
                            metadata_records.append(metadata)
                            
                            logger.debug(
                                f"Extracted image from {sheet_name}!{column}{cell_row}: "
                                f"{image_id} -> {output_path}"
                            )
                            
                        except Exception as e:
                            error_msg = (
                                f"Failed to extract image from {sheet_name}!{column}{cell_row}: "
                                f"{type(e).__name__}: {str(e)}"
                            )
                            logger.warning(error_msg)
                            self.load_errors.append((f"{sheet_name}!{column}{cell_row}", error_msg))
            
            success_count = len(metadata_records)
            expected_count = len(sheets_to_process) * (end_row - start_row + 1)
            
            if success_count > 0:
                logger.info(
                    f"Image extraction completed: {success_count} images extracted, "
                    f"saved to {self.output_dir}"
                )
            else:
                logger.warning(
                    f"No images extracted from {len(sheets_to_process)} sheets. "
                    f"Expected up to {expected_count} images in cell range {cell_range}"
                )
            
            return metadata_records
            
        except Exception as e:
            error_msg = (
                f"Unexpected error extracting images from XLSX: {xlsx_path}. "
                f"Error: {type(e).__name__}: {str(e)}"
            )
            logger.error(error_msg)
            self.load_errors.append((xlsx_path, error_msg))
            return []
    
    def get_load_errors(self) -> List[Tuple[str, str]]:
        """
        Get list of all extraction errors encountered.
        
        Returns:
            List of (identifier, error_message) tuples
        """
        return self.load_errors.copy()
    
    def clear_errors(self) -> None:
        """Clear the accumulated errors."""
        self.load_errors.clear()
