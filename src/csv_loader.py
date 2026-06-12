"""
CSV metadata loading for IMAGE-LLP-VISION system.

This module handles loading image metadata from CSV files, validating paths,
and handling errors gracefully for missing or invalid files.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import logging

import pandas as pd

from src.data_models import ImageMetadata

logger = logging.getLogger(__name__)


class CSVLoader:
    """
    Handles loading image metadata from CSV files.
    
    Parses CSV files containing image metadata and creates ImageMetadata records.
    Validates that required columns exist and image paths are valid. Tracks
    errors for missing or invalid files.
    """
    
    # Required columns in CSV
    REQUIRED_COLUMNS = {'id', 'path'}
    
    # Optional columns that map to ImageMetadata fields
    OPTIONAL_COLUMNS = {'prompt', 'model'}
    
    def __init__(self):
        """Initialize CSVLoader."""
        self.load_errors: List[Tuple[str, str]] = []  # Track (identifier, error_message) pairs
    
    def load_metadata(self, csv_path: str, sheet_range: Optional[str] = None, 
                      cell_range: Optional[str] = None) -> List[ImageMetadata]:
        """
        Load image metadata from a CSV or XLSX file.
        
        For CSV files:
        - Must contain 'id' and 'path' columns
        - Optional columns: 'prompt', 'model'
        - Additional columns stored in attributes dictionary
        
        For XLSX files:
        - Standard mode (sheet_range=None): Reads first sheet, requires 'id' and 'path' columns
        - Multi-sheet mode (sheet_range specified): Reads from specific cell range across sheets
          Example: sheet_range="Sheet1:Sheet5", cell_range="E2:E11"
          This reads cells E2-E11 from Sheet1 through Sheet5
        
        Args:
            csv_path: Path to CSV/XLSX file containing image metadata
            sheet_range: Optional sheet range for multi-sheet XLSX (e.g., "Sheet1:Sheet5")
            cell_range: Optional cell range for multi-sheet mode (e.g., "E2:E11")
            
        Returns:
            List of ImageMetadata records for successfully validated images
            
        Raises:
            No exceptions raised - errors are logged and tracked internally
        """
        try:
            # Validate file exists
            csv_path_obj = Path(csv_path)
            if not csv_path_obj.exists():
                error_msg = f"File does not exist: {csv_path}"
                logger.error(error_msg)
                self.load_errors.append((csv_path, error_msg))
                return []
            
            if not csv_path_obj.is_file():
                error_msg = f"Path is not a file: {csv_path}"
                logger.error(error_msg)
                self.load_errors.append((csv_path, error_msg))
                return []
            
            file_ext = csv_path_obj.suffix.lower()
            
            # Multi-sheet XLSX mode
            if file_ext in ['.xlsx', '.xls'] and sheet_range and cell_range:
                return self._load_multisheet_xlsx(csv_path, sheet_range, cell_range)
            
            # Load file (auto-detect CSV vs XLSX based on extension)
            try:
                if file_ext in ['.xlsx', '.xls']:
                    # Excel file - single sheet mode
                    df = pd.read_excel(csv_path, engine='openpyxl' if file_ext == '.xlsx' else None, 
                                      keep_default_na=False)
                    logger.info(f"Loaded Excel file with {len(df)} rows: {csv_path}")
                else:
                    # CSV file (default)
                    df = pd.read_csv(csv_path, keep_default_na=False)  # Don't convert empty strings to NaN
                    logger.info(f"Loaded CSV file with {len(df)} rows: {csv_path}")
                    
            except Exception as e:
                error_msg = f"Failed to parse file: {csv_path}. Error: {type(e).__name__}: {str(e)}"
                logger.error(error_msg)
                self.load_errors.append((csv_path, error_msg))
                return []
            
            # Validate required columns exist
            missing_columns = self.REQUIRED_COLUMNS - set(df.columns)
            if missing_columns:
                error_msg = (
                    f"File missing required columns: {missing_columns}. "
                    f"Required columns: {self.REQUIRED_COLUMNS}. "
                    f"Found columns: {set(df.columns)}"
                )
                logger.error(error_msg)
                self.load_errors.append((csv_path, error_msg))
                return []
            
            # Process each row and create ImageMetadata records
            metadata_records = []
            for idx, row in df.iterrows():
                record = self._parse_row(row, idx)
                if record is not None:
                    metadata_records.append(record)
            
            success_count = len(metadata_records)
            failure_count = len(df) - success_count
            
            if failure_count > 0:
                logger.warning(
                    f"CSV load completed: {success_count}/{len(df)} records valid, "
                    f"{failure_count} failed validation"
                )
            else:
                logger.info(
                    f"CSV load completed successfully: {success_count}/{len(df)} records loaded"
                )
            
            return metadata_records
            
        except Exception as e:
            # Catch-all for any unexpected errors
            error_msg = f"Unexpected error loading CSV: {csv_path}. Error: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            self.load_errors.append((csv_path, error_msg))
            return []
    
    def _load_multisheet_xlsx(self, xlsx_path: str, sheet_range: str, 
                              cell_range: str) -> List[ImageMetadata]:
        """
        Load image metadata from multiple sheets in an XLSX file.
        
        Reads from a specific cell range across multiple sheets. Each cell value
        is treated as an image path. Image IDs are auto-generated.
        
        Args:
            xlsx_path: Path to XLSX file
            sheet_range: Sheet range (e.g., "Sheet1:Sheet5" or "Sheet 1:Sheet 5")
            cell_range: Cell range (e.g., "E2:E11")
            
        Returns:
            List of ImageMetadata records
        """
        try:
            # Parse sheet range
            sheet_start, sheet_end = sheet_range.split(':')
            sheet_start = sheet_start.strip()
            sheet_end = sheet_end.strip()
            
            # Parse cell range (e.g., "E2:E11" -> column=E, start_row=2, end_row=11)
            import re
            match = re.match(r'^([A-Z]+)(\d+):([A-Z]+)(\d+)$', cell_range)
            if not match:
                error_msg = f"Invalid cell range format: {cell_range}. Expected format: 'E2:E11'"
                logger.error(error_msg)
                self.load_errors.append((xlsx_path, error_msg))
                return []
            
            col_start, row_start, col_end, row_end = match.groups()
            
            # Validate single column range
            if col_start != col_end:
                error_msg = f"Multi-column ranges not supported: {cell_range}. Use single column (e.g., 'E2:E11')"
                logger.error(error_msg)
                self.load_errors.append((xlsx_path, error_msg))
                return []
            
            column = col_start
            start_row = int(row_start)
            end_row = int(row_end)
            
            # Load Excel file and get sheet names
            xl_file = pd.ExcelFile(xlsx_path, engine='openpyxl')
            sheet_names = xl_file.sheet_names
            
            # Find sheet indices
            try:
                start_idx = sheet_names.index(sheet_start)
                end_idx = sheet_names.index(sheet_end)
            except ValueError as e:
                available_sheets = ', '.join(sheet_names)
                error_msg = (
                    f"Sheet not found in {xlsx_path}. "
                    f"Requested: '{sheet_start}' to '{sheet_end}'. "
                    f"Available sheets: {available_sheets}"
                )
                logger.error(error_msg)
                self.load_errors.append((xlsx_path, error_msg))
                return []
            
            if start_idx > end_idx:
                error_msg = (
                    f"Invalid sheet range: '{sheet_start}' comes after '{sheet_end}' "
                    f"in the workbook"
                )
                logger.error(error_msg)
                self.load_errors.append((xlsx_path, error_msg))
                return []
            
            # Process sheets in range
            metadata_records = []
            sheets_to_process = sheet_names[start_idx:end_idx + 1]
            
            logger.info(
                f"Loading multi-sheet XLSX: {len(sheets_to_process)} sheets, "
                f"cells {cell_range} from each sheet"
            )
            
            for sheet_name in sheets_to_process:
                try:
                    # Read the entire sheet without header to preserve 1-based indexing
                    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, 
                                     engine='openpyxl', keep_default_na=False, header=None)
                    
                    # Convert column letter to index (A=0, B=1, etc.)
                    col_idx = ord(column) - ord('A')
                    
                    # Check if column exists
                    if col_idx >= len(df.columns):
                        error_msg = (
                            f"Column {column} not found in sheet '{sheet_name}'. "
                            f"Sheet has {len(df.columns)} columns"
                        )
                        logger.warning(error_msg)
                        self.load_errors.append((f"{sheet_name}:{column}", error_msg))
                        continue
                    
                    # Extract values from the specified cell range
                    # Note: DataFrame is 0-indexed, but Excel rows are 1-indexed
                    # Row 2 in Excel = index 1 in DataFrame (row 1 is header at index 0)
                    df_start_row = start_row - 1  # Convert Excel row to DataFrame index
                    df_end_row = end_row - 1      # Inclusive
                    
                    if df_start_row < 0 or df_end_row >= len(df):
                        error_msg = (
                            f"Row range {start_row}:{end_row} out of bounds for sheet '{sheet_name}'. "
                            f"Sheet has {len(df)} data rows (plus header)"
                        )
                        logger.warning(error_msg)
                        self.load_errors.append((f"{sheet_name}", error_msg))
                        continue
                    
                    # Get column name from index
                    col_name = df.columns[col_idx]
                    
                    # Extract cell values
                    for row_idx in range(df_start_row, df_end_row + 1):
                        image_path = df.iloc[row_idx, col_idx]
                        
                        # Skip empty cells
                        if pd.isna(image_path) or str(image_path).strip() == '':
                            continue
                        
                        image_path = str(image_path).strip()
                        
                        # Generate unique ID: sheet_name + row number
                        # Sanitize sheet name for ID (replace spaces with underscores)
                        safe_sheet_name = sheet_name.replace(' ', '_').lower()
                        excel_row_num = row_idx + 1  # Convert back to Excel row number
                        image_id = f"{safe_sheet_name}_row{excel_row_num}"
                        
                        # ── Upward-scan for merged-cell prompt (Column C) ─────────
                        # In the XLSX the prompt in Column C often spans multiple
                        # rows via a merged cell. When openpyxl reads merged cells
                        # only the top-left cell contains the value; all other cells
                        # in the merge read as None. To recover the prompt for any
                        # image row we scan upward from the current row until we
                        # find a non-empty value in Column C (index 2).
                        prompt = None
                        if len(df.columns) > 2:
                            for r in range(row_idx, 0, -1):
                                val = df.iloc[r, 2]
                                if val is not None and str(val).strip() != '':
                                    prompt = str(val).strip()
                                    break

                        # Model name is in Column B (index 1) on the same row
                        # (not merged, so no upward scan needed).
                        model = None
                        if len(df.columns) > 1:
                            val = df.iloc[row_idx, 1]
                            if val is not None and str(val).strip() != '':
                                model = str(val).strip()

                        # Create ImageMetadata
                        try:
                            metadata = ImageMetadata(
                                id=image_id,
                                path=image_path,
                                prompt=prompt,
                                model=model,
                                group=sheet_name,  # Use sheet name as group identifier
                                attributes={
                                    'source_sheet': sheet_name,
                                    'source_cell': f"{column}{excel_row_num}",
                                    'source_file': Path(xlsx_path).name
                                }
                            )
                            metadata_records.append(metadata)
                            logger.debug(
                                f"Loaded image from {sheet_name}!{column}{excel_row_num}: "
                                f"{image_id} -> {image_path}"
                            )
                            
                        except ValueError as e:
                            error_msg = (
                                f"Failed to create metadata for {sheet_name}!{column}{excel_row_num}: "
                                f"{str(e)}"
                            )
                            logger.warning(error_msg)
                            self.load_errors.append((image_id, error_msg))
                
                except Exception as e:
                    error_msg = (
                        f"Error processing sheet '{sheet_name}': "
                        f"{type(e).__name__}: {str(e)}"
                    )
                    logger.error(error_msg)
                    self.load_errors.append((sheet_name, error_msg))
            
            success_count = len(metadata_records)
            expected_count = len(sheets_to_process) * (end_row - start_row + 1)
            failure_count = expected_count - success_count
            
            if failure_count > 0:
                logger.warning(
                    f"Multi-sheet XLSX load completed: {success_count}/{expected_count} "
                    f"records valid, {failure_count} failed or empty"
                )
            else:
                logger.info(
                    f"Multi-sheet XLSX load completed successfully: "
                    f"{success_count}/{expected_count} records loaded"
                )
            # ── Embedded-image fallback ────────────────────────────────────────
            # Some XLSX files store images embedded directly in cells (as
            # DrawingML objects) rather than as text file-path references.
            # When no text paths were found in the cell range, try extracting
            # embedded images via XLSXImageExtractor, which reads the XLSX zip
            # archive's xl/media/ directory and matches anchors to cell ranges.
            if not metadata_records:
                logger.info("No text image paths found in cell range. Attempting to extract embedded images...")
                try:
                    from src.xlsx_image_extractor import XLSXImageExtractor
                    extractor = XLSXImageExtractor()
                    metadata_records = extractor.extract_images_from_sheets(
                        xlsx_path, sheet_range, cell_range
                    )
                    # Propagate any errors from extractor
                    for identifier, error_msg in extractor.get_load_errors():
                        self.load_errors.append((identifier, error_msg))
                    success_count = len(metadata_records)
                except Exception as ext_err:
                    error_msg = f"Failed to run embedded image extractor: {ext_err}"
                    logger.error(error_msg)
                    self.load_errors.append((xlsx_path, error_msg))

            return metadata_records
            
        except Exception as e:
            error_msg = (
                f"Unexpected error loading multi-sheet XLSX: {xlsx_path}. "
                f"Error: {type(e).__name__}: {str(e)}"
            )
            logger.error(error_msg)
            self.load_errors.append((xlsx_path, error_msg))
            return []
    
    def _parse_row(self, row: pd.Series, row_idx: int) -> Optional[ImageMetadata]:
        """
        Parse a single CSV row into ImageMetadata.
        
        Args:
            row: Pandas Series representing CSV row
            row_idx: Row index for error reporting
            
        Returns:
            ImageMetadata if parsing succeeds, None otherwise
        """
        try:
            # Extract required fields
            image_id = str(row['id']).strip()
            image_path = str(row['path']).strip()
            
            # Validate required fields are not empty
            if not image_id:
                error_msg = f"Row {row_idx}: Empty image ID"
                logger.error(error_msg)
                self.load_errors.append((f"row_{row_idx}", error_msg))
                return None
            
            if not image_path:
                error_msg = f"Row {row_idx} (ID: {image_id}): Empty image path"
                logger.error(error_msg)
                self.load_errors.append((image_id, error_msg))
                return None
            
            # Extract optional fields
            prompt = None
            if 'prompt' in row.index and pd.notna(row['prompt']) and row['prompt'] != '':
                prompt = str(row['prompt']).strip()
            
            model = None
            if 'model' in row.index and pd.notna(row['model']) and row['model'] != '':
                model = str(row['model']).strip()
            
            # Extract additional attributes (any column not in required/optional)
            attributes = {}
            known_columns = self.REQUIRED_COLUMNS | self.OPTIONAL_COLUMNS
            for col in row.index:
                if col not in known_columns and pd.notna(row[col]) and row[col] != '':
                    # Store additional attributes as strings for consistency
                    attributes[col] = str(row[col])
            
            # ── group field for CSV mode ──────────────────────────────────────
            # In XLSX mode, group = sheet name (set by _load_multisheet_xlsx).
            # In plain CSV mode there is no sheet concept, so we use the
            # prompt text as a group identifier. This means all images with
            # the same prompt will be compared against each other (but NOT
            # against images with a different prompt), which matches the
            # intended within-group-only comparison semantics.
            try:
                metadata = ImageMetadata(
                    id=image_id,
                    path=image_path,
                    prompt=prompt,
                    model=model,
                    group=prompt,  # Use prompt as group identifier for CSV mode
                    attributes=attributes
                )
                logger.debug(f"Successfully parsed row {row_idx} (ID: {image_id})")
                return metadata
                
            except ValueError as e:
                # ImageMetadata validation failed (path doesn't exist, wrong format, etc.)
                error_msg = f"Row {row_idx} (ID: {image_id}): {str(e)}"
                logger.error(error_msg)
                self.load_errors.append((image_id, error_msg))
                return None
                
        except KeyError as e:
            # Missing required column (shouldn't happen if we validated above)
            error_msg = f"Row {row_idx}: Missing column {str(e)}"
            logger.error(error_msg)
            self.load_errors.append((f"row_{row_idx}", error_msg))
            return None
            
        except Exception as e:
            # Catch-all for any unexpected errors during row parsing
            try:
                image_id = str(row['id']) if 'id' in row.index else f"row_{row_idx}"
            except:
                image_id = f"row_{row_idx}"
            
            error_msg = f"Unexpected error parsing row {row_idx} (ID: {image_id}): {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            self.load_errors.append((image_id, error_msg))
            return None
    
    def get_load_errors(self) -> List[Tuple[str, str]]:
        """
        Get list of all load errors encountered.
        
        Returns:
            List of (identifier, error_message) tuples
        """
        return self.load_errors.copy()
    
    def clear_errors(self) -> None:
        """Clear the accumulated load errors."""
        self.load_errors.clear()
    
    @staticmethod
    def validate_csv_format(csv_path: str) -> Tuple[bool, str]:
        """
        Validate that a CSV file has the required format without loading all data.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Tuple of (is_valid, error_message). error_message is empty if valid.
        """
        try:
            csv_path_obj = Path(csv_path)
            
            if not csv_path_obj.exists():
                return False, f"CSV file does not exist: {csv_path}"
            
            if not csv_path_obj.is_file():
                return False, f"CSV path is not a file: {csv_path}"
            
            # Read just the header
            df = pd.read_csv(csv_path, nrows=0)
            
            missing_columns = CSVLoader.REQUIRED_COLUMNS - set(df.columns)
            if missing_columns:
                return False, (
                    f"CSV missing required columns: {missing_columns}. "
                    f"Required: {CSVLoader.REQUIRED_COLUMNS}"
                )
            
            return True, ""
            
        except Exception as e:
            return False, f"Error validating CSV: {type(e).__name__}: {str(e)}"
