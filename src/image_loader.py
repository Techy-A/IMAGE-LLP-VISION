"""
Image loading and preprocessing for IMAGE-LLP-VISION system.

This module handles loading images from disk with support for multiple formats
(PNG, JPG, JPEG, WEBP) and provides batch loading capabilities for efficient
processing.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import logging

from PIL import Image
from PIL import UnidentifiedImageError

# Type alias for PIL Images
PILImage = Image.Image

logger = logging.getLogger(__name__)


class ImageLoader:
    """
    Handles loading and preprocessing of image files.
    
    Supports PNG, JPG, JPEG, and WEBP formats with robust error handling
    for corrupted or invalid files.
    """
    
    # Supported image formats
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.webp'}
    
    def __init__(self):
        """Initialize ImageLoader."""
        self.load_errors: List[Tuple[str, str]] = []  # Track (path, error_message) pairs
    
    def load_image(self, path: str) -> Optional[PILImage]:
        """
        Load and preprocess a single image from disk.
        
        Args:
            path: File path to the image
            
        Returns:
            PIL Image if successful, None if loading fails
            
        Raises:
            No exceptions raised - errors are logged and tracked internally
        """
        try:
            # Validate path exists
            path_obj = Path(path)
            if not path_obj.exists():
                error_msg = f"Image file does not exist: {path}"
                logger.error(error_msg)
                self.load_errors.append((path, error_msg))
                return None
            
            if not path_obj.is_file():
                error_msg = f"Path is not a file: {path}"
                logger.error(error_msg)
                self.load_errors.append((path, error_msg))
                return None
            
            # Validate format is supported
            file_extension = path_obj.suffix.lower()
            if file_extension not in self.SUPPORTED_FORMATS:
                error_msg = (
                    f"Unsupported image format '{file_extension}' for file: {path}. "
                    f"Supported formats: {self.SUPPORTED_FORMATS}"
                )
                logger.error(error_msg)
                self.load_errors.append((path, error_msg))
                return None
            
            # Attempt to load image
            try:
                image = Image.open(path)
                
                # Convert to RGB if needed (handles RGBA, grayscale, etc.)
                if image.mode != 'RGB':
                    logger.debug(f"Converting image from {image.mode} to RGB: {path}")
                    image = image.convert('RGB')
                
                # Verify the image is actually loadable (sometimes corrupted files
                # can be opened but fail when trying to access pixel data)
                image.load()
                
                logger.debug(f"Successfully loaded image: {path} (size: {image.size})")
                return image
                
            except UnidentifiedImageError as e:
                error_msg = f"Cannot identify image file (possibly corrupted): {path}. Error: {str(e)}"
                logger.error(error_msg)
                self.load_errors.append((path, error_msg))
                return None
            
            except (IOError, OSError) as e:
                error_msg = f"IO error loading image: {path}. Error: {str(e)}"
                logger.error(error_msg)
                self.load_errors.append((path, error_msg))
                return None
            
            except Exception as e:
                error_msg = f"Unexpected error loading image: {path}. Error: {type(e).__name__}: {str(e)}"
                logger.error(error_msg)
                self.load_errors.append((path, error_msg))
                return None
                
        except Exception as e:
            # Catch-all for any unexpected errors during path validation
            error_msg = f"Unexpected error processing path: {path}. Error: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            self.load_errors.append((path, error_msg))
            return None
    
    def batch_load(self, paths: List[str]) -> List[Optional[PILImage]]:
        """
        Efficiently load multiple images in batch.
        
        Args:
            paths: List of file paths to images
            
        Returns:
            List of PIL Images (or None for failed loads) in the same order as input paths
        """
        if not paths:
            logger.warning("batch_load called with empty paths list")
            return []
        
        logger.info(f"Batch loading {len(paths)} images")
        
        images = []
        success_count = 0
        
        for path in paths:
            image = self.load_image(path)
            images.append(image)
            if image is not None:
                success_count += 1
        
        failure_count = len(paths) - success_count
        if failure_count > 0:
            logger.warning(
                f"Batch load completed: {success_count}/{len(paths)} successful, "
                f"{failure_count} failed"
            )
        else:
            logger.info(f"Batch load completed successfully: {success_count}/{len(paths)} images loaded")
        
        return images
    
    def get_load_errors(self) -> List[Tuple[str, str]]:
        """
        Get list of all load errors encountered.
        
        Returns:
            List of (path, error_message) tuples
        """
        return self.load_errors.copy()
    
    def clear_errors(self) -> None:
        """Clear the accumulated load errors."""
        self.load_errors.clear()
    
    @staticmethod
    def is_supported_format(path: str) -> bool:
        """
        Check if a file path has a supported image format extension.
        
        Args:
            path: File path to check
            
        Returns:
            True if format is supported, False otherwise
        """
        path_obj = Path(path)
        return path_obj.suffix.lower() in ImageLoader.SUPPORTED_FORMATS
