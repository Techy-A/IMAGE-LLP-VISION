"""
Base scorer interface for pairwise image comparison.

This module defines the abstract base class that all scoring implementations
must inherit from, ensuring a consistent interface for comparing images.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import logging

from PIL.Image import Image as PILImage


logger = logging.getLogger(__name__)


class BaseScorer(ABC):
    """
    Abstract base class for image comparison scorers.
    
    All scorers must implement the compare() method that takes two images
    and returns a score in the range [0.0, 1.0], where:
    - score > 0.5 indicates preference for image A
    - score < 0.5 indicates preference for image B
    - score = 0.5 indicates no preference
    
    Scorers use sigmoid normalization on score differences to ensure
    outputs are within the valid range.
    """
    
    @abstractmethod
    def compare(self, image_a: PILImage, image_b: PILImage) -> float:
        """
        Compare two images and return a preference score.
        
        Args:
            image_a: First image to compare
            image_b: Second image to compare
        
        Returns:
            Score in range [0.0, 1.0] where >0.5 means image_a is preferred
        
        Raises:
            RuntimeError: If comparison fails
        """
        pass
    
    @abstractmethod
    def batch_compare(
        self,
        pairs: List[Tuple[PILImage, PILImage]]
    ) -> List[float]:
        """
        Compare multiple image pairs efficiently.
        
        Args:
            pairs: List of (image_a, image_b) tuples to compare
        
        Returns:
            List of scores in range [0.0, 1.0] for each pair
        
        Raises:
            RuntimeError: If batch comparison fails
        """
        pass
    
    def get_name(self) -> str:
        """
        Get the name of this scorer.
        
        Returns:
            Scorer name (class name by default)
        """
        return self.__class__.__name__
