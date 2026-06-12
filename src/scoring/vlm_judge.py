"""
VLMJudge metric for Vision-Language Model judgment-based image scoring.

VLMJudge uses InstructBLIP to directly compare and judge image quality
through detailed prompts that ask the model to assess and compare images.
This scorer MUST receive an InstructBLIP instance via dependency injection -
it does NOT load its own model to avoid duplicate loading.
"""

import logging
import math
from typing import List, Tuple, Optional

from PIL.Image import Image as PILImage

from src.scoring.base_scorer import BaseScorer
from src.models.instructblip_adapter import InstructBLIPAdapter


logger = logging.getLogger(__name__)


class VLMJudge(BaseScorer):
    """
    VLMJudge quality assessment via direct model judgment.
    
    Uses InstructBLIP to make holistic judgments about image quality
    based on comprehensive evaluation prompts. Unlike VQAScore which
    asks specific yes/no questions, VLMJudge asks for detailed assessments.
    
    CRITICAL: This scorer accepts an InstructBLIPAdapter instance via
    dependency injection. It MUST NOT load its own model. This instance
    is shared with VQAScore to avoid duplicate model loading.
    
    Judgment prompts used:
    - Overall quality rating requests
    - Aesthetic appeal assessments
    - Technical quality evaluations
    """
    
    # Judgment prompts for quality assessment
    JUDGMENT_PROMPTS = [
        "Rate the overall quality of this image on a scale from 1 to 10, where 10 is excellent. Only provide the number.",
        "How aesthetically appealing is this image? Rate from 1 to 10. Only provide the number.",
        "Evaluate the technical quality (sharpness, lighting, composition) of this image from 1 to 10. Only provide the number.",
    ]
    
    def __init__(self, instructblip_model: InstructBLIPAdapter):
        """
        Initialize VLMJudge with shared InstructBLIP instance.
        
        Args:
            instructblip_model: Pre-loaded InstructBLIP adapter instance
                                (shared with VQAScore)
        
        Raises:
            ValueError: If instructblip_model is None
        """
        if instructblip_model is None:
            raise ValueError(
                "VLMJudge requires a loaded InstructBLIP model instance. "
                "Pass via dependency injection to avoid duplicate loading."
            )
        
        self.model = instructblip_model
        logger.info("VLMJudge initialized with shared InstructBLIP instance")
    
    def compare(self, image_a: PILImage, image_b: PILImage) -> float:
        """
        Compare two images using VLM judgment-based assessment.
        
        Args:
            image_a: First image to compare
            image_b: Second image to compare
        
        Returns:
            Score in [0.0, 1.0] where >0.5 means image_a has better quality
        
        Raises:
            RuntimeError: If comparison fails
        """
        try:
            # Score both images
            score_a = self._score_image(image_a)
            score_b = self._score_image(image_b)
            
            # Compute difference
            score_diff = score_a - score_b
            
            # Apply sigmoid normalization to get value in [0.0, 1.0]
            normalized_score = self._sigmoid(score_diff)
            
            # DEBUG: Log raw scores and final result
            logger.debug(
                f"VLMJudge RAW: score_a={score_a:.4f}, score_b={score_b:.4f}, "
                f"diff={score_diff:.4f}, sigmoid={normalized_score:.4f}"
            )
            
            return float(normalized_score)
            
        except Exception as e:
            logger.error(f"VLMJudge comparison failed: {e}")
            raise RuntimeError(f"VLMJudge comparison failed: {e}") from e
    
    def batch_compare(
        self,
        pairs: List[Tuple[PILImage, PILImage]]
    ) -> List[float]:
        """
        Compare multiple image pairs using VLMJudge.
        
        Args:
            pairs: List of (image_a, image_b) tuples
        
        Returns:
            List of scores in [0.0, 1.0] for each pair
        
        Raises:
            RuntimeError: If batch comparison fails
        """
        try:
            results = []
            
            # Process each pair
            # TODO: Could optimize with true batch processing
            for image_a, image_b in pairs:
                score = self.compare(image_a, image_b)
                results.append(score)
            
            return results
            
        except Exception as e:
            logger.error(f"VLMJudge batch comparison failed: {e}")
            raise RuntimeError(f"VLMJudge batch comparison failed: {e}") from e
    
    def _score_image(self, image: PILImage) -> float:
        """
        Score a single image using VLM judgment prompts.
        
        Args:
            image: Image to score
        
        Returns:
            Quality score normalized to [0.0, 10.0] range
        """
        scores = []
        responses_debug = []
        
        # Ask each judgment question
        for prompt in self.JUDGMENT_PROMPTS:
            try:
                # Query InstructBLIP
                response = self.model.query_image(
                    image=image,
                    prompt=prompt,
                    max_new_tokens=20  # Enough for "10" plus explanation
                )
                
                # Parse response for numeric rating
                rating = self._parse_rating(response)
                scores.append(rating)
                responses_debug.append(f"{prompt[:30]}... -> '{response}' -> {rating:.1f}")
                
            except Exception as e:
                logger.warning(
                    f"VLM judgment failed for prompt '{prompt[:30]}...': {e}. "
                    f"Using neutral score."
                )
                scores.append(5.0)  # Neutral score (middle of 1-10 scale)
                responses_debug.append(f"{prompt[:30]}... -> ERROR: {e}")
        
        # Average scores across prompts
        avg_score = sum(scores) / len(scores) if scores else 5.0
        
        # DEBUG: Log individual responses
        logger.debug(f"VLMJudge responses: {'; '.join(responses_debug)} -> avg={avg_score:.4f}")
        
        return avg_score
    
    def _parse_rating(self, response: str) -> float:
        """
        Parse VLM response to extract numeric rating.
        
        Looks for numbers in the response and extracts the rating.
        Expects ratings in 1-10 scale.
        
        Args:
            response: Model's text response
        
        Returns:
            Rating in [1.0, 10.0] range (defaults to 5.0 if parsing fails)
        """
        import re
        
        # Extract all numbers from response
        numbers = re.findall(r'\d+\.?\d*', response)
        
        if not numbers:
            # No number found, use neutral
            logger.debug(f"No rating found in response: '{response}'. Using neutral 5.0")
            return 5.0
        
        # Take first number as the rating
        try:
            rating = float(numbers[0])
            
            # Clamp to valid range [1, 10]
            rating = max(1.0, min(10.0, rating))
            
            return rating
            
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse rating from '{response}': {e}. Using neutral 5.0")
            return 5.0
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        """
        Apply sigmoid function to normalize score difference.

        VLMJudge scores are on a 1–10 integer scale, so the difference x
        between two images lies in [-9, +9]. Dividing by a scale factor
        before sigmoid maps the range to [0, 1].

        Amplitude table for scale = 1.5:

            |  diff  |  sigmoid  |  interpretation  |
            |--------|-----------|------------------|
            |   0    |  0.500    |  tie             |
            |  +1    |  0.731    |  clear win       |
            |  +2    |  0.880    |  strong win      |
            |  +3    |  0.953    |  decisive        |
            |  +4    |  0.982    |  very decisive   |

        At scale=3.0 a 1-point difference only maps to 0.596 (barely
        above tie), giving poor discrimination when models tend to rate
        most images between 6–8. Halving the scale to 1.5 makes single-
        point differences meaningful.

        Args:
            x: Score difference (typically in range [-9, 9])

        Returns:
            Normalized value in [0.0, 1.0]
        """
        # Amplify: divide by 1.5 (not 3.0) for better sensitivity
        scaled_x = x / 1.5
        return 1.0 / (1.0 + math.exp(-scaled_x))
