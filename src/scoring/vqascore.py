"""
VQAScore metric for Visual Question Answering-based image scoring.

VQAScore uses InstructBLIP to ask questions about image quality and
prompt fidelity, then scores based on the model's responses. This scorer
MUST receive an InstructBLIP instance via dependency injection - it does
NOT load its own model to avoid duplicate loading.
"""

import logging
import math
from typing import List, Tuple, Optional

from PIL.Image import Image as PILImage

from src.scoring.base_scorer import BaseScorer
from src.models.instructblip_adapter import InstructBLIPAdapter


logger = logging.getLogger(__name__)


class VQAScore(BaseScorer):
    """
    VQAScore quality assessment via visual question answering.

    Uses InstructBLIP to ask quality-related questions about images,
    then scores based on response confidence and quality indicators.

    When a text prompt is provided, additional prompt-fidelity questions
    are asked (e.g. "Does this image accurately depict <prompt>?") so
    the scorer evaluates both quality *and* prompt alignment — similar
    in spirit to CLIPScore but using a generative VLM.

    CRITICAL: This scorer accepts an InstructBLIPAdapter instance via
    dependency injection. It MUST NOT load its own model. This instance
    is shared with VLMJudge to avoid duplicate model loading.

    Quality questions asked (always):
    - "Is this image of high quality? Answer yes or no."
    - "Does this image have good composition? Answer yes or no."
    - "Is this image aesthetically pleasing? Answer yes or no."

    Prompt-fidelity question (when prompt is provided):
    - "Does this image accurately depict '<prompt>'? Answer yes or no."
    - "Is the subject of this image a '<prompt>'? Answer yes or no."
    """

    # Generic quality prompts — used regardless of whether a text prompt is given
    QUALITY_PROMPTS = [
        "Is this image of high quality? Answer yes or no.",
        "Does this image have good composition? Answer yes or no.",
        "Is this image aesthetically pleasing? Answer yes or no.",
    ]

    def __init__(self, instructblip_model: InstructBLIPAdapter):
        """
        Initialize VQAScore with shared InstructBLIP instance.

        Args:
            instructblip_model: Pre-loaded InstructBLIP adapter instance
                                (shared with VLMJudge)

        Raises:
            ValueError: If instructblip_model is None
        """
        if instructblip_model is None:
            raise ValueError(
                "VQAScore requires a loaded InstructBLIP model instance. "
                "Pass via dependency injection to avoid duplicate loading."
            )

        self.model = instructblip_model
        logger.info("VQAScore initialized with shared InstructBLIP instance")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        image_a: PILImage,
        image_b: PILImage,
        prompt_a: Optional[str] = None,
        prompt_b: Optional[str] = None,
    ) -> float:
        """
        Compare two images using VQA-based quality (and optionally prompt
        fidelity) assessment.

        Args:
            image_a: First image to compare
            image_b: Second image to compare
            prompt_a: Optional text prompt that image_a was generated from.
                      When provided, prompt-fidelity questions are added.
            prompt_b: Optional text prompt that image_b was generated from.

        Returns:
            Score in [0.0, 1.0] where >0.5 means image_a is preferred

        Raises:
            RuntimeError: If comparison fails
        """
        try:
            score_a = self._score_image(image_a, prompt=prompt_a)
            score_b = self._score_image(image_b, prompt=prompt_b)

            score_diff = score_a - score_b
            normalized_score = self._sigmoid(score_diff)

            logger.debug(
                f"VQAScore RAW: score_a={score_a:.4f}, score_b={score_b:.4f}, "
                f"diff={score_diff:.4f}, sigmoid={normalized_score:.4f}"
            )

            return float(normalized_score)

        except Exception as e:
            logger.error(f"VQAScore comparison failed: {e}")
            raise RuntimeError(f"VQAScore comparison failed: {e}") from e

    def batch_compare(
        self,
        pairs: List[Tuple[PILImage, PILImage]],
        prompts: Optional[List[Tuple[Optional[str], Optional[str]]]] = None,
    ) -> List[float]:
        """
        Compare multiple image pairs using VQAScore.

        Unlike the old implementation this scores all images in one pass
        over the pair list, reusing per-image scores when the same image
        appears in multiple pairs.

        Args:
            pairs: List of (image_a, image_b) tuples
            prompts: Optional list of (prompt_a, prompt_b) tuples,
                     one per pair.  If omitted, generic quality questions
                     are used for every pair.

        Returns:
            List of scores in [0.0, 1.0] for each pair

        Raises:
            RuntimeError: If batch comparison fails
            ValueError: If prompts length does not match pairs length
        """
        if prompts is not None and len(prompts) != len(pairs):
            raise ValueError(
                f"prompts length ({len(prompts)}) must match pairs length "
                f"({len(pairs)})"
            )

        try:
            # Score each unique image exactly once.
            # Key: (id(image), prompt) so that the same PIL object scored
            # against different prompts is treated as distinct entries.
            score_cache: dict = {}

            def _cached_score(image: PILImage, prompt: Optional[str]) -> float:
                key = (id(image), prompt)
                if key not in score_cache:
                    score_cache[key] = self._score_image(image, prompt=prompt)
                return score_cache[key]

            results = []
            for idx, (image_a, image_b) in enumerate(pairs):
                pa, pb = (prompts[idx] if prompts is not None else (None, None))
                score_a = _cached_score(image_a, pa)
                score_b = _cached_score(image_b, pb)
                results.append(float(self._sigmoid(score_a - score_b)))

            return results

        except Exception as e:
            logger.error(f"VQAScore batch comparison failed: {e}")
            raise RuntimeError(f"VQAScore batch comparison failed: {e}") from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_image(self, image: PILImage, prompt: Optional[str] = None) -> float:
        """
        Score a single image using VQA quality prompts.

        When *prompt* is given two extra prompt-fidelity questions are
        appended so the score reflects both aesthetics and how well the
        image matches the intended subject.

        Args:
            image: Image to score
            prompt: Optional generation prompt for fidelity questions

        Returns:
            Quality score in [0.0, 1.0] (higher is better)
        """
        questions = list(self.QUALITY_PROMPTS)

        # Add prompt-aware fidelity questions when a prompt is available
        if prompt and prompt.strip():
            safe_prompt = prompt.strip()
            questions.append(
                f'Does this image accurately depict "{safe_prompt}"? '
                f"Answer yes or no."
            )
            questions.append(
                f'Is the main subject of this image "{safe_prompt}"? '
                f"Answer yes or no."
            )

        scores: List[float] = []
        responses_debug: List[str] = []

        for question in questions:
            try:
                response = self.model.query_image(
                    image=image,
                    prompt=question,
                    max_new_tokens=10,  # Short answers (yes/no)
                )

                quality_score = self._parse_response(response)
                scores.append(quality_score)
                responses_debug.append(
                    f"{question[:40]}... -> '{response}' -> {quality_score:.3f}"
                )

            except Exception as e:
                logger.warning(
                    f"VQA query failed for question '{question[:40]}...': {e}. "
                    f"Using neutral score."
                )
                scores.append(0.5)  # Neutral score on failure
                responses_debug.append(f"{question[:40]}... -> ERROR: {e}")

        avg_score = sum(scores) / len(scores) if scores else 0.5

        logger.debug(
            f"VQAScore responses: {'; '.join(responses_debug)} -> avg={avg_score:.4f}"
        )

        return avg_score

    @staticmethod
    def _parse_response(response: str) -> float:
        """
        Parse a VQA yes/no response into a quality score in [0.0, 1.0].

        Strategy
        --------
        1. Strip the response and look for the *first clear signal word*
           (yes/no) rather than counting all occurrences, which prevents
           double-counting in responses like "No, it is not good quality"
           that contain both positive and negative words.
        2. Positive qualifiers (e.g. "very yes", "absolutely") push the
           score toward 1.0; hedges (e.g. "somewhat", "maybe") pull it
           back toward 0.5.
        3. Completely ambiguous / empty responses return 0.5.

        Args:
            response: Model's text response

        Returns:
            Score in [0.0, 1.0]
        """
        text = response.lower().strip()

        if not text:
            return 0.5

        # Hedge words that indicate lower confidence regardless of polarity
        HEDGES = {"somewhat", "maybe", "perhaps", "possibly", "sort of",
                  "kind of", "not sure", "unclear", "partially"}
        # Strong-positive qualifiers
        STRONG_POS = {"very", "absolutely", "definitely", "certainly",
                      "clearly", "highly", "extremely", "excellent",
                      "great", "outstanding"}

        is_hedged = any(h in text for h in HEDGES)
        is_strong = any(s in text for s in STRONG_POS)

        # Determine primary polarity: search for the first yes/no token
        # Use word-boundary style check so "notable" doesn't match "no"
        import re
        yes_match = re.search(r'\byes\b', text)
        no_match = re.search(r'\bno\b', text)

        if yes_match and (not no_match or yes_match.start() < no_match.start()):
            # Positive response
            if is_strong:
                return 1.0       # "Yes, absolutely" → top score
            elif is_hedged:
                return 0.65      # "Yes, somewhat" → weak positive
            else:
                return 0.8       # Plain "Yes" → solid positive
        elif no_match and (not yes_match or no_match.start() < yes_match.start()):
            # Negative response
            if is_strong:
                return 0.0       # "No, definitely not" → bottom score
            elif is_hedged:
                return 0.35      # "No, not really" → weak negative
            else:
                return 0.2       # Plain "No" → solid negative
        else:
            # No clear yes/no — treat as ambiguous
            return 0.5

    @staticmethod
    def _sigmoid(x: float) -> float:
        """
        Apply sigmoid function to normalize score difference.

        VQAScore per-image averages now span the full [0.0, 1.0] range
        (versus the old capped [0.2, 0.8]).  Diffs between two images
        therefore lie in [-1.0, 1.0], and ×5 amplification gives:

            diff=0.10 → 0.62   (weak preference)
            diff=0.20 → 0.73   (moderate preference)
            diff=0.40 → 0.88   (strong preference)
            diff=0.60 → 0.95   (decisive)

        This matches the old amplification factor but now operates over
        the wider natural range.

        Args:
            x: Score difference in roughly [-1.0, 1.0]

        Returns:
            Normalized value in [0.0, 1.0]
        """
        scaled_x = x * 5.0
        return 1.0 / (1.0 + math.exp(-scaled_x))
