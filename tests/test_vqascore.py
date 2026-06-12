"""
Unit tests for VQAScore.

Covers all four fixed issues:
1. Prompt-aware fidelity questions
2. Robust _parse_response() (no spurious keyword matches)
3. Full [0.0, 1.0] score range (no artificial cap)
4. Batch caching (same image scored only once per prompt)
"""

import math
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, call, patch

import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from src.scoring.vqascore import VQAScore


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _solid_image(color: str = "red") -> PILImage:
    return Image.new("RGB", (32, 32), color=color)


def _make_scorer(responses: List[str]) -> VQAScore:
    """
    Build a VQAScore whose InstructBLIP model returns *responses* in order
    (cycling if exhausted).
    """
    mock_model = MagicMock()
    # cycle through responses
    response_iter = iter(responses * 100)
    mock_model.query_image.side_effect = lambda **kwargs: next(response_iter)
    return VQAScore(instructblip_model=mock_model)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_init_raises_without_model():
    with pytest.raises(ValueError, match="requires a loaded InstructBLIP"):
        VQAScore(instructblip_model=None)


def test_init_stores_model():
    mock_model = MagicMock()
    scorer = VQAScore(instructblip_model=mock_model)
    assert scorer.model is mock_model


# ---------------------------------------------------------------------------
# Fix 1: _parse_response — robust word-boundary matching
# ---------------------------------------------------------------------------

class TestParseResponse:
    """Tests for the improved _parse_response() static method."""

    def test_plain_yes(self):
        assert VQAScore._parse_response("Yes") == pytest.approx(0.8)

    def test_plain_no(self):
        assert VQAScore._parse_response("No") == pytest.approx(0.2)

    def test_yes_absolutely(self):
        assert VQAScore._parse_response("Yes, absolutely!") == pytest.approx(1.0)

    def test_no_definitely(self):
        assert VQAScore._parse_response("No, definitely not.") == pytest.approx(0.0)

    def test_yes_somewhat(self):
        assert VQAScore._parse_response("Yes, somewhat.") == pytest.approx(0.65)

    def test_no_not_really(self):
        # "not really" is not in the HEDGES set, so this is a plain "No" → 0.2
        assert VQAScore._parse_response("No, not really.") == pytest.approx(0.2)

    def test_no_somewhat(self):
        # "somewhat" IS a hedge word → weak negative
        assert VQAScore._parse_response("No, somewhat.") == pytest.approx(0.35)

    def test_ambiguous_empty(self):
        assert VQAScore._parse_response("") == pytest.approx(0.5)

    def test_ambiguous_no_signal(self):
        assert VQAScore._parse_response("It depends on the lighting.") == pytest.approx(0.5)

    def test_no_in_notable_does_not_match(self):
        """'notable' must NOT trigger a negative match."""
        # "notable" contains "no" but is not a standalone word
        result = VQAScore._parse_response("This image is notable for its quality.")
        assert result == pytest.approx(0.5)

    def test_first_word_wins_no_then_yes(self):
        """'No, but yes it is interesting' — 'no' comes first → negative."""
        result = VQAScore._parse_response("No, but yes it is interesting.")
        assert result == pytest.approx(0.2)

    def test_first_word_wins_yes_then_no(self):
        """'Yes, although no it's not the best' — 'yes' comes first → positive."""
        result = VQAScore._parse_response("Yes, although no it's not the best.")
        assert result == pytest.approx(0.8)

    def test_case_insensitive(self):
        assert VQAScore._parse_response("YES") == pytest.approx(0.8)
        assert VQAScore._parse_response("NO") == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Fix 3: Score range — _parse_response spans full [0.0, 1.0]
# ---------------------------------------------------------------------------

class TestScoreRange:
    def test_max_score_reachable(self):
        """Strong positive → 1.0 (no artificial cap)."""
        assert VQAScore._parse_response("Yes, absolutely!") == pytest.approx(1.0)

    def test_min_score_reachable(self):
        """Strong negative → 0.0 (no artificial 0.2 floor)."""
        assert VQAScore._parse_response("No, definitely not.") == pytest.approx(0.0)

    def test_compare_returns_extreme_when_scores_differ_maximally(self):
        """
        Image A gets all 1.0 responses, Image B gets all 0.0 responses.
        The final compare() score should be very close to 1.0.
        """
        # We need a scorer whose model alternates: first 3 calls for image A
        # return "Yes, absolutely", next 3 for image B return "No, definitely not"
        mock_model = MagicMock()
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 3:
                return "Yes, absolutely"
            return "No, definitely not"

        mock_model.query_image.side_effect = side_effect
        scorer = VQAScore(instructblip_model=mock_model)

        img_a = _solid_image("red")
        img_b = _solid_image("blue")
        score = scorer.compare(img_a, img_b)

        # sigmoid(5 * (1.0 - 0.0)) = sigmoid(5) ≈ 0.993
        assert score > 0.98


# ---------------------------------------------------------------------------
# Fix 1 (continued): Prompt-aware fidelity questions
# ---------------------------------------------------------------------------

class TestPromptAwareness:
    def test_no_prompt_uses_3_questions(self):
        mock_model = MagicMock()
        mock_model.query_image.return_value = "Yes"
        scorer = VQAScore(instructblip_model=mock_model)

        img = _solid_image()
        scorer._score_image(img, prompt=None)

        # 3 generic quality questions only
        assert mock_model.query_image.call_count == 3

    def test_with_prompt_uses_5_questions(self):
        mock_model = MagicMock()
        mock_model.query_image.return_value = "Yes"
        scorer = VQAScore(instructblip_model=mock_model)

        img = _solid_image()
        scorer._score_image(img, prompt="a red sunset")

        # 3 generic + 2 fidelity questions
        assert mock_model.query_image.call_count == 5

    def test_prompt_text_appears_in_fidelity_questions(self):
        mock_model = MagicMock()
        mock_model.query_image.return_value = "Yes"
        scorer = VQAScore(instructblip_model=mock_model)

        prompt_text = "a golden retriever"
        img = _solid_image()
        scorer._score_image(img, prompt=prompt_text)

        all_prompts = [c.kwargs.get("prompt", "") for c in mock_model.query_image.call_args_list]
        # At least one fidelity question should contain the prompt text
        assert any(prompt_text in p for p in all_prompts), (
            f"Expected '{prompt_text}' in at least one question, got: {all_prompts}"
        )

    def test_empty_prompt_treated_as_no_prompt(self):
        mock_model = MagicMock()
        mock_model.query_image.return_value = "Yes"
        scorer = VQAScore(instructblip_model=mock_model)

        img = _solid_image()
        scorer._score_image(img, prompt="   ")  # whitespace-only

        # Only 3 generic questions — whitespace prompt ignored
        assert mock_model.query_image.call_count == 3

    def test_compare_forwards_prompts(self):
        mock_model = MagicMock()
        mock_model.query_image.return_value = "Yes"
        scorer = VQAScore(instructblip_model=mock_model)

        img_a = _solid_image("red")
        img_b = _solid_image("blue")
        scorer.compare(img_a, img_b, prompt_a="a rose", prompt_b="the ocean")

        # 5 calls for img_a + 5 calls for img_b = 10 total
        assert mock_model.query_image.call_count == 10


# ---------------------------------------------------------------------------
# Fix 4: Batch caching
# ---------------------------------------------------------------------------

class TestBatchCompare:
    def test_batch_same_image_scored_once(self):
        """
        When the same PIL object appears in multiple pairs with the same prompt,
        _score_image should only be called once for it (cache hit).
        """
        mock_model = MagicMock()
        mock_model.query_image.return_value = "Yes"
        scorer = VQAScore(instructblip_model=mock_model)

        img_a = _solid_image("red")
        img_b = _solid_image("blue")
        img_c = _solid_image("green")

        # img_a appears in both pairs, but should only be scored once
        pairs = [(img_a, img_b), (img_a, img_c)]
        scorer.batch_compare(pairs)

        # img_a: 3 questions × 1 (cached), img_b: 3, img_c: 3 → total 9
        assert mock_model.query_image.call_count == 9

    def test_batch_returns_correct_length(self):
        mock_model = MagicMock()
        mock_model.query_image.return_value = "Yes"
        scorer = VQAScore(instructblip_model=mock_model)

        pairs = [
            (_solid_image("red"), _solid_image("blue")),
            (_solid_image("green"), _solid_image("yellow")),
            (_solid_image("white"), _solid_image("black")),
        ]
        results = scorer.batch_compare(pairs)
        assert len(results) == 3

    def test_batch_scores_in_valid_range(self):
        mock_model = MagicMock()
        mock_model.query_image.return_value = "Yes"
        scorer = VQAScore(instructblip_model=mock_model)

        pairs = [(_solid_image(), _solid_image()) for _ in range(5)]
        results = scorer.batch_compare(pairs)
        for score in results:
            assert 0.0 <= score <= 1.0

    def test_batch_prompts_length_mismatch_raises(self):
        mock_model = MagicMock()
        mock_model.query_image.return_value = "Yes"
        scorer = VQAScore(instructblip_model=mock_model)

        pairs = [(_solid_image(), _solid_image())]
        with pytest.raises(ValueError, match="prompts length"):
            scorer.batch_compare(pairs, prompts=[("a", "b"), ("c", "d")])

    def test_batch_with_prompts_uses_fidelity_questions(self):
        mock_model = MagicMock()
        mock_model.query_image.return_value = "Yes"
        scorer = VQAScore(instructblip_model=mock_model)

        img_a = _solid_image("red")
        img_b = _solid_image("blue")
        pairs = [(img_a, img_b)]
        scorer.batch_compare(pairs, prompts=[("a sunset", "the moon")])

        # img_a gets 5 questions (3 + 2 fidelity), img_b gets 5 → total 10
        assert mock_model.query_image.call_count == 10

    def test_batch_same_image_different_prompt_scored_separately(self):
        """
        Same PIL object but different prompts → two separate _score_image calls
        (different cache keys).
        """
        mock_model = MagicMock()
        mock_model.query_image.return_value = "Yes"
        scorer = VQAScore(instructblip_model=mock_model)

        img = _solid_image("red")
        img_b = _solid_image("blue")
        img_c = _solid_image("green")

        pairs = [(img, img_b), (img, img_c)]
        # Different prompts for img across two pairs
        scorer.batch_compare(pairs, prompts=[("a rose", "sky"), ("a sunset", "water")])

        # img appears twice with different prompts → scored twice (5 each = 10)
        # img_b: 5, img_c: 5 → total 20
        assert mock_model.query_image.call_count == 20


# ---------------------------------------------------------------------------
# _sigmoid helper
# ---------------------------------------------------------------------------

class TestSigmoid:
    def test_zero_difference_is_neutral(self):
        assert VQAScore._sigmoid(0.0) == pytest.approx(0.5)

    def test_positive_diff_above_half(self):
        assert VQAScore._sigmoid(0.2) > 0.5

    def test_negative_diff_below_half(self):
        assert VQAScore._sigmoid(-0.2) < 0.5

    def test_symmetry(self):
        x = 0.3
        assert VQAScore._sigmoid(x) == pytest.approx(1.0 - VQAScore._sigmoid(-x))


# ---------------------------------------------------------------------------
# compare() error handling
# ---------------------------------------------------------------------------

class TestCompareErrorHandling:
    def test_compare_raises_on_model_error(self):
        mock_model = MagicMock()
        mock_model.query_image.side_effect = RuntimeError("GPU OOM")
        scorer = VQAScore(instructblip_model=mock_model)

        img_a = _solid_image("red")
        img_b = _solid_image("blue")

        # All 3 prompts will fail → neutral 0.5 per image → diff 0 → 0.5
        # (errors are caught inside _score_image and produce 0.5 neutral)
        score = scorer.compare(img_a, img_b)
        assert score == pytest.approx(0.5)
