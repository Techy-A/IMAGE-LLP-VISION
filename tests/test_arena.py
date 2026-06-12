"""
Unit tests for Arena tournament orchestration.

Focus: Elo rating updates must reward the actual winner, including the case
where image_b wins (ensemble_score < 0.5). This path had no coverage and
previously rewarded the loser because the raw image_a-perspective score was
passed straight to EloSystem (which interprets `score` as the winner's result).
"""

import os
import tempfile
from typing import List, Tuple

import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from src.arena import Arena
from src.elo_system import EloSystem
from src.bradley_terry_mle import BradleyTerryMLE
from src.ensemble import Ensemble
from src.checkpoint_manager import CheckpointManager
from src.data_models import ImageMetadata
from src.scoring.base_scorer import BaseScorer


class ConstantScorer(BaseScorer):
    """Returns a fixed score (from image_a's perspective) for every pair."""

    def __init__(self, value: float):
        self.value = value

    def compare(self, image_a: PILImage, image_b: PILImage) -> float:
        return self.value

    def batch_compare(self, pairs: List[Tuple[PILImage, PILImage]]) -> List[float]:
        return [self.value for _ in pairs]


class RaisingScorer(BaseScorer):
    """Always raises in compare() to exercise the scorer-failure path."""

    def compare(self, image_a: PILImage, image_b: PILImage) -> float:
        raise RuntimeError("scorer boom")

    def batch_compare(self, pairs: List[Tuple[PILImage, PILImage]]) -> List[float]:
        raise RuntimeError("scorer boom")


# Canonical ensemble weights (must sum to 1.0); keys match the metric scorers.
WEIGHTS = {"vqa": 0.25, "pickscore": 0.15, "hpsv2": 0.20,
           "image_reward": 0.15, "vlm_judge": 0.25}


@pytest.fixture
def two_images():
    """Two real temp PNG files wrapped in ImageMetadata."""
    paths = []
    for color in ("red", "blue"):
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.new("RGB", (10, 10), color=color).save(f.name)
        f.close()
        paths.append(f.name)
    metas = [ImageMetadata(id=f"img_{i}", path=p) for i, p in enumerate(paths)]
    yield metas
    for p in paths:
        os.unlink(p)


def _make_arena_with(images, scorers):
    """Build an Arena over `images` with the given scorer dict and WEIGHTS."""
    ckpt_dir = tempfile.mkdtemp()
    return Arena(
        images=images,
        scorers=scorers,
        elo_system=EloSystem(k_factor=32, initial_rating=1500.0),
        bt_mle=BradleyTerryMLE(beta=1e-10),
        ensemble=Ensemble(weights=WEIGHTS),
        checkpoint_manager=CheckpointManager(checkpoint_dir=ckpt_dir),
        model_rotator=None,
        config={"checkpoint_every": 100},
    )


def _make_arena(images, scorer_value):
    scorers = {name: ConstantScorer(scorer_value) for name in WEIGHTS}
    return _make_arena_with(images, scorers)


def test_image_b_win_rewards_image_b(two_images):
    """When ensemble < 0.5, image_b wins and must GAIN Elo (regression test)."""
    a_id, b_id = two_images[0].id, two_images[1].id
    arena = _make_arena(two_images, scorer_value=0.2)  # image_b decisively wins

    result = arena.run_tournament()

    assert result.elo_ratings[b_id] > 1500.0, "winner image_b should gain rating"
    assert result.elo_ratings[a_id] < 1500.0, "loser image_a should lose rating"
    assert result.elo_ratings[b_id] > result.elo_ratings[a_id]
    # Bradley-Terry should agree on the ordering
    assert result.bt_ratings[b_id] > result.bt_ratings[a_id]
    # The single match's recorded winner is image_b
    assert result.matches[0].winner_id == b_id


def test_image_a_win_rewards_image_a(two_images):
    """When ensemble > 0.5, image_a wins and must GAIN Elo."""
    a_id, b_id = two_images[0].id, two_images[1].id
    arena = _make_arena(two_images, scorer_value=0.8)  # image_a decisively wins

    result = arena.run_tournament()

    assert result.elo_ratings[a_id] > 1500.0 > result.elo_ratings[b_id]
    assert result.matches[0].winner_id == a_id


def test_symmetric_elo_zero_sum(two_images):
    """Elo is zero-sum: winner gain equals loser loss."""
    a_id, b_id = two_images[0].id, two_images[1].id
    arena = _make_arena(two_images, scorer_value=0.3)

    result = arena.run_tournament()
    gain = result.elo_ratings[b_id] - 1500.0
    loss = 1500.0 - result.elo_ratings[a_id]
    assert gain == pytest.approx(loss, abs=1e-9)


def test_failed_scorer_is_recorded_and_excluded(two_images):
    """A scorer that raises in compare() is counted in failed_scorers and dropped
    from the match; the ensemble renormalises over the survivors instead of
    injecting a neutral score (regression test for the dead fallback chain)."""
    a_id, b_id = two_images[0].id, two_images[1].id
    scorers = {name: ConstantScorer(0.8) for name in WEIGHTS}
    scorers["vqa"] = RaisingScorer()  # one scorer always fails
    arena = _make_arena_with(two_images, scorers)

    result = arena.run_tournament()

    # Tournament completed despite the failing scorer.
    assert len(result.matches) == 1
    match = result.matches[0]

    # The failure is recorded once (one match) and the scorer is omitted.
    assert result.metadata.failed_scorers == {"vqa": 1}
    assert "vqa" not in match.scores_by_scorer
    assert set(match.scores_by_scorer) == set(WEIGHTS) - {"vqa"}

    # Survivors all returned 0.8, so the renormalised ensemble is exactly 0.8 —
    # NOT diluted toward 0.5 by injecting a neutral score for the failure.
    assert match.ensemble_score == pytest.approx(0.8)
    assert match.winner_id == a_id
    assert result.elo_ratings[a_id] > 1500.0 > result.elo_ratings[b_id]


def test_all_scorers_fail_falls_back_to_neutral(two_images):
    """If every scorer fails, the ensemble returns a neutral 0.5, the match
    records no scorer scores, and every failure is counted — the tournament
    still completes rather than crashing."""
    a_id, b_id = two_images[0].id, two_images[1].id
    scorers = {name: RaisingScorer() for name in WEIGHTS}
    arena = _make_arena_with(two_images, scorers)

    result = arena.run_tournament()

    match = result.matches[0]
    assert match.scores_by_scorer == {}
    assert match.ensemble_score == pytest.approx(0.5)
    assert result.metadata.failed_scorers == {name: 1 for name in WEIGHTS}
    # 0.5 is not > 0.5, so image_b is recorded as the nominal winner.
    assert match.winner_id == b_id
