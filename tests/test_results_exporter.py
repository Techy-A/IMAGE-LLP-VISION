"""
Unit tests for ResultsExporter.

Tests validate CSV structure, column presence, value correctness,
and edge cases like images with no matches.
"""

import csv
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.data_models import ImageMetadata, MatchResult, TournamentMetadata, TournamentResult
from src.results_exporter import CSV_COLUMNS, ResultsExporter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_image_files(tmp_path):
    """Create two real tiny PNG files so ImageMetadata validation passes."""
    paths = []
    for i in range(3):
        p = tmp_path / f"img_{i}.png"
        # Write minimal 1x1 PNG bytes
        p.write_bytes(
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00'
            b'\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18'
            b'\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        )
        paths.append(str(p))
    return paths


@pytest.fixture
def sample_images(tmp_image_files):
    """Three ImageMetadata objects."""
    return [
        ImageMetadata(id="img_a", path=tmp_image_files[0], prompt="a cat", model="dalle"),
        ImageMetadata(id="img_b", path=tmp_image_files[1], prompt="a dog", model="sd"),
        ImageMetadata(id="img_c", path=tmp_image_files[2], prompt=None, model=None),
    ]


@pytest.fixture
def sample_matches():
    """Three matches covering all pairs."""
    return [
        MatchResult(
            image_a_id="img_a",
            image_b_id="img_b",
            scores_by_scorer={
                "pickscore": 0.7,
                "hpsv2": 0.65,
                "image_reward": 0.6,
                "vqa": 0.8,
                "vlm_judge": 0.75,
            },
            ensemble_score=0.7,
            winner_id="img_a",
            timestamp=datetime.now(),
        ),
        MatchResult(
            image_a_id="img_a",
            image_b_id="img_c",
            scores_by_scorer={
                "pickscore": 0.55,
                "hpsv2": 0.50,
                "image_reward": 0.48,
                "vqa": 0.60,
                "vlm_judge": 0.52,
            },
            ensemble_score=0.55,
            winner_id="img_a",
            timestamp=datetime.now(),
        ),
        MatchResult(
            image_a_id="img_b",
            image_b_id="img_c",
            scores_by_scorer={
                "pickscore": 0.4,
                "hpsv2": 0.45,
                "image_reward": 0.35,
                "vqa": 0.42,
                "vlm_judge": 0.38,
            },
            ensemble_score=0.4,
            winner_id="img_c",
            timestamp=datetime.now(),
        ),
    ]


@pytest.fixture
def sample_result(sample_matches):
    """A TournamentResult built from the sample matches."""
    elo_ratings = {"img_a": 1520.0, "img_b": 1490.0, "img_c": 1510.0}
    bt_ratings  = {"img_a": 0.45,   "img_b": 0.28,   "img_c": 0.27}
    final_rankings = [("img_a", 0.45), ("img_b", 0.28), ("img_c", 0.27)]

    return TournamentResult(
        matches=sample_matches,
        elo_ratings=elo_ratings,
        bt_ratings=bt_ratings,
        final_rankings=final_rankings,
        metadata=TournamentMetadata(
            start_time=datetime.now(),
            total_images=3,
            total_matches=3,
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestResultsExporterColumns:
    """Validate CSV columns are correct and complete."""

    def test_csv_has_all_expected_columns(self, sample_result, sample_images, tmp_path):
        exporter = ResultsExporter()
        out = str(tmp_path / "out.csv")
        exporter.export_to_csv(sample_result, sample_images, out)

        with open(out, newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames

        assert header == CSV_COLUMNS

    def test_csv_has_correct_row_count(self, sample_result, sample_images, tmp_path):
        exporter = ResultsExporter()
        out = str(tmp_path / "out.csv")
        exporter.export_to_csv(sample_result, sample_images, out)

        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 3  # one per image


class TestResultsExporterValues:
    """Validate computed metric values are correct."""

    def test_winner_has_most_wins(self, sample_result, sample_images, tmp_path):
        exporter = ResultsExporter()
        out = str(tmp_path / "out.csv")
        exporter.export_to_csv(sample_result, sample_images, out)

        with open(out, newline="") as f:
            rows = {r["image_id"]: r for r in csv.DictReader(f)}

        # img_a won 2/2 matches
        assert int(rows["img_a"]["wins"]) == 2
        assert int(rows["img_a"]["losses"]) == 0
        assert float(rows["img_a"]["win_rate"]) == 1.0

    def test_loser_has_zero_wins(self, sample_result, sample_images, tmp_path):
        exporter = ResultsExporter()
        out = str(tmp_path / "out.csv")
        exporter.export_to_csv(sample_result, sample_images, out)

        with open(out, newline="") as f:
            rows = {r["image_id"]: r for r in csv.DictReader(f)}

        # img_b lost 2/2 matches (lost to img_a, lost to img_c)
        assert int(rows["img_b"]["wins"]) == 0
        assert int(rows["img_b"]["losses"]) == 2
        assert float(rows["img_b"]["win_rate"]) == 0.0

    def test_elo_ratings_written_correctly(self, sample_result, sample_images, tmp_path):
        exporter = ResultsExporter()
        out = str(tmp_path / "out.csv")
        exporter.export_to_csv(sample_result, sample_images, out)

        with open(out, newline="") as f:
            rows = {r["image_id"]: r for r in csv.DictReader(f)}

        assert float(rows["img_a"]["elo_rating"]) == pytest.approx(1520.0, abs=0.01)
        assert float(rows["img_b"]["elo_rating"]) == pytest.approx(1490.0, abs=0.01)

    def test_bt_ratings_written_correctly(self, sample_result, sample_images, tmp_path):
        exporter = ResultsExporter()
        out = str(tmp_path / "out.csv")
        exporter.export_to_csv(sample_result, sample_images, out)

        with open(out, newline="") as f:
            rows = {r["image_id"]: r for r in csv.DictReader(f)}

        assert float(rows["img_a"]["bt_rating"]) == pytest.approx(0.45, abs=0.001)

    def test_rank_1_is_highest_rated(self, sample_result, sample_images, tmp_path):
        exporter = ResultsExporter()
        out = str(tmp_path / "out.csv")
        exporter.export_to_csv(sample_result, sample_images, out)

        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))

        # First row should be rank 1 (img_a)
        assert rows[0]["rank"] == "1"
        assert rows[0]["image_id"] == "img_a"

    def test_image_metadata_written(self, sample_result, sample_images, tmp_path):
        exporter = ResultsExporter()
        out = str(tmp_path / "out.csv")
        exporter.export_to_csv(sample_result, sample_images, out)

        with open(out, newline="") as f:
            rows = {r["image_id"]: r for r in csv.DictReader(f)}

        assert rows["img_a"]["prompt"] == "a cat"
        assert rows["img_a"]["model"] == "dalle"
        # img_c has no prompt/model
        assert rows["img_c"]["prompt"] == ""
        assert rows["img_c"]["model"] == ""

    def test_avg_pickscore_computed(self, sample_result, sample_images, tmp_path):
        exporter = ResultsExporter()
        out = str(tmp_path / "out.csv")
        exporter.export_to_csv(sample_result, sample_images, out)

        with open(out, newline="") as f:
            rows = {r["image_id"]: r for r in csv.DictReader(f)}

        # img_a was image_a in match 0 (0.7) and match 1 (0.55) → mean = 0.625
        assert float(rows["img_a"]["avg_pickscore"]) == pytest.approx(0.625, abs=0.001)

    def test_score_inverted_for_image_b(self, sample_result, sample_images, tmp_path):
        """Scores are inverted when image appears as image_b so values reflect its performance."""
        exporter = ResultsExporter()
        out = str(tmp_path / "out.csv")
        exporter.export_to_csv(sample_result, sample_images, out)

        with open(out, newline="") as f:
            rows = {r["image_id"]: r for r in csv.DictReader(f)}

        # img_b was image_b in match 0 (score 0.7 → inverted 0.3)
        # img_b was image_a in match 2 (score 0.4)
        # mean pickscore for img_b = (1-0.7 + 0.4) / 2 = (0.3 + 0.4)/2 = 0.35
        assert float(rows["img_b"]["avg_pickscore"]) == pytest.approx(0.35, abs=0.001)


class TestResultsExporterFileOutput:
    """Validate file creation and path handling."""

    def test_creates_output_directory(self, sample_result, sample_images, tmp_path):
        exporter = ResultsExporter()
        nested = str(tmp_path / "deep" / "nested" / "out.csv")
        exporter.export_to_csv(sample_result, sample_images, nested)
        assert Path(nested).exists()

    def test_returns_absolute_path(self, sample_result, sample_images, tmp_path):
        exporter = ResultsExporter()
        out = str(tmp_path / "out.csv")
        returned = exporter.export_to_csv(sample_result, sample_images, out)
        assert os.path.isabs(returned)

    def test_overwrites_existing_file(self, sample_result, sample_images, tmp_path):
        exporter = ResultsExporter()
        out = str(tmp_path / "out.csv")
        # Write twice — second call should not fail
        exporter.export_to_csv(sample_result, sample_images, out)
        exporter.export_to_csv(sample_result, sample_images, out)
        assert Path(out).exists()


class TestResultsExporterEdgeCases:
    """Edge cases: empty matches, missing scorer fields."""

    def test_raises_on_empty_images(self, sample_result):
        exporter = ResultsExporter()
        with pytest.raises(ValueError, match="images list cannot be empty"):
            exporter.export_to_csv(sample_result, [], "/tmp/out.csv")

    def test_image_with_no_matches_has_empty_scorer_cells(
        self, tmp_path, tmp_image_files
    ):
        """An image that never participated in any match should have empty scorer columns."""
        images = [
            ImageMetadata(id="solo", path=tmp_image_files[0]),
        ]
        # Build minimal result with 'solo' having no matches at all
        result = TournamentResult(
            matches=[],
            elo_ratings={"solo": 1500.0},
            bt_ratings={"solo": 1.0},
            final_rankings=[("solo", 1.0)],
            metadata=TournamentMetadata(
                start_time=datetime.now(),
                total_images=1,
                total_matches=0,
            ),
        )

        exporter = ResultsExporter()
        out = str(tmp_path / "out.csv")
        exporter.export_to_csv(result, images, out)

        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        assert rows[0]["image_id"] == "solo"
        assert rows[0]["avg_pickscore"] == ""
        assert rows[0]["wins"] == "0"
        assert rows[0]["matches_played"] == "0"
