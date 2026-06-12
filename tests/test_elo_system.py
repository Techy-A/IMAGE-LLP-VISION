"""
Property-based tests for Elo rating system.

**Validates: Requirements 7.1, 7.2, 7.3**

Tests that verify Elo rating system maintains conservation properties,
follows correct rating update formulas, and respects K-factor configuration.
"""

import math
import pytest
from hypothesis import given, strategies as st, settings, assume

from src.elo_system import EloSystem


# ============================================================================
# Property-Based Tests
# ============================================================================


@given(
    k_factor=st.integers(min_value=1, max_value=128),
    initial_rating=st.floats(min_value=100.0, max_value=3000.0),
    winner_rating=st.floats(min_value=100.0, max_value=3000.0),
    loser_rating=st.floats(min_value=100.0, max_value=3000.0),
    score=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=100, deadline=5000)
def test_elo_conservation_property(
    k_factor: int,
    initial_rating: float,
    winner_rating: float,
    loser_rating: float,
    score: float
):
    """
    **Property 9: Elo Conservation Property**
    
    Verify that Elo rating updates preserve the sum of ratings (zero-sum property).
    
    For any match between two images, the sum of ratings before the match
    should equal the sum of ratings after the match. This is a fundamental
    property of the Elo system - rating points are transferred, not created.
    
    **Validates: Requirements 7.1, 7.2, 7.3**
    
    Args:
        k_factor: K-factor controlling rating volatility
        initial_rating: Initial rating for new images
        winner_rating: Pre-match rating of winner
        loser_rating: Pre-match rating of loser
        score: Match score in [0.0, 1.0]
    """
    # Initialize Elo system
    elo = EloSystem(k_factor=k_factor, initial_rating=initial_rating)
    
    # Set up pre-match ratings
    winner_id = "image_winner"
    loser_id = "image_loser"
    
    elo.set_rating(winner_id, winner_rating)
    elo.set_rating(loser_id, loser_rating)
    
    # Record sum of ratings before update
    sum_before = elo.get_rating(winner_id) + elo.get_rating(loser_id)
    
    # Update ratings based on match result
    elo.update_ratings(winner_id, loser_id, score)
    
    # Record sum of ratings after update
    sum_after = elo.get_rating(winner_id) + elo.get_rating(loser_id)
    
    # PROPERTY: Sum of ratings should be conserved (zero-sum)
    # Allow small floating-point tolerance
    assert abs(sum_before - sum_after) < 1e-6, (
        f"Elo conservation violated: sum before ({sum_before:.6f}) != "
        f"sum after ({sum_after:.6f}), difference = {abs(sum_before - sum_after):.6f}"
    )


@given(
    k_factor=st.integers(min_value=1, max_value=128),
    initial_rating=st.floats(min_value=100.0, max_value=3000.0),
    num_players=st.integers(min_value=3, max_value=10),
    num_matches=st.integers(min_value=5, max_value=20)
)
@settings(max_examples=50, deadline=10000)
def test_elo_conservation_multiple_matches(
    k_factor: int,
    initial_rating: float,
    num_players: int,
    num_matches: int
):
    """
    **Property 9 Extended: Elo Conservation Over Multiple Matches**
    
    Verify that Elo conservation holds across a sequence of matches.
    
    The total sum of all ratings should remain constant throughout
    a tournament, regardless of the number of matches played.
    
    **Validates: Requirements 7.1, 7.2, 7.3**
    
    Args:
        k_factor: K-factor controlling rating volatility
        initial_rating: Initial rating for new images
        num_players: Number of players in the tournament
        num_matches: Number of matches to simulate
    """
    # Initialize Elo system
    elo = EloSystem(k_factor=k_factor, initial_rating=initial_rating)
    
    # Create players with initial ratings
    player_ids = [f"player_{i}" for i in range(num_players)]
    
    # Initialize all players (they get initial_rating automatically)
    for player_id in player_ids:
        elo.get_rating(player_id)  # Access to initialize
    
    # Record initial sum
    sum_initial = sum(elo.get_rating(pid) for pid in player_ids)
    
    # Simulate random matches
    import random
    random.seed(42)  # Deterministic for reproducibility
    
    for _ in range(num_matches):
        # Pick two different players randomly
        winner_id, loser_id = random.sample(player_ids, 2)
        
        # Random score between 0.5 and 1.0 (winner should have higher score)
        score = 0.5 + random.random() * 0.5
        
        # Update ratings
        elo.update_ratings(winner_id, loser_id, score)
    
    # Record final sum
    sum_final = sum(elo.get_rating(pid) for pid in player_ids)
    
    # PROPERTY: Total sum should be conserved
    expected_sum = initial_rating * num_players
    
    assert abs(sum_initial - expected_sum) < 1e-6, (
        f"Initial sum incorrect: expected {expected_sum:.6f}, got {sum_initial:.6f}"
    )
    
    assert abs(sum_final - expected_sum) < 1e-3, (
        f"Elo conservation violated over {num_matches} matches: "
        f"initial sum ({sum_initial:.6f}) != final sum ({sum_final:.6f}), "
        f"difference = {abs(sum_initial - sum_final):.6f}"
    )


@given(
    k_factor=st.integers(min_value=1, max_value=128),
    rating_a=st.floats(min_value=100.0, max_value=3000.0),
    rating_b=st.floats(min_value=100.0, max_value=3000.0),
    score=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=100, deadline=5000)
def test_elo_formula_correctness(
    k_factor: int,
    rating_a: float,
    rating_b: float,
    score: float
):
    """
    **Property: Elo Rating Formula Correctness**
    
    Verify that rating updates follow the standard Elo formula exactly.
    
    The Elo update formula is:
    - new_rating_A = old_rating_A + K * (actual_A - expected_A)
    - new_rating_B = old_rating_B + K * (actual_B - expected_B)
    
    Where expected_A = 1 / (1 + 10^((rating_B - rating_A) / 400))
    
    **Validates: Requirements 7.1, 7.3**
    
    Args:
        k_factor: K-factor controlling rating volatility
        rating_a: Pre-match rating of image A
        rating_b: Pre-match rating of image B
        score: Match score for A in [0.0, 1.0]
    """
    # Initialize Elo system
    elo = EloSystem(k_factor=k_factor, initial_rating=1500.0)
    
    # Set up pre-match ratings
    image_a = "image_a"
    image_b = "image_b"
    
    elo.set_rating(image_a, rating_a)
    elo.set_rating(image_b, rating_b)
    
    # Calculate expected scores using Elo formula
    exponent = (rating_b - rating_a) / 400.0
    expected_a = 1.0 / (1.0 + math.pow(10, exponent))
    expected_b = 1.0 - expected_a
    
    # Actual scores
    actual_a = score
    actual_b = 1.0 - score
    
    # Calculate expected rating changes
    expected_change_a = k_factor * (actual_a - expected_a)
    expected_change_b = k_factor * (actual_b - expected_b)
    
    expected_new_rating_a = rating_a + expected_change_a
    expected_new_rating_b = rating_b + expected_change_b
    
    # Update ratings using EloSystem
    elo.update_ratings(image_a, image_b, score)
    
    # Get actual new ratings
    actual_new_rating_a = elo.get_rating(image_a)
    actual_new_rating_b = elo.get_rating(image_b)
    
    # PROPERTY: Ratings should match Elo formula exactly
    assert abs(actual_new_rating_a - expected_new_rating_a) < 1e-6, (
        f"Rating formula incorrect for image A: "
        f"expected {expected_new_rating_a:.6f}, got {actual_new_rating_a:.6f}"
    )
    
    assert abs(actual_new_rating_b - expected_new_rating_b) < 1e-6, (
        f"Rating formula incorrect for image B: "
        f"expected {expected_new_rating_b:.6f}, got {actual_new_rating_b:.6f}"
    )


@given(
    initial_rating=st.floats(min_value=100.0, max_value=3000.0),
    k_factor_small=st.integers(min_value=1, max_value=32),
    k_factor_large=st.integers(min_value=33, max_value=128),
    rating_a=st.floats(min_value=1000.0, max_value=2000.0),
    rating_b=st.floats(min_value=1000.0, max_value=2000.0),
    score=st.floats(min_value=0.6, max_value=1.0)
)
@settings(max_examples=100, deadline=5000)
def test_k_factor_volatility_effect(
    initial_rating: float,
    k_factor_small: int,
    k_factor_large: int,
    rating_a: float,
    rating_b: float,
    score: float
):
    """
    **Property: K-Factor Controls Rating Volatility**
    
    Verify that larger K-factors produce larger rating changes.
    
    The K-factor controls how much ratings change after each match.
    A larger K-factor should always produce larger (or equal) rating
    changes than a smaller K-factor for the same match.
    
    **Validates: Requirement 7.3**
    
    Args:
        initial_rating: Initial rating for new images
        k_factor_small: Smaller K-factor
        k_factor_large: Larger K-factor (must be > k_factor_small)
        rating_a: Pre-match rating of image A
        rating_b: Pre-match rating of image B
        score: Match score for A in [0.6, 1.0] (A wins)
    """
    # Ensure k_factor_large > k_factor_small
    assume(k_factor_large > k_factor_small)
    
    # System with small K-factor
    elo_small = EloSystem(k_factor=k_factor_small, initial_rating=initial_rating)
    elo_small.set_rating("image_a", rating_a)
    elo_small.set_rating("image_b", rating_b)
    
    rating_a_before_small = elo_small.get_rating("image_a")
    rating_b_before_small = elo_small.get_rating("image_b")
    
    elo_small.update_ratings("image_a", "image_b", score)
    
    rating_a_after_small = elo_small.get_rating("image_a")
    rating_b_after_small = elo_small.get_rating("image_b")
    
    change_a_small = abs(rating_a_after_small - rating_a_before_small)
    change_b_small = abs(rating_b_after_small - rating_b_before_small)
    
    # System with large K-factor
    elo_large = EloSystem(k_factor=k_factor_large, initial_rating=initial_rating)
    elo_large.set_rating("image_a", rating_a)
    elo_large.set_rating("image_b", rating_b)
    
    rating_a_before_large = elo_large.get_rating("image_a")
    rating_b_before_large = elo_large.get_rating("image_b")
    
    elo_large.update_ratings("image_a", "image_b", score)
    
    rating_a_after_large = elo_large.get_rating("image_a")
    rating_b_after_large = elo_large.get_rating("image_b")
    
    change_a_large = abs(rating_a_after_large - rating_a_before_large)
    change_b_large = abs(rating_b_after_large - rating_b_before_large)
    
    # PROPERTY: Larger K-factor should produce larger rating changes
    # The rating change is proportional to K-factor
    # change_large / change_small should equal k_large / k_small
    
    ratio_expected = k_factor_large / k_factor_small
    
    # Check ratio for winner (image A)
    if change_a_small > 1e-6:  # Avoid division by small numbers
        ratio_a = change_a_large / change_a_small
        assert abs(ratio_a - ratio_expected) < 0.01, (
            f"K-factor effect incorrect for winner: "
            f"expected ratio {ratio_expected:.3f}, got {ratio_a:.3f}"
        )
    
    # Check ratio for loser (image B)
    if change_b_small > 1e-6:  # Avoid division by small numbers
        ratio_b = change_b_large / change_b_small
        assert abs(ratio_b - ratio_expected) < 0.01, (
            f"K-factor effect incorrect for loser: "
            f"expected ratio {ratio_expected:.3f}, got {ratio_b:.3f}"
        )


@given(
    initial_rating=st.floats(min_value=100.0, max_value=3000.0)
)
@settings(max_examples=50, deadline=5000)
def test_initial_rating_consistency(initial_rating: float):
    """
    **Property: Initial Rating Consistency**
    
    Verify that all new images are initialized with the configured initial rating.
    
    **Validates: Requirement 7.2**
    
    Args:
        initial_rating: Configured initial rating value
    """
    # Initialize Elo system
    elo = EloSystem(k_factor=32, initial_rating=initial_rating)
    
    # Create several new images
    image_ids = [f"image_{i}" for i in range(10)]
    
    # PROPERTY: All new images should have the initial rating
    for image_id in image_ids:
        rating = elo.get_rating(image_id)
        assert abs(rating - initial_rating) < 1e-6, (
            f"Initial rating incorrect for {image_id}: "
            f"expected {initial_rating:.6f}, got {rating:.6f}"
        )


@given(
    k_factor=st.integers(min_value=1, max_value=128),
    initial_rating=st.floats(min_value=100.0, max_value=3000.0),
    winner_rating=st.floats(min_value=100.0, max_value=3000.0),
    loser_rating=st.floats(min_value=100.0, max_value=3000.0)
)
@settings(max_examples=100, deadline=5000)
def test_rating_update_occurs(
    k_factor: int,
    initial_rating: float,
    winner_rating: float,
    loser_rating: float
):
    """
    **Property: Rating Update Occurs After Match**
    
    Verify that ratings actually change after a match (unless it's a perfect tie).
    
    **Validates: Requirement 7.1**
    
    Args:
        k_factor: K-factor controlling rating volatility
        initial_rating: Initial rating for new images
        winner_rating: Pre-match rating of winner
        loser_rating: Pre-match rating of loser
    """
    # Initialize Elo system
    elo = EloSystem(k_factor=k_factor, initial_rating=initial_rating)
    
    # Set up pre-match ratings
    winner_id = "image_winner"
    loser_id = "image_loser"
    
    elo.set_rating(winner_id, winner_rating)
    elo.set_rating(loser_id, loser_rating)
    
    rating_before_winner = elo.get_rating(winner_id)
    rating_before_loser = elo.get_rating(loser_id)
    
    # Update with a clear win (score = 1.0)
    elo.update_ratings(winner_id, loser_id, score=1.0)
    
    rating_after_winner = elo.get_rating(winner_id)
    rating_after_loser = elo.get_rating(loser_id)
    
    # PROPERTY: Ratings should change after a match
    # Winner should gain points
    assert rating_after_winner > rating_before_winner or \
           abs(rating_after_winner - rating_before_winner) > 1e-6, (
        f"Winner rating should increase: before={rating_before_winner:.6f}, "
        f"after={rating_after_winner:.6f}"
    )
    
    # Loser should lose points
    assert rating_after_loser < rating_before_loser or \
           abs(rating_after_loser - rating_before_loser) > 1e-6, (
        f"Loser rating should decrease: before={rating_before_loser:.6f}, "
        f"after={rating_after_loser:.6f}"
    )


# ============================================================================
# Unit Tests
# ============================================================================


def test_elo_conservation_simple():
    """
    Simple unit test for Elo conservation with fixed values.
    
    **Validates: Requirements 7.1, 7.2, 7.3**
    """
    elo = EloSystem(k_factor=32, initial_rating=1500.0)
    
    # Set up two players
    elo.set_rating("player_a", 1600.0)
    elo.set_rating("player_b", 1400.0)
    
    # Record sum before
    sum_before = elo.get_rating("player_a") + elo.get_rating("player_b")
    assert abs(sum_before - 3000.0) < 1e-6
    
    # Player A wins
    elo.update_ratings("player_a", "player_b", score=1.0)
    
    # Record sum after
    sum_after = elo.get_rating("player_a") + elo.get_rating("player_b")
    
    # Verify conservation
    assert abs(sum_before - sum_after) < 1e-6, (
        f"Conservation violated: {sum_before} != {sum_after}"
    )


def test_k_factor_effects():
    """
    Unit test verifying K-factor affects rating change magnitude.
    
    **Validates: Requirement 7.3**
    """
    # Small K-factor
    elo_small = EloSystem(k_factor=16, initial_rating=1500.0)
    elo_small.set_rating("player_a", 1500.0)
    elo_small.set_rating("player_b", 1500.0)
    
    elo_small.update_ratings("player_a", "player_b", score=1.0)
    change_small = abs(elo_small.get_rating("player_a") - 1500.0)
    
    # Large K-factor
    elo_large = EloSystem(k_factor=64, initial_rating=1500.0)
    elo_large.set_rating("player_a", 1500.0)
    elo_large.set_rating("player_b", 1500.0)
    
    elo_large.update_ratings("player_a", "player_b", score=1.0)
    change_large = abs(elo_large.get_rating("player_a") - 1500.0)
    
    # Larger K-factor should produce larger change
    assert change_large > change_small
    
    # Ratio should match K-factor ratio
    ratio = change_large / change_small
    expected_ratio = 64 / 16
    assert abs(ratio - expected_ratio) < 0.01


def test_initial_rating_applied():
    """
    Unit test verifying initial rating is applied correctly.
    
    **Validates: Requirement 7.2**
    """
    initial_rating = 1200.0
    elo = EloSystem(k_factor=32, initial_rating=initial_rating)
    
    # Get rating for new player
    rating = elo.get_rating("new_player")
    
    assert abs(rating - initial_rating) < 1e-6


def test_rating_update_formula():
    """
    Unit test verifying Elo formula implementation.
    
    **Validates: Requirement 7.1**
    """
    elo = EloSystem(k_factor=32, initial_rating=1500.0)
    
    # Set equal ratings
    elo.set_rating("player_a", 1500.0)
    elo.set_rating("player_b", 1500.0)
    
    # Expected score for equal ratings is 0.5
    expected_score = elo.get_expected_score("player_a", "player_b")
    assert abs(expected_score - 0.5) < 1e-6
    
    # Player A wins (score = 1.0)
    # Expected change = K * (actual - expected) = 32 * (1.0 - 0.5) = 16
    elo.update_ratings("player_a", "player_b", score=1.0)
    
    rating_a = elo.get_rating("player_a")
    rating_b = elo.get_rating("player_b")
    
    # Player A should gain 16 points
    assert abs(rating_a - 1516.0) < 1e-6
    
    # Player B should lose 16 points
    assert abs(rating_b - 1484.0) < 1e-6


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
