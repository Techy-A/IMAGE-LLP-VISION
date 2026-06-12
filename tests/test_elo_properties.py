"""
Property-based tests for Elo rating system.

**Validates: Requirements 7.1, 7.2, 7.3**

Tests that the Elo rating system maintains mathematical properties
across all valid match configurations.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import math
from typing import List, Tuple

from src.elo_system import EloSystem


@given(
    k_factor=st.integers(min_value=1, max_value=100),
    initial_rating=st.floats(min_value=100.0, max_value=3000.0),
    num_images=st.integers(min_value=2, max_value=10),
    match_score=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=100, deadline=5000)
def test_elo_conservation_property(
    k_factor: int,
    initial_rating: float,
    num_images: int,
    match_score: float
):
    """
    **Property 9: Elo Conservation Property**
    
    Test that the sum of all rating changes equals zero after any match.
    
    **Validates: Requirements 7.1, 7.2, 7.3**
    
    The Elo rating system is zero-sum: when one player's rating increases,
    another's decreases by the same amount. This ensures rating inflation
    doesn't occur and the total rating points remain constant.
    
    For any match between two images:
    - sum(ratings_before) = sum(ratings_after)
    - winner_change + loser_change = 0
    
    Args:
        k_factor: Rating volatility parameter
        initial_rating: Starting rating for all images
        num_images: Number of images in system
        match_score: Match outcome score
    """
    
    # Initialize Elo system
    elo = EloSystem(k_factor=k_factor, initial_rating=initial_rating)
    
    # Create image IDs
    image_ids = [f"img_{i:03d}" for i in range(num_images)]
    
    # Initialize all images with their starting rating
    for image_id in image_ids:
        elo.get_rating(image_id)  # This initializes the rating
    
    # Record total ratings before match
    ratings_before = sum(elo.get_rating(img_id) for img_id in image_ids)
    
    # Select two different images for the match
    import random
    random.seed(hash((k_factor, initial_rating, num_images, match_score)))
    image_a_id, image_b_id = random.sample(image_ids, 2)
    
    rating_a_before = elo.get_rating(image_a_id)
    rating_b_before = elo.get_rating(image_b_id)
    
    # Determine winner and loser based on score
    # score > 0.5 means image A wins, score < 0.5 means image B wins
    if match_score > 0.5:
        winner_id = image_a_id
        loser_id = image_b_id
    elif match_score < 0.5:
        winner_id = image_b_id
        loser_id = image_a_id
        match_score = 1.0 - match_score  # Flip score for winner
    else:
        # Tie - use image A as "winner" with score 0.5
        winner_id = image_a_id
        loser_id = image_b_id
    
    # Update ratings
    elo.update_ratings(winner_id, loser_id, match_score)
    
    # Record total ratings after match
    ratings_after = sum(elo.get_rating(img_id) for img_id in image_ids)
    
    # Get individual ratings after match
    rating_a_after = elo.get_rating(image_a_id)
    rating_b_after = elo.get_rating(image_b_id)
    
    # Calculate individual rating changes
    change_a = rating_a_after - rating_a_before
    change_b = rating_b_after - rating_b_before
    
    # PROPERTY 9: Conservation - sum of ratings unchanged
    # Allow small floating point error (1e-10)
    assert abs(ratings_before - ratings_after) < 1e-10, \
        f"Total ratings changed: {ratings_before:.10f} -> {ratings_after:.10f} " \
        f"(delta: {abs(ratings_before - ratings_after):.10e})"
    
    # ADDITIONAL CHECK: Rating changes are opposite
    # change_a + change_b should equal 0 (within floating point tolerance)
    assert abs(change_a + change_b) < 1e-10, \
        f"Rating changes don't sum to zero: {change_a:.10f} + {change_b:.10f} = " \
        f"{change_a + change_b:.10e}"


@given(
    initial_rating=st.floats(min_value=100.0, max_value=3000.0),
    num_images=st.integers(min_value=2, max_value=8),
    k_factor_small=st.integers(min_value=1, max_value=20),
    k_factor_large=st.integers(min_value=50, max_value=100),
    match_score=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=50, deadline=5000)
def test_k_factor_effect_on_rating_changes(
    initial_rating: float,
    num_images: int,
    k_factor_small: int,
    k_factor_large: int,
    match_score: float
):
    """
    **Property: K-Factor Effect on Rating Changes**
    
    Test that larger K-factors produce larger rating changes.
    
    **Validates: Requirements 7.1, 7.2, 7.3**
    
    The K-factor controls rating volatility. For the same match outcome:
    - Larger K-factor → larger rating changes
    - Smaller K-factor → smaller rating changes
    
    This property verifies: |change_large_k| > |change_small_k|
    when k_large > k_small for identical match conditions.
    
    Args:
        initial_rating: Starting rating for all images
        num_images: Number of images in system
        k_factor_small: Small K-factor value
        k_factor_large: Large K-factor value
        match_score: Match outcome score
    """
    # Ensure k_large > k_small
    assume(k_factor_large > k_factor_small)
    assume(num_images >= 2)
    
    # Create image IDs
    image_ids = [f"img_{i:03d}" for i in range(num_images)]
    image_a_id = image_ids[0]
    image_b_id = image_ids[1]
    
    # Determine winner/loser
    if match_score > 0.5:
        winner_id = image_a_id
        loser_id = image_b_id
    elif match_score < 0.5:
        winner_id = image_b_id
        loser_id = image_a_id
        match_score = 1.0 - match_score
    else:
        winner_id = image_a_id
        loser_id = image_b_id
    
    # Run match with small K-factor
    elo_small = EloSystem(k_factor=k_factor_small, initial_rating=initial_rating)
    rating_before_small = elo_small.get_rating(winner_id)
    elo_small.update_ratings(winner_id, loser_id, match_score)
    rating_after_small = elo_small.get_rating(winner_id)
    change_small = abs(rating_after_small - rating_before_small)
    
    # Run match with large K-factor
    elo_large = EloSystem(k_factor=k_factor_large, initial_rating=initial_rating)
    rating_before_large = elo_large.get_rating(winner_id)
    elo_large.update_ratings(winner_id, loser_id, match_score)
    rating_after_large = elo_large.get_rating(winner_id)
    change_large = abs(rating_after_large - rating_before_large)
    
    # PROPERTY: Larger K-factor produces larger rating changes
    # The change is proportional to K-factor: change = K * (actual - expected)
    # So change_large / change_small ≈ k_large / k_small
    
    # For non-trivial cases (where actual != expected score)
    if change_small > 0.001:  # Avoid division by zero in near-equal matches
        ratio_changes = change_large / change_small
        ratio_k_factors = k_factor_large / k_factor_small
        
        # The ratios should be approximately equal
        assert abs(ratio_changes - ratio_k_factors) < 0.01, \
            f"K-factor effect mismatch: change ratio {ratio_changes:.4f} != " \
            f"k-factor ratio {ratio_k_factors:.4f}"
    
    # In all cases, larger K should produce larger or equal change
    assert change_large >= change_small - 1e-10, \
        f"Larger K-factor should produce larger change: " \
        f"{change_large:.6f} < {change_small:.6f}"


@given(
    k_factor=st.integers(min_value=1, max_value=100),
    initial_rating=st.floats(min_value=100.0, max_value=3000.0),
    num_matches=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=50, deadline=5000)
def test_elo_conservation_over_multiple_matches(
    k_factor: int,
    initial_rating: float,
    num_matches: int
):
    """
    **Property: Elo Conservation Over Multiple Matches**
    
    Test that rating conservation holds over a sequence of matches.
    
    **Validates: Requirements 7.1, 7.2, 7.3**
    
    After any number of matches, the total sum of all ratings should
    equal the total sum of initial ratings. This ensures the Elo system
    maintains conservation over time, not just for single matches.
    
    Args:
        k_factor: Rating volatility parameter
        initial_rating: Starting rating for all images
        num_matches: Number of matches to simulate
    """
    # Initialize Elo system with 5 images
    num_images = 5
    elo = EloSystem(k_factor=k_factor, initial_rating=initial_rating)
    
    image_ids = [f"img_{i:03d}" for i in range(num_images)]
    
    # Initialize all ratings
    for image_id in image_ids:
        elo.get_rating(image_id)
    
    # Record initial total
    initial_total = sum(elo.get_rating(img_id) for img_id in image_ids)
    
    # Simulate random matches
    import random
    random.seed(42)  # Deterministic for reproducibility
    
    for _ in range(num_matches):
        # Pick two different images
        img_a, img_b = random.sample(image_ids, 2)
        
        # Random match outcome
        score = random.uniform(0.0, 1.0)
        
        if score > 0.5:
            winner, loser = img_a, img_b
        elif score < 0.5:
            winner, loser = img_b, img_a
            score = 1.0 - score
        else:
            winner, loser = img_a, img_b
        
        elo.update_ratings(winner, loser, score)
    
    # Record final total
    final_total = sum(elo.get_rating(img_id) for img_id in image_ids)
    
    # PROPERTY: Total ratings unchanged after multiple matches
    assert abs(initial_total - final_total) < 1e-9, \
        f"Total ratings changed after {num_matches} matches: " \
        f"{initial_total:.10f} -> {final_total:.10f} " \
        f"(delta: {abs(initial_total - final_total):.10e})"


@given(
    k_factor=st.integers(min_value=16, max_value=64),
    initial_rating=st.floats(min_value=1000.0, max_value=2000.0),
    score=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=50, deadline=5000)
def test_k_factor_bounds_rating_change(
    k_factor: int,
    initial_rating: float,
    score: float
):
    """
    **Property: K-Factor Bounds Rating Change**
    
    Test that K-factor provides an upper bound on rating changes.
    
    **Validates: Requirements 7.1, 7.2, 7.3**
    
    The maximum rating change in a single match is bounded by the K-factor.
    For any match: |rating_change| ≤ K
    
    This occurs when the actual outcome is completely unexpected
    (expected score = 0, actual score = 1 or vice versa).
    
    Args:
        k_factor: Rating volatility parameter
        initial_rating: Starting rating
        score: Match outcome score
    """
    elo = EloSystem(k_factor=k_factor, initial_rating=initial_rating)
    
    image_a_id = "img_a"
    image_b_id = "img_b"
    
    # Determine winner/loser
    if score > 0.5:
        winner_id = image_a_id
        loser_id = image_b_id
    elif score < 0.5:
        winner_id = image_b_id
        loser_id = image_a_id
        score = 1.0 - score
    else:
        winner_id = image_a_id
        loser_id = image_b_id
    
    # Record before
    winner_before = elo.get_rating(winner_id)
    loser_before = elo.get_rating(loser_id)
    
    # Update
    elo.update_ratings(winner_id, loser_id, score)
    
    # Record after
    winner_after = elo.get_rating(winner_id)
    loser_after = elo.get_rating(loser_id)
    
    # Calculate changes
    winner_change = abs(winner_after - winner_before)
    loser_change = abs(loser_after - loser_before)
    
    # PROPERTY: Rating changes bounded by K-factor
    assert winner_change <= k_factor + 1e-10, \
        f"Winner rating change {winner_change:.6f} exceeds K-factor {k_factor}"
    
    assert loser_change <= k_factor + 1e-10, \
        f"Loser rating change {loser_change:.6f} exceeds K-factor {k_factor}"


@given(
    k_factor=st.integers(min_value=1, max_value=100),
    initial_rating=st.floats(min_value=100.0, max_value=3000.0),
    rating_diff=st.floats(min_value=-500.0, max_value=500.0)
)
@settings(max_examples=50, deadline=5000)
def test_upset_wins_produce_larger_changes(
    k_factor: int,
    initial_rating: float,
    rating_diff: float
):
    """
    **Property: Upset Wins Produce Larger Rating Changes**
    
    Test that unexpected outcomes (upsets) cause larger rating changes.
    
    **Validates: Requirements 7.1, 7.2, 7.3**
    
    When a lower-rated image beats a higher-rated image, the rating change
    should be larger than when a higher-rated image wins as expected.
    
    Args:
        k_factor: Rating volatility parameter
        initial_rating: Starting rating for reference
        rating_diff: Rating difference between images
    """
    elo = EloSystem(k_factor=k_factor, initial_rating=initial_rating)
    
    # Set up two images with different ratings
    high_rated_id = "img_high"
    low_rated_id = "img_low"
    
    elo.set_rating(high_rated_id, initial_rating + abs(rating_diff))
    elo.set_rating(low_rated_id, initial_rating)
    
    # Case 1: Expected outcome (high rated wins)
    elo_expected = EloSystem(k_factor=k_factor, initial_rating=initial_rating)
    elo_expected.set_rating(high_rated_id, initial_rating + abs(rating_diff))
    elo_expected.set_rating(low_rated_id, initial_rating)
    
    high_before_exp = elo_expected.get_rating(high_rated_id)
    elo_expected.update_ratings(high_rated_id, low_rated_id, 1.0)
    high_after_exp = elo_expected.get_rating(high_rated_id)
    change_expected = abs(high_after_exp - high_before_exp)
    
    # Case 2: Upset outcome (low rated wins)
    elo_upset = EloSystem(k_factor=k_factor, initial_rating=initial_rating)
    elo_upset.set_rating(high_rated_id, initial_rating + abs(rating_diff))
    elo_upset.set_rating(low_rated_id, initial_rating)
    
    low_before_upset = elo_upset.get_rating(low_rated_id)
    elo_upset.update_ratings(low_rated_id, high_rated_id, 1.0)
    low_after_upset = elo_upset.get_rating(low_rated_id)
    change_upset = abs(low_after_upset - low_before_upset)
    
    # PROPERTY: Upset produces larger change than expected outcome
    # Only check if there's a meaningful rating difference
    if abs(rating_diff) > 10.0:
        assert change_upset > change_expected - 1e-10, \
            f"Upset change {change_upset:.6f} should be larger than " \
            f"expected change {change_expected:.6f} with rating_diff={rating_diff:.2f}"


def test_elo_conservation_simple_case():
    """
    Simple unit test for Elo conservation with fixed values.
    
    **Validates: Requirements 7.1, 7.2, 7.3**
    
    This test provides a concrete example of conservation property.
    """
    elo = EloSystem(k_factor=32, initial_rating=1500.0)
    
    # Create 3 images
    images = ["img_a", "img_b", "img_c"]
    
    # Initialize ratings
    for img_id in images:
        elo.get_rating(img_id)
    
    # Total before: 3 * 1500 = 4500
    total_before = sum(elo.get_rating(img_id) for img_id in images)
    assert abs(total_before - 4500.0) < 1e-10
    
    # Run a match: img_a wins against img_b
    elo.update_ratings("img_a", "img_b", 1.0)
    
    # Total after should still be 4500
    total_after = sum(elo.get_rating(img_id) for img_id in images)
    assert abs(total_before - total_after) < 1e-10, \
        f"Conservation violated: {total_before} != {total_after}"
    
    # Verify individual changes sum to zero
    rating_a = elo.get_rating("img_a")
    rating_b = elo.get_rating("img_b")
    rating_c = elo.get_rating("img_c")
    
    change_a = rating_a - 1500.0
    change_b = rating_b - 1500.0
    change_c = rating_c - 1500.0
    
    assert abs(change_c) < 1e-10, "Uninvolved image rating should not change"
    assert abs(change_a + change_b) < 1e-10, \
        f"Rating changes should sum to zero: {change_a} + {change_b} = {change_a + change_b}"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
