from __future__ import annotations

from server.dataset_generator import generate_dataset
from server.graders import grade_easy, grade_hard, grade_medium


def test_easy_grader_perfect_score():
    score = grade_easy(
        dataset=[],
        ground_truth=[],
        issues_fixed={"fixed_missing_values": 4, "original_missing_values": 4, "fixed_duplicates": 2, "original_duplicates": 2},
        steps_taken=0,
        max_steps=30,
    )
    assert score == 1.0


def test_easy_grader_zero_score():
    score = grade_easy(
        dataset=[],
        ground_truth=[],
        issues_fixed={"fixed_missing_values": 0, "original_missing_values": 4, "fixed_duplicates": 0, "original_duplicates": 2},
        steps_taken=30,
        max_steps=30,
    )
    assert score == 0.0


def test_easy_grader_partial_score():
    score = grade_easy(
        dataset=[],
        ground_truth=[],
        issues_fixed={"fixed_missing_values": 2, "original_missing_values": 4, "fixed_duplicates": 1, "original_duplicates": 2},
        steps_taken=15,
        max_steps=30,
    )
    assert 0.3 < score < 0.7


def test_medium_grader_varies_by_quality():
    low_score = grade_medium(
        dataset=[],
        ground_truth=[],
        issues_fixed={"fixed_missing_values": 0, "original_missing_values": 4, "fixed_duplicates": 0, "original_duplicates": 2, "fixed_wrong_format": 0, "original_wrong_format": 4, "fixed_wrong_type": 0, "original_wrong_type": 2},
        steps_taken=50,
        max_steps=50,
    )
    high_score = grade_medium(
        dataset=[],
        ground_truth=[],
        issues_fixed={"fixed_missing_values": 4, "original_missing_values": 4, "fixed_duplicates": 2, "original_duplicates": 2, "fixed_wrong_format": 4, "original_wrong_format": 4, "fixed_wrong_type": 2, "original_wrong_type": 2},
        steps_taken=5,
        max_steps=50,
    )
    assert high_score > low_score


def test_hard_grader_naive_agent_below_half():
    bundle = generate_dataset("hard")
    score = grade_hard(
        dataset=bundle["messy"],
        ground_truth=bundle["ground_truth"],
        issues_fixed={
            "fixed_missing_values": 0,
            "original_missing_values": bundle["issues"].get("missing_values", 0),
            "fixed_duplicates": 0,
            "original_duplicates": bundle["issues"].get("duplicates", 0),
            "fixed_wrong_format": 0,
            "original_wrong_format": bundle["issues"].get("wrong_format", 0),
            "fixed_wrong_type": 0,
            "original_wrong_type": bundle["issues"].get("wrong_type", 0),
            "fixed_invalid_email": 0,
            "original_invalid_email": bundle["issues"].get("invalid_email", 0),
            "fixed_outliers": 0,
            "original_outliers": bundle["issues"].get("outliers", 0),
            "fixed_near_duplicates": 0,
            "original_near_duplicates": bundle["issues"].get("near_duplicates", 0),
            "fixed_inconsistent_values": 0,
            "original_inconsistent_values": bundle["issues"].get("inconsistent_values", 0),
            "destructive_actions": 0,
        },
        steps_taken=80,
        max_steps=80,
    )
    assert score < 0.5


def test_graders_are_deterministic():
    bundle = generate_dataset("medium")
    payload = {
        "fixed_missing_values": 2,
        "original_missing_values": bundle["issues"].get("missing_values", 0),
        "fixed_duplicates": 1,
        "original_duplicates": bundle["issues"].get("duplicates", 0),
        "fixed_wrong_format": 2,
        "original_wrong_format": bundle["issues"].get("wrong_format", 0),
        "fixed_wrong_type": 1,
        "original_wrong_type": bundle["issues"].get("wrong_type", 0),
    }
    score_one = grade_medium(bundle["messy"], bundle["ground_truth"], payload, steps_taken=10, max_steps=50)
    score_two = grade_medium(bundle["messy"], bundle["ground_truth"], payload, steps_taken=10, max_steps=50)
    assert score_one == score_two
