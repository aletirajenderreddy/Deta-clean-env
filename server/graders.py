from __future__ import annotations

import copy
from typing import Any, Dict, List


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _sorted_rows(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted([copy.deepcopy(row) for row in dataset], key=lambda row: (str(row.get("id")), str(row.get("name")), str(row.get("email"))))


def _cell_match_ratio(dataset: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> float:
    current_rows = _sorted_rows(dataset)
    truth_rows = _sorted_rows(ground_truth)
    all_columns = list(ground_truth[0].keys()) if ground_truth else []
    total_rows = max(len(current_rows), len(truth_rows))
    total_cells = max(total_rows * max(len(all_columns), 1), 1)

    matches = 0
    for index in range(total_rows):
        current_row = current_rows[index] if index < len(current_rows) else {}
        truth_row = truth_rows[index] if index < len(truth_rows) else {}
        for column in all_columns:
            if current_row.get(column) == truth_row.get(column):
                matches += 1
    return matches / total_cells


def grade_easy(dataset, ground_truth, issues_fixed, steps_taken, max_steps) -> float:
    missing_fixed = _safe_divide(issues_fixed.get("fixed_missing_values", 0), issues_fixed.get("original_missing_values", 0)) * 0.5
    duplicates_removed = _safe_divide(issues_fixed.get("fixed_duplicates", 0), issues_fixed.get("original_duplicates", 0)) * 0.5
    step_bonus = max(0.0, 0.1 * (1 - steps_taken / max_steps))
    return min(1.0, missing_fixed + duplicates_removed + step_bonus)


def grade_medium(dataset, ground_truth, issues_fixed, steps_taken, max_steps) -> float:
    missing_score = _safe_divide(issues_fixed.get("fixed_missing_values", 0), issues_fixed.get("original_missing_values", 0)) * 0.25
    duplicate_score = _safe_divide(issues_fixed.get("fixed_duplicates", 0), issues_fixed.get("original_duplicates", 0)) * 0.20
    format_score = _safe_divide(issues_fixed.get("fixed_wrong_format", 0), issues_fixed.get("original_wrong_format", 0)) * 0.30
    type_score = _safe_divide(issues_fixed.get("fixed_wrong_type", 0), issues_fixed.get("original_wrong_type", 0)) * 0.25
    step_penalty = max(0.0, -0.05 * (steps_taken / max_steps))
    return max(0.0, min(1.0, missing_score + duplicate_score + format_score + type_score + step_penalty))


def grade_hard(dataset, ground_truth, issues_fixed, steps_taken, max_steps) -> float:
    total_score = _cell_match_ratio(dataset, ground_truth) * 0.7
    original_total = sum(
        issues_fixed.get(key, 0)
        for key in [
            "original_missing_values",
            "original_duplicates",
            "original_wrong_format",
            "original_wrong_type",
            "original_invalid_email",
            "original_outliers",
            "original_near_duplicates",
            "original_inconsistent_values",
        ]
    )
    fixed_total = sum(
        issues_fixed.get(key, 0)
        for key in [
            "fixed_missing_values",
            "fixed_duplicates",
            "fixed_wrong_format",
            "fixed_wrong_type",
            "fixed_invalid_email",
            "fixed_outliers",
            "fixed_near_duplicates",
            "fixed_inconsistent_values",
        ]
    )
    issue_score = _safe_divide(fixed_total, original_total) * 0.2
    efficiency = max(0.0, 0.1 * (1 - steps_taken / max_steps))
    data_quality_penalty = -0.05 * issues_fixed.get("destructive_actions", 0)
    return max(0.0, min(1.0, total_score + issue_score + efficiency + data_quality_penalty))
