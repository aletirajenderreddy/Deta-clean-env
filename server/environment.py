from __future__ import annotations

import copy
import os
import re
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

# Ensure project root is on path
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from models import CleaningAction, DataCleanState, DatasetObservation
from server.dataset_generator import generate_dataset
from server.graders import grade_easy, grade_hard, grade_medium
from server.reward import calculate_reward

try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:
        pass


COLUMN_SCHEMA = {
    "id": "int",
    "name": "str",
    "age": "int",
    "email": "email",
    "phone": "phone",
    "salary": "float",
    "hire_date": "date",
    "department": "str",
    "is_active": "bool",
}

ALLOWED_ACTIONS = {"fix_missing", "fix_format", "remove_duplicate", "fix_type", "fix_outlier", "no_op"}
MAX_STEPS = {"easy": 30, "medium": 50, "hard": 80}
CANONICAL_DEPARTMENTS = {"engineering": "Engineering", "eng": "Engineering", "finance": "Finance", "sales": "Sales", "hr": "HR", "operations": "Operations", "ops": "Operations", "marketing": "Marketing"}


class DataCleanEnvironment(Environment):
    def __init__(self):
        self._state = DataCleanState()
        self._dataset: List[Dict[str, Any]] = []
        self._ground_truth: List[Dict[str, Any]] = []
        self._original_issues: Dict[str, int] = {}
        self._issues_remaining: Dict[str, int] = {}
        self._issues_fixed: Dict[str, int] = {}
        self._task_level = "easy"
        self._max_steps = 30
        self._action_history: List[Dict[str, Any]] = []
        self._destructive_actions = 0
        self._last_before_dataset: List[Dict[str, Any]] = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        *,
        task_level: str = "easy",
        **kwargs: Any,
    ) -> DatasetObservation:
        bundle = generate_dataset(task_level)
        self._dataset = copy.deepcopy(bundle["messy"])
        self._ground_truth = copy.deepcopy(bundle["ground_truth"])
        self._task_level = task_level
        self._max_steps = MAX_STEPS[task_level]
        self._state = DataCleanState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_level=task_level,
            total_issues_original=0,
            issues_fixed_count=0,
            score=0.0,
        )
        self._original_issues = self._detect_issues(self._dataset)
        self._issues_remaining = copy.deepcopy(self._original_issues)
        self._issues_fixed = {key: 0 for key in self._original_issues}
        self._action_history = []
        self._destructive_actions = 0
        self._state.total_issues_original = self._total_issues(self._original_issues)
        self._state.score = self._grade_current_dataset()
        return self._build_observation(reward=None, done=False, message="Fresh messy dataset generated.")

    def step(self, action: CleaningAction, timeout_s: Optional[float] = None, **kwargs: Any) -> DatasetObservation:  # noqa: ARG002
        if self._state.episode_id is None:
            self.reset(task_level=self._task_level)

        self._last_before_dataset = copy.deepcopy(self._dataset)
        issues_before_map = self._detect_issues(self._dataset)
        issues_before = self._total_issues(issues_before_map)

        was_valid = self._apply_fix(action)
        issues_after_map = self._detect_issues(self._dataset)
        issues_after = self._total_issues(issues_after_map)
        self._update_fixed_counts(issues_before_map, issues_after_map)

        was_correct_fix = self._is_correct_fix(action)
        if was_valid and not was_correct_fix and self._dataset_match_score(self._dataset) < self._dataset_match_score(self._last_before_dataset):
            self._destructive_actions += 1

        self._issues_remaining = issues_after_map
        self._state.step_count += 1
        self._state.issues_fixed_count = sum(self._issues_fixed.values())
        self._state.score = self._grade_current_dataset()

        done = issues_after == 0 or self._state.step_count >= self._max_steps
        reward = calculate_reward(
            action=action,
            was_valid=was_valid,
            was_correct_fix=was_correct_fix,
            issues_before=issues_before,
            issues_after=issues_after,
            step=self._state.step_count,
            max_steps=self._max_steps,
            task_level=self._task_level,
        )
        if done and issues_after == 0:
            reward = round(min(0.8, reward + 0.30), 4)
            self._state.score = self._grade_current_dataset()

        self._action_history.append(action.model_dump())
        message = self._build_message(action, was_valid, was_correct_fix, done)
        return self._build_observation(reward=reward, done=done, message=message)

    @property
    def state(self) -> DataCleanState:
        return self._state

    def _build_observation(self, reward: float | None, done: bool, message: str) -> DatasetObservation:
        return DatasetObservation(
            done=done,
            reward=reward,
            dataset=copy.deepcopy(self._dataset),
            issues_remaining=copy.deepcopy(self._issues_remaining),
            issues_fixed=copy.deepcopy(self._issues_fixed),
            total_issues_original=self._state.total_issues_original,
            current_step=self._state.step_count,
            max_steps=self._max_steps,
            message=message,
            task_level=self._task_level,
            column_schema=copy.deepcopy(COLUMN_SCHEMA),
        )

    def _build_message(self, action: CleaningAction, was_valid: bool, was_correct_fix: bool, done: bool) -> str:
        if done and self._total_issues(self._issues_remaining) == 0:
            return "All detectable issues fixed. Episode complete."
        if done:
            return "Maximum step limit reached."
        if not was_valid:
            return "Invalid or unhelpful action."
        if was_correct_fix:
            return f"Action '{action.action_type}' improved dataset quality."
        return f"Action '{action.action_type}' was applied but did not improve quality enough."

    def _total_issues(self, issues: Dict[str, int]) -> int:
        return sum(max(value, 0) for value in issues.values())

    def _update_fixed_counts(self, before: Dict[str, int], after: Dict[str, int]) -> None:
        for key in self._issues_fixed:
            delta = before.get(key, 0) - after.get(key, 0)
            if delta > 0:
                self._issues_fixed[key] += delta

    def _apply_fix(self, action: CleaningAction) -> bool:
        if action.action_type not in ALLOWED_ACTIONS:
            return False
        if action.action_type == "no_op":
            return True
        if action.action_type == "remove_duplicate":
            if not 0 <= action.row_index < len(self._dataset):
                return False
            if self._is_duplicate_row(action.row_index) or self._is_near_duplicate_row(action.row_index):
                del self._dataset[action.row_index]
                return True
            return False
        if action.column_name not in COLUMN_SCHEMA:
            return False
        if action.row_index != -1 and not 0 <= action.row_index < len(self._dataset):
            return False

        target_indexes = list(range(len(self._dataset))) if action.row_index == -1 else [action.row_index]
        changed = False
        for row_index in target_indexes:
            row = self._dataset[row_index]
            current_value = row.get(action.column_name)
            updated_value = current_value
            if action.action_type == "fix_missing":
                if not self._is_missing(current_value):
                    continue
                updated_value = self._cast_to_schema(action.column_name, action.new_value)
            elif action.action_type == "fix_format":
                updated_value = self._format_value(action.column_name, action.new_value if action.new_value is not None else current_value)
            elif action.action_type == "fix_type":
                source_value = action.new_value if action.new_value is not None else current_value
                updated_value = self._cast_to_schema(action.column_name, source_value)
            elif action.action_type == "fix_outlier":
                if not self._is_outlier_value(action.column_name, current_value):
                    continue
                updated_value = self._cast_to_schema(action.column_name, action.new_value)

            if updated_value != current_value:
                row[action.column_name] = updated_value
                changed = True
        return changed

    def _is_correct_fix(self, action: CleaningAction) -> bool:
        issues_before = self._detect_issues(self._last_before_dataset)
        issues_after = self._detect_issues(self._dataset)
        if action.action_type == "no_op":
            return self._total_issues(issues_after) == 0
        if action.action_type == "remove_duplicate":
            return issues_after.get("duplicates", 0) < issues_before.get("duplicates", 0) or issues_after.get("near_duplicates", 0) < issues_before.get("near_duplicates", 0)
        if action.action_type == "fix_missing":
            return issues_after.get("missing_values", 0) < issues_before.get("missing_values", 0)
        if action.action_type == "fix_format":
            return issues_after.get("wrong_format", 0) < issues_before.get("wrong_format", 0) or issues_after.get("inconsistent_values", 0) < issues_before.get("inconsistent_values", 0)
        if action.action_type == "fix_type":
            return issues_after.get("wrong_type", 0) < issues_before.get("wrong_type", 0)
        if action.action_type == "fix_outlier":
            return issues_after.get("outliers", 0) < issues_before.get("outliers", 0)
        return self._dataset_match_score(self._dataset) > self._dataset_match_score(self._last_before_dataset)

    def _grade_current_dataset(self) -> float:
        payload = {
            "fixed_missing_values": self._issues_fixed.get("missing_values", 0),
            "original_missing_values": self._original_issues.get("missing_values", 0),
            "fixed_duplicates": self._issues_fixed.get("duplicates", 0),
            "original_duplicates": self._original_issues.get("duplicates", 0),
            "fixed_wrong_format": self._issues_fixed.get("wrong_format", 0),
            "original_wrong_format": self._original_issues.get("wrong_format", 0),
            "fixed_wrong_type": self._issues_fixed.get("wrong_type", 0),
            "original_wrong_type": self._original_issues.get("wrong_type", 0),
            "fixed_invalid_email": self._issues_fixed.get("invalid_email", 0),
            "original_invalid_email": self._original_issues.get("invalid_email", 0),
            "fixed_outliers": self._issues_fixed.get("outliers", 0),
            "original_outliers": self._original_issues.get("outliers", 0),
            "fixed_near_duplicates": self._issues_fixed.get("near_duplicates", 0),
            "original_near_duplicates": self._original_issues.get("near_duplicates", 0),
            "fixed_inconsistent_values": self._issues_fixed.get("inconsistent_values", 0),
            "original_inconsistent_values": self._original_issues.get("inconsistent_values", 0),
            "destructive_actions": self._destructive_actions,
        }
        if self._task_level == "easy":
            return grade_easy(self._dataset, self._ground_truth, payload, self._state.step_count, self._max_steps)
        if self._task_level == "medium":
            return grade_medium(self._dataset, self._ground_truth, payload, self._state.step_count, self._max_steps)
        return grade_hard(self._dataset, self._ground_truth, payload, self._state.step_count, self._max_steps)

    def _detect_issues(self, dataset: List[Dict[str, Any]]) -> Dict[str, int]:
        issues = {
            "missing_values": 0,
            "duplicates": 0,
            "wrong_format": 0,
            "wrong_type": 0,
            "invalid_email": 0,
            "outliers": 0,
            "near_duplicates": 0,
            "inconsistent_values": 0,
        }

        signatures: Dict[str, int] = {}
        ids_seen: Dict[Any, List[Dict[str, Any]]] = {}
        for row in dataset:
            signature = repr(self._normalized_row(row))
            signatures[signature] = signatures.get(signature, 0) + 1
            ids_seen.setdefault(row.get("id"), []).append(row)

            for column, value in row.items():
                if self._is_missing(value):
                    issues["missing_values"] += 1
                    continue
                if column == "hire_date" and not self._is_iso_date(value):
                    issues["wrong_format"] += 1
                elif column == "phone" and not self._is_phone_format(value):
                    issues["wrong_format"] += 1
                elif column == "age" and not isinstance(value, int):
                    issues["wrong_type"] += 1
                elif column == "email" and not self._is_valid_email(value):
                    issues["invalid_email"] += 1
                elif column in {"age", "salary"} and self._is_outlier_value(column, value):
                    issues["outliers"] += 1
                elif column == "department" and self._canonical_department(value) != value:
                    issues["inconsistent_values"] += 1
                elif column == "name" and isinstance(value, str) and value != value.title():
                    issues["inconsistent_values"] += 1
                elif column == "is_active" and not isinstance(value, bool):
                    issues["inconsistent_values"] += 1

        issues["duplicates"] = sum(count - 1 for count in signatures.values() if count > 1)
        issues["near_duplicates"] = sum(max(0, len(rows) - 1) for rows in ids_seen.values() if len(rows) > 1 and len({repr(self._normalized_row(row)) for row in rows}) > 1)
        return issues

    def _dataset_match_score(self, dataset: List[Dict[str, Any]]) -> float:
        current_rows = sorted([self._normalized_row(row) for row in dataset], key=lambda row: (str(row.get("id")), str(row.get("name")), str(row.get("email"))))
        truth_rows = sorted([self._normalized_row(row) for row in self._ground_truth], key=lambda row: (str(row.get("id")), str(row.get("name")), str(row.get("email"))))
        total_rows = max(len(current_rows), len(truth_rows))
        total_cells = max(total_rows * len(COLUMN_SCHEMA), 1)
        matches = 0
        for index in range(total_rows):
            current_row = current_rows[index] if index < len(current_rows) else {}
            truth_row = truth_rows[index] if index < len(truth_rows) else {}
            for column in COLUMN_SCHEMA:
                if current_row.get(column) == truth_row.get(column):
                    matches += 1
        return matches / total_cells

    def _normalized_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        normalized = copy.deepcopy(row)
        normalized["name"] = str(normalized.get("name", "")).strip()
        normalized["department"] = self._canonical_department(normalized.get("department"))
        normalized["hire_date"] = self._format_value("hire_date", normalized.get("hire_date"))
        normalized["phone"] = self._format_value("phone", normalized.get("phone"))
        normalized["email"] = str(normalized.get("email", "")).strip().lower() if normalized.get("email") is not None else None
        normalized["is_active"] = self._cast_to_schema("is_active", normalized.get("is_active"))
        normalized["age"] = self._cast_to_schema("age", normalized.get("age"))
        normalized["salary"] = self._cast_to_schema("salary", normalized.get("salary"))
        return normalized

    def _is_duplicate_row(self, row_index: int) -> bool:
        target = repr(self._normalized_row(self._dataset[row_index]))
        return sum(1 for row in self._dataset if repr(self._normalized_row(row)) == target) > 1

    def _is_near_duplicate_row(self, row_index: int) -> bool:
        row = self._dataset[row_index]
        return sum(1 for item in self._dataset if item.get("id") == row.get("id")) > 1

    def _is_missing(self, value: Any) -> bool:
        return value is None or value == ""

    def _is_iso_date(self, value: Any) -> bool:
        return isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None

    def _is_phone_format(self, value: Any) -> bool:
        return isinstance(value, str) and re.fullmatch(r"\d{3}-\d{3}-\d{4}", value) is not None

    def _is_valid_email(self, value: Any) -> bool:
        return isinstance(value, str) and re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value) is not None

    def _is_outlier_value(self, column: str, value: Any) -> bool:
        if column == "age":
            return isinstance(value, int) and (value < 18 or value > 80)
        if column == "salary":
            return isinstance(value, (int, float)) and (value <= 0 or value > 300000)
        return False

    def _format_value(self, column: str, value: Any) -> Any:
        if value is None:
            return None
        if column == "hire_date":
            for date_format in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"):
                try:
                    return datetime.strptime(str(value), date_format).strftime("%Y-%m-%d")
                except ValueError:
                    continue
            return value
        if column == "phone":
            digits = re.sub(r"\D", "", str(value))
            if len(digits) == 10:
                return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
            return value
        if column == "department":
            return self._canonical_department(value)
        if column == "name" and isinstance(value, str):
            return value.title()
        return value

    def _cast_to_schema(self, column: str, value: Any) -> Any:
        if value is None or value == "":
            return value
        try:
            if column in {"id", "age"}:
                return int(float(str(value)))
            if column == "salary":
                return round(float(str(value)), 2)
            if column == "is_active":
                if isinstance(value, bool):
                    return value
                lowered = str(value).strip().lower()
                if lowered in {"true", "1", "yes"}:
                    return True
                if lowered in {"false", "0", "no"}:
                    return False
            if column == "department":
                return self._canonical_department(value)
        except (TypeError, ValueError):
            return value
        return value

    def _canonical_department(self, value: Any) -> Any:
        if value is None:
            return None
        cleaned = str(value).strip()
        return CANONICAL_DEPARTMENTS.get(cleaned.lower(), cleaned.title())
