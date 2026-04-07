from __future__ import annotations

from models import CleaningAction
from server.reward import calculate_reward


def test_correct_fix_positive_reward():
    reward = calculate_reward(
        action=CleaningAction(action_type="fix_missing", row_index=0, column_name="age", new_value="30", reason="Fix age"),
        was_valid=True,
        was_correct_fix=True,
        issues_before=5,
        issues_after=4,
        step=1,
        max_steps=30,
        task_level="easy",
    )
    assert reward > 0


def test_wrong_fix_negative_reward():
    reward = calculate_reward(
        action=CleaningAction(action_type="fix_type", row_index=0, column_name="age", new_value="bad", reason="Wrong type"),
        was_valid=True,
        was_correct_fix=False,
        issues_before=5,
        issues_after=5,
        step=1,
        max_steps=30,
        task_level="medium",
    )
    assert reward < 0


def test_no_op_negative_reward():
    reward = calculate_reward(
        action=CleaningAction(action_type="no_op", row_index=-1, column_name="id", new_value=None, reason="Skip"),
        was_valid=True,
        was_correct_fix=False,
        issues_before=3,
        issues_after=3,
        step=2,
        max_steps=30,
        task_level="easy",
    )
    assert reward < 0


def test_invalid_action_negative_reward():
    reward = calculate_reward(
        action=CleaningAction(action_type="fix_missing", row_index=99, column_name="age", new_value="10", reason="Invalid row"),
        was_valid=False,
        was_correct_fix=False,
        issues_before=4,
        issues_after=4,
        step=1,
        max_steps=30,
        task_level="easy",
    )
    assert reward < 0


def test_reward_range_is_bounded():
    reward = calculate_reward(
        action=CleaningAction(action_type="fix_outlier", row_index=1, column_name="salary", new_value="90000", reason="Fix outlier"),
        was_valid=True,
        was_correct_fix=True,
        issues_before=8,
        issues_after=2,
        step=1,
        max_steps=80,
        task_level="hard",
    )
    assert -0.3 <= reward <= 0.5
