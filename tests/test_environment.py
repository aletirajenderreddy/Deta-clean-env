from __future__ import annotations

import copy

from models import CleaningAction
from server.environment import DataCleanEnvironment


def _find_missing_cell(dataset):
    for index, row in enumerate(dataset):
        for column in ("age", "email"):
            if row.get(column) in (None, ""):
                return index, column
    raise AssertionError("No missing cell found in easy dataset")


def test_reset_returns_clean_state():
    env = DataCleanEnvironment()
    observation = env.reset(task_level="easy")
    assert observation.done is False
    assert observation.current_step == 0
    assert observation.task_level == "easy"
    assert env.state.total_issues_original > 0


def test_step_fix_missing_gives_positive_reward():
    env = DataCleanEnvironment()
    env.reset(task_level="easy")
    row_index, column_name = _find_missing_cell(env._dataset)
    correct_value = env._ground_truth[row_index][column_name]
    observation = env.step(
        CleaningAction(
            action_type="fix_missing",
            row_index=row_index,
            column_name=column_name,
            new_value=str(correct_value),
            reason="Fill the missing value with the known correct value.",
        )
    )
    assert observation.reward is not None and observation.reward > 0


def test_step_no_op_gives_negative_reward():
    env = DataCleanEnvironment()
    env.reset(task_level="easy")
    observation = env.step(
        CleaningAction(action_type="no_op", row_index=-1, column_name="id", new_value=None, reason="Skip one step.")
    )
    assert observation.reward is not None and observation.reward < 0


def test_step_wrong_fix_gives_negative_reward():
    env = DataCleanEnvironment()
    env.reset(task_level="easy")
    # Apply a fix_format to a non-format column with a nonsense value.
    # This will succeed as an "apply" operation but won't reduce any issue counts.
    observation = env.step(
        CleaningAction(
            action_type="fix_format",
            row_index=0,
            column_name="age",
            new_value="not-a-number",
            reason="Deliberately wrong fix to test negative reward.",
        )
    )
    assert observation.reward is not None and observation.reward < 0


def test_episode_ends_when_all_issues_fixed():
    env = DataCleanEnvironment()
    env.reset(task_level="easy")
    env._dataset = copy.deepcopy(env._ground_truth)
    env._issues_remaining = env._detect_issues(env._dataset)
    observation = env.step(
        CleaningAction(action_type="no_op", row_index=-1, column_name="id", new_value=None, reason="Check completion.")
    )
    assert observation.done is True


def test_max_steps_terminates_episode():
    env = DataCleanEnvironment()
    env.reset(task_level="easy")
    env._max_steps = 1
    observation = env.step(
        CleaningAction(action_type="no_op", row_index=-1, column_name="id", new_value=None, reason="Spend the only step.")
    )
    assert observation.done is True
