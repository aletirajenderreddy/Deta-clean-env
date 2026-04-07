from __future__ import annotations


def calculate_reward(action, was_valid, was_correct_fix, issues_before, issues_after, step, max_steps, task_level) -> float:
    reward = 0.0

    if was_correct_fix:
        issue_type = action.action_type
        rewards_by_type = {
            "fix_missing": 0.15,
            "remove_duplicate": 0.12,
            "fix_format": 0.18,
            "fix_type": 0.15,
            "fix_outlier": 0.20,
        }
        reward += rewards_by_type.get(issue_type, 0.10)

        net_reduction = issues_before - issues_after
        if net_reduction > 1:
            reward += 0.05 * min(net_reduction, 3)
    elif not was_correct_fix and action.action_type != "no_op":
        reward -= 0.08

    if action.action_type == "no_op":
        reward -= 0.05

    if was_valid is False:
        reward -= 0.12

    reward -= 0.01

    reward = round(reward, 4)
    return max(-0.3, min(0.5, reward))
