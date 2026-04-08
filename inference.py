"""
inference.py — OpenEnv-compatible inference script for DataCleanEnv

This script is required by the OpenEnv hackathon checker.
It demonstrates running a full RL episode using the environment.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

# Ensure root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import CleaningAction, DatasetObservation
from server.environment import DataCleanEnvironment


def _derived_email(name: str) -> str:
    parts = [part.lower() for part in name.replace("'", "").split()]
    if len(parts) == 1:
        return f"{parts[0]}@example.com"
    return f"{parts[0]}.{parts[-1]}@example.com"


def _find_duplicate_index(dataset):
    seen = {}
    for index, row in enumerate(dataset):
        marker = json.dumps(row, sort_keys=True, default=str)
        if marker in seen:
            return index
        seen[marker] = index
    return None


def pick_action(observation: DatasetObservation) -> CleaningAction:
    """Rule-based agent: picks the next best cleaning action."""
    dataset = observation.dataset

    # 1. Remove duplicates first
    dup_idx = _find_duplicate_index(dataset)
    if dup_idx is not None:
        return CleaningAction(
            action_type="remove_duplicate",
            row_index=dup_idx,
            column_name="id",
            new_value=None,
            reason="Remove exact duplicate row.",
        )

    # 2. Fix missing values
    ages = [row["age"] for row in dataset if isinstance(row.get("age"), int)]
    median_age = int(statistics.median(ages)) if ages else 30

    for idx, row in enumerate(dataset):
        if row.get("age") in (None, ""):
            return CleaningAction(
                action_type="fix_missing",
                row_index=idx,
                column_name="age",
                new_value=str(median_age),
                reason="Fill missing age with median.",
            )
        if row.get("email") in (None, ""):
            return CleaningAction(
                action_type="fix_missing",
                row_index=idx,
                column_name="email",
                new_value=_derived_email(row["name"]),
                reason="Reconstruct missing email from name.",
            )
        if isinstance(row.get("hire_date"), str) and "/" in row["hire_date"]:
            from datetime import datetime
            fixed = datetime.strptime(row["hire_date"], "%Y/%m/%d").strftime("%Y-%m-%d")
            return CleaningAction(
                action_type="fix_format",
                row_index=idx,
                column_name="hire_date",
                new_value=fixed,
                reason="Normalize hire_date to YYYY-MM-DD.",
            )

    return CleaningAction(
        action_type="no_op",
        row_index=-1,
        column_name="id",
        new_value=None,
        reason="No remaining issues detected.",
    )


def run_inference(task_level: str = "easy") -> None:
    """Run one full RL episode and emit [START]/[STEP]/[END] to stdout."""
    env = DataCleanEnvironment()
    obs = env.reset(task_level=task_level)

    print(f"[START] task={task_level}", flush=True)

    steps = 0
    total_reward = 0.0

    while not obs.done:
        action = pick_action(obs)
        obs = env.step(action)
        steps += 1
        reward = obs.reward if obs.reward is not None else 0.0
        total_reward += reward
        print(f"[STEP] step={steps} reward={round(reward, 4)}", flush=True)

    score = round(total_reward / max(steps, 1), 4)
    print(f"[END] task={task_level} score={score} steps={steps}", flush=True)


if __name__ == "__main__":
    for level in ["easy", "medium", "hard"]:
        run_inference(task_level=level)