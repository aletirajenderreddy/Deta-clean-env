from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime

import httpx

# Ensure project root on path
_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from client import DataCleanEnv
from models import CleaningAction
from server.graders import grade_easy
from server.dataset_generator import generate_dataset


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


def _pick_action(observation) -> CleaningAction:
    dataset = observation.dataset

    duplicate_index = _find_duplicate_index(dataset)
    if duplicate_index is not None:
        return CleaningAction(
            action_type="remove_duplicate",
            row_index=duplicate_index,
            column_name="id",
            new_value=None,
            reason="Remove the exact duplicate row first.",
        )

    ages = [row["age"] for row in dataset if isinstance(row.get("age"), int)]
    median_age = int(statistics.median(ages)) if ages else 30

    for index, row in enumerate(dataset):
        if row.get("age") in (None, ""):
            return CleaningAction(
                action_type="fix_missing",
                row_index=index,
                column_name="age",
                new_value=str(median_age),
                reason="Fill missing age with the median observed age.",
            )
        if row.get("email") in (None, ""):
            return CleaningAction(
                action_type="fix_missing",
                row_index=index,
                column_name="email",
                new_value=_derived_email(row["name"]),
                reason="Reconstruct the missing email from the employee name pattern.",
            )
        if isinstance(row.get("hire_date"), str) and "/" in row["hire_date"]:
            fixed_date = datetime.strptime(row["hire_date"], "%Y/%m/%d").strftime("%Y-%m-%d")
            return CleaningAction(
                action_type="fix_format",
                row_index=index,
                column_name="hire_date",
                new_value=fixed_date,
                reason="Normalize the hire_date field to YYYY-MM-DD.",
            )

    return CleaningAction(
        action_type="no_op",
        row_index=-1,
        column_name="id",
        new_value=None,
        reason="No remaining easy-task issues were detected.",
    )


def _wait_for_server(base_url: str, timeout_seconds: float = 15.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=1.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.25)
    raise RuntimeError("Timed out waiting for the local server to start.")


def run_demo(base_url: str | None) -> None:
    process = None
    if base_url is None:
        base_url = "http://127.0.0.1:8000"
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "8000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_for_server(base_url)

    try:
        with DataCleanEnv(base_url=base_url) as env:
            result = env.reset(task_level="easy")
            before_dataset = json.loads(json.dumps(result.observation.dataset))
            original_issues = result.observation.issues_remaining.copy()
            print("Before")
            print(json.dumps(before_dataset, indent=2))

            total_steps = 0
            while not result.done:
                action = _pick_action(result.observation)
                result = env.step(action)
                total_steps += 1
                print(
                    f"Action: {action.action_type} | Reward: {result.reward:.4f} "
                    f"| Issues Remaining: {result.observation.issues_remaining}"
                )

            print("\nAfter")
            print(json.dumps(result.observation.dataset, indent=2))

            # Compute final score using grader directly (real OpenEnv /state only returns minimal state)
            issues_fixed = result.observation.issues_fixed
            score_payload = {
                "fixed_missing_values": issues_fixed.get("missing_values", 0),
                "original_missing_values": original_issues.get("missing_values", 0),
                "fixed_duplicates": issues_fixed.get("duplicates", 0),
                "original_duplicates": original_issues.get("duplicates", 0),
            }
            final_score = grade_easy(
                dataset=result.observation.dataset,
                ground_truth=[],
                issues_fixed=score_payload,
                steps_taken=total_steps,
                max_steps=result.observation.max_steps,
            )
            print(f"\nFinal score (easy grader): {final_score:.3f}")
            print(f"Steps used: {total_steps} / {result.observation.max_steps}")
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DataCleanEnv rule-based demo.")
    parser.add_argument("--base-url", default=None, help="Connect to an existing local or HF Space server.")
    args = parser.parse_args()
    run_demo(base_url=args.base_url)
