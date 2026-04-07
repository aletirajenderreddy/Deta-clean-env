from __future__ import annotations

import json
import os
import re
from typing import Dict

from openai import OpenAI

from client import DataCleanEnv
from models import CleaningAction


SYSTEM_PROMPT = """You are a data cleaning AI agent. You will receive a dataset with issues.
Identify ONE issue and fix it. Available actions:
- fix_missing: Fill in a missing value with appropriate data
- fix_format: Correct a wrongly formatted value (e.g., date format)
- remove_duplicate: Remove a duplicate row (set row_index to the duplicate)
- fix_type: Convert a value to the correct data type
- fix_outlier: Replace an outlier value with a reasonable one
- no_op: No action needed this step
Always respond with valid JSON only."""


def _parse_json_from_response(text: str) -> dict:
    """Extract a JSON object from GPT response text, tolerating markdown fences."""
    # Strip markdown code fences if present
    stripped = re.sub(r"```(?:json)?", "", text).strip()
    return json.loads(stripped)


def _request_action(client: OpenAI, model: str, observation_payload: dict) -> CleaningAction:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Analyze the dataset and fix one issue at a time. "
                    "Return JSON with keys: action_type, row_index, column_name, new_value, reason\n"
                    + json.dumps(observation_payload, indent=2)
                ),
            },
        ],
    )
    raw = response.choices[0].message.content or "{}"
    payload = _parse_json_from_response(raw)
    return CleaningAction(**payload)


def run_baseline(base_url: str, model: str) -> Dict[str, float]:
    openai_client = OpenAI(api_key=os.environ[open-ai-api-key])
    scores: Dict[str, float] = {}

    with DataCleanEnv(base_url=base_url) as env:
        for task_level in ["easy", "medium", "hard"]:
            result = env.reset(task_level=task_level)
            steps = 0
            while not result.done:
                action = _request_action(openai_client, model, result.observation.model_dump())
                result = env.step(action)
                steps += 1
            score = env.state().score
            scores[task_level] = score
            print(f"Task: {task_level} | Score: {score:.3f} | Steps: {steps}")

    print("\nSummary")
    print(f"{'Task':<10} | {'Score':>6}")
    print("-" * 20)
    for task_level in ["easy", "medium", "hard"]:
        print(f"{task_level:<10} | {scores[task_level]:.3f}")
    return scores


if __name__ == "__main__":
    base_url = os.getenv("DATACLEANENV_BASE_URL", "http://localhost:8000")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    run_baseline(base_url=base_url, model=model)
