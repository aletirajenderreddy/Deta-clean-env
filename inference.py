"""
inference.py — OpenEnv-compatible inference script for DataCleanEnv

Uses the hackathon's LiteLLM proxy (API_BASE_URL + API_KEY env vars)
and emits [START] / [STEP] / [END] blocks to stdout as required by Phase 2.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from models import CleaningAction, DatasetObservation
from server.environment import DataCleanEnvironment

# ---------------------------------------------------------------------------
# LLM client — always uses the hackathon-injected proxy credentials
# ---------------------------------------------------------------------------

def get_llm_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
    )


# ---------------------------------------------------------------------------
# Rule-based fallback (used when LLM response cannot be parsed)
# ---------------------------------------------------------------------------

def _derived_email(name: str) -> str:
    parts = [p.lower() for p in name.replace("'", "").split()]
    if len(parts) == 1:
        return f"{parts[0]}@example.com"
    return f"{parts[0]}.{parts[-1]}@example.com"


def _find_duplicate_index(dataset: list) -> int | None:
    seen: dict = {}
    for idx, row in enumerate(dataset):
        key = json.dumps(row, sort_keys=True, default=str)
        if key in seen:
            return idx
        seen[key] = idx
    return None


def _rule_based_action(observation: DatasetObservation) -> CleaningAction:
    """Deterministic fallback agent when the LLM fails."""
    dataset = observation.dataset

    # 1. duplicates
    dup = _find_duplicate_index(dataset)
    if dup is not None:
        return CleaningAction(
            action_type="remove_duplicate",
            row_index=dup,
            column_name="id",
            new_value=None,
            reason="Remove exact duplicate row.",
        )

    # 2. missing age / email / hire_date
    ages = [r["age"] for r in dataset if isinstance(r.get("age"), int)]
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
                new_value=_derived_email(str(row.get("name", "user"))),
                reason="Reconstruct missing email from name.",
            )
        if isinstance(row.get("hire_date"), str) and "/" in row["hire_date"]:
            from datetime import datetime
            try:
                fixed = datetime.strptime(row["hire_date"], "%Y/%m/%d").strftime("%Y-%m-%d")
                return CleaningAction(
                    action_type="fix_format",
                    row_index=idx,
                    column_name="hire_date",
                    new_value=fixed,
                    reason="Normalize hire_date to ISO format.",
                )
            except ValueError:
                pass

    return CleaningAction(
        action_type="no_op",
        row_index=-1,
        column_name="id",
        new_value=None,
        reason="No issues detected by rule engine.",
    )


# ---------------------------------------------------------------------------
# LLM-based action picker
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert data cleaning agent.
You will be given a sample of a messy dataset and a summary of remaining issues.
Your job is to choose the single best cleaning action to take next.

Available action_types:
  fix_missing       — fill a null/empty cell
  fix_format        — correct wrong format (date, phone, email)
  remove_duplicate  — delete a duplicate row
  fix_type          — convert a wrong data type
  fix_outlier       — replace an impossible value (e.g. age=999)
  no_op             — do nothing (only if no issues remain)

Column schema:
  id: int, name: str, age: int (18-80), email: email,
  phone: str (format: 123-456-7890), salary: float (>0, <=300000),
  hire_date: str (YYYY-MM-DD), department: str, is_active: bool

Reply with ONLY a single valid JSON object — no markdown, no explanation:
{"action_type": "...", "row_index": 0, "column_name": "...", "new_value": "...", "reason": "..."}
Set new_value to null for remove_duplicate or no_op.
"""


def _pick_action_llm(observation: DatasetObservation, client: OpenAI) -> CleaningAction:
    """Ask the LLM proxy for the next cleaning action."""

    # Build compact context — send first 8 rows + issue summary
    sample = observation.dataset[:8]
    issues = {k: v for k, v in observation.issues_remaining.items() if v > 0}

    user_msg = (
        f"Issues remaining: {json.dumps(issues)}\n"
        f"Step {observation.current_step}/{observation.max_steps}\n"
        f"Dataset sample ({len(sample)} rows):\n"
        f"{json.dumps(sample, indent=2)}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=200,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

        data = json.loads(raw)
        return CleaningAction(
            action_type=data.get("action_type", "no_op"),
            row_index=int(data.get("row_index", -1)),
            column_name=str(data.get("column_name", "id")),
            new_value=data.get("new_value"),
            reason=str(data.get("reason", "LLM decision")),
        )

    except Exception as exc:  # parse error, timeout, etc.
        print(f"# LLM fallback: {exc}", file=sys.stderr, flush=True)
        return _rule_based_action(observation)


# ---------------------------------------------------------------------------
# Main episode runner
# ---------------------------------------------------------------------------

def run_inference(task_level: str = "easy") -> None:
    """Run one full RL episode and emit required structured output to stdout."""
    env = DataCleanEnvironment()
    obs = env.reset(task_level=task_level)
    client = get_llm_client()

    print(f"[START] task={task_level}", flush=True)

    steps = 0
    total_reward = 0.0

    while not obs.done:
        action = _pick_action_llm(obs, client)
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
