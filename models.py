from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from openenv.core.env_server import Action, Observation, State
except ImportError:
    class Action(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

    class Observation(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

    class State(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)


class CleaningAction(Action):
    model_config = ConfigDict(extra="forbid")

    action_type: str = Field(
        description='One of: "fix_missing", "fix_format", "remove_duplicate", "fix_type", "fix_outlier", "no_op"'
    )
    row_index: int = Field(description="Which row to fix (-1 means apply to whole column)")
    column_name: str = Field(description="Which column to fix")
    new_value: Optional[str] = Field(default=None, description="The corrected value")
    reason: str = Field(description="Why the agent is making this fix")


class DatasetObservation(Observation):
    model_config = ConfigDict(extra="forbid")

    done: bool
    reward: Optional[float]
    dataset: List[Dict[str, Any]]
    issues_remaining: Dict[str, int]
    issues_fixed: Dict[str, int]
    total_issues_original: int
    current_step: int
    max_steps: int
    message: str
    task_level: str
    column_schema: Dict[str, str]


class DataCleanState(State):
    model_config = ConfigDict(extra="forbid")

    episode_id: Optional[str] = None
    step_count: int = 0
    task_level: str = "easy"
    total_issues_original: int = 0
    issues_fixed_count: int = 0
    score: float = 0.0
