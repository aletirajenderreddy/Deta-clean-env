from __future__ import annotations

import os
import sys
from typing import Any, Generic, Optional, TypeVar

import httpx
from pydantic import BaseModel

# Ensure project root on path
_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from models import CleaningAction, DataCleanState, DatasetObservation

try:
    from openenv.core.http_env_client import HTTPEnvClient
    from openenv.core.types import StepResult
except ImportError:
    ActionType = TypeVar("ActionType")
    ObservationType = TypeVar("ObservationType")

    class StepResult(BaseModel):
        observation: DatasetObservation
        reward: float = 0.0
        done: bool

    class HTTPEnvClient(Generic[ActionType, ObservationType]):
        def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
            self.base_url = base_url.rstrip("/")
            self.timeout = timeout
            self._client: Optional[httpx.Client] = None

        def __enter__(self):
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
            return self

        def __exit__(self, *_args: Any):
            if self._client is not None:
                self._client.close()
                self._client = None

        def _http(self) -> httpx.Client:
            if self._client is None:
                self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
            return self._client


def _parse_step_result(payload: dict) -> StepResult:
    """
    The OpenEnv server returns:
      { "observation": {...obs fields...}, "reward": float|null, "done": bool }

    DatasetObservation also contains 'done' and 'reward' fields that are required,
    so we merge the top-level done/reward into the observation dict before parsing.
    """
    obs_dict = dict(payload.get("observation", {}))
    raw_reward = payload.get("reward")
    reward: float = float(raw_reward) if raw_reward is not None else 0.0
    done = bool(payload.get("done", False))

    # Inject into observation so DatasetObservation validates correctly
    obs_dict.setdefault("done", done)
    obs_dict.setdefault("reward", raw_reward)  # keep None for Optional[float]

    observation = DatasetObservation(**obs_dict)
    return StepResult(observation=observation, reward=reward, done=done)


class DataCleanEnv(HTTPEnvClient[CleaningAction, DatasetObservation]):
    def _step_payload(self, action: CleaningAction) -> dict:
        return {
            "action_type": action.action_type,
            "row_index": action.row_index,
            "column_name": action.column_name,
            "new_value": action.new_value,
            "reason": action.reason,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        return _parse_step_result(payload)

    def _parse_state(self, payload: dict) -> DataCleanState:
        return DataCleanState(**payload)

    def reset(self, task_level: str = "easy") -> StepResult:
        # OpenEnv reset: POST /reset with {"seed": null, "episode_id": null, ...kwargs}
        # task_level is passed as an extra kwarg that maps to the Environment.reset(**kwargs)
        response = self._http().post(
            "/reset",
            json={"task_level": task_level},
        )
        response.raise_for_status()
        return _parse_step_result(response.json())

    def step(self, action: CleaningAction) -> StepResult:
        # OpenEnv step: POST /step with {"action": {...action_fields...}}
        response = self._http().post(
            "/step",
            json={"action": self._step_payload(action)},
        )
        response.raise_for_status()
        return _parse_step_result(response.json())

    def state(self) -> DataCleanState:
        response = self._http().get("/state")
        response.raise_for_status()
        return self._parse_state(response.json())

