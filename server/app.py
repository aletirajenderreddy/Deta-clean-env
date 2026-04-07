from __future__ import annotations

import os
import pathlib
import sys

# Ensure project root (datacleanenv/) is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from models import CleaningAction, DatasetObservation
from server.environment import DataCleanEnvironment

# ── Shared RL environment instance ──
_environment = DataCleanEnvironment()

# ── App ──
try:
    from openenv.core.env_server import create_fastapi_app
    app = create_fastapi_app(
        env=DataCleanEnvironment,
        action_cls=CleaningAction,
        observation_cls=DatasetObservation,
    )
except (ImportError, Exception):
    app = FastAPI(
        title="DataCleanEnv",
        version="1.0.0",
        description="RL Environment + Batch Data Cleaning System with AI Integration",
    )

    @app.get("/health")
    def health():
        return {"status": "healthy", "version": "1.0.0"}

    @app.post("/reset")
    def reset(payload: dict | None = None):
        level = (payload or {}).get("task_level", "easy")
        obs = _environment.reset(task_level=level)
        return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}

    @app.post("/step")
    def step(body: dict):
        action_data = body.get("action", body)
        action = CleaningAction(**action_data)
        obs = _environment.step(action)
        return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}

    @app.get("/state")
    def state():
        return _environment.state.model_dump()

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Batch Cleaning API ──
try:
    from batch_cleaner.api.routes import router as batch_router
    app.include_router(batch_router)
    print("[OK] Batch Cleaner API loaded at /batch/*")
except ImportError as e:
    print(f"[WARNING] batch_cleaner not loaded: {e}")

# ── Dashboard ──
_TMPL_DIR = pathlib.Path(__file__).parent / "templates"
_DASHBOARD = _TMPL_DIR / "dashboard.html"
_LEGACY_DASH = pathlib.Path(_ROOT) / "dashboard.html"


def _read_dashboard() -> str:
    if _DASHBOARD.exists():
        return _DASHBOARD.read_text(encoding="utf-8")
    if _LEGACY_DASH.exists():
        return _LEGACY_DASH.read_text(encoding="utf-8")
    return "<h1>dashboard.html not found</h1>"


@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
async def web_dashboard():
    return HTMLResponse(content=_read_dashboard())


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_alias():
    return HTMLResponse(content=_read_dashboard())


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return HTMLResponse(content=_read_dashboard())
