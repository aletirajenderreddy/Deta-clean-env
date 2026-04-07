"""FastAPI router for batch cleaning — /batch/* endpoints."""
from __future__ import annotations
import io
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse

from batch_cleaner.ingestion.loader import load_file
from batch_cleaner.profiling.profiler import profile, to_json, to_html
from batch_cleaner.cleaning.engine import CleaningEngine, load_config
from batch_cleaner.validation.validator import SchemaValidator
from batch_cleaner.ai_module.advisor import AIAdvisor
from batch_cleaner.utils.logger import CleaningLogger

router = APIRouter(prefix="/batch", tags=["batch"])

# In-memory job store (use Redis/DB in production)
_JOBS: Dict[str, Dict[str, Any]] = {}
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _new_job(name: str) -> str:
    jid = str(uuid.uuid4())[:8]
    _JOBS[jid] = {"id": jid, "name": name, "status": "created", "created_at": datetime.utcnow().isoformat() + "Z"}
    return jid


def _get_job(job_id: str) -> Dict:
    if job_id not in _JOBS:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return _JOBS[job_id]


# ── UPLOAD ────────────────────────────────────────────────────────────────────
@router.post("/upload", summary="Upload a dataset (CSV / Excel / JSON)")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file and return a job_id for subsequent operations."""
    jid = _new_job(file.filename or "upload")
    contents = await file.read()
    try:
        df = load_file(contents, filename=file.filename)
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}")

    # Save raw to disk (CSV to avoid type coercion issues)
    raw_path = REPORTS_DIR / f"{jid}_raw.csv"
    df.to_csv(raw_path, index=False)

    _JOBS[jid].update({
        "status": "uploaded",
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "raw_path": str(raw_path),
        "raw_fmt": "csv",
    })
    return {"job_id": jid, "rows": len(df), "columns": list(df.columns), "status": "uploaded"}


# ── PROFILE ───────────────────────────────────────────────────────────────────
@router.post("/profile/{job_id}", summary="Generate profiling report")
async def profile_dataset(job_id: str, format: str = "json"):
    job = _get_job(job_id)
    df = pd.read_csv(job['raw_path'])
    report = profile(df)

    json_path = REPORTS_DIR / f"{job_id}_profile.json"
    html_path = REPORTS_DIR / f"{job_id}_profile.html"
    to_json(report, json_path)
    to_html(report, html_path)

    _JOBS[job_id]["profile"] = report
    _JOBS[job_id]["status"] = "profiled"

    if format == "html":
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return JSONResponse(content=report)


# ── AI SUGGEST ────────────────────────────────────────────────────────────────
@router.post("/ai-suggest/{job_id}", summary="Get AI cleaning suggestions")
async def ai_suggest(job_id: str):
    job = _get_job(job_id)
    if "profile" not in job:
        # Auto-profile first
        df = pd.read_csv(job['raw_path'])
        job["profile"] = profile(df)

    advisor = AIAdvisor(api_key=os.getenv("OPENAI_API_KEY", ""))
    suggestions = advisor.get_suggestions(job["profile"])

    sug_path = REPORTS_DIR / f"{job_id}_ai_suggestions.json"
    sug_path.write_text(json.dumps(suggestions, indent=2), encoding="utf-8")

    _JOBS[job_id]["ai_suggestions"] = suggestions
    _JOBS[job_id]["status"] = "ai_analyzed"
    return suggestions


# ── CLEAN ─────────────────────────────────────────────────────────────────────
@router.post("/clean/{job_id}", summary="Run cleaning pipeline")
async def clean_dataset(
    job_id: str,
    config_override: Optional[str] = Form(default=None),
    apply_ai_suggestions: bool = Form(default=False),
    validation_schema: Optional[str] = Form(default=None),
):
    """
    Run the cleaning engine on the uploaded dataset.
    - config_override: JSON string overriding cleaning rules
    - apply_ai_suggestions: merge AI suggestions into config
    - validation_schema: JSON schema for post-cleaning validation
    """
    job = _get_job(job_id)
    df = pd.read_csv(job['raw_path'])

    cfg = load_config()  # load defaults from YAML

    if config_override:
        try:
            override = json.loads(config_override)
            cfg.update(override)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"Invalid config JSON: {e}")

    if apply_ai_suggestions and "ai_suggestions" in job:
        advisor = AIAdvisor()
        cfg = advisor.apply_suggestions(job["ai_suggestions"], cfg)

    logger = CleaningLogger(job_id)
    logger.info("start", f"Cleaning job {job_id}: {len(df)} rows, {len(df.columns)} cols")

    engine = CleaningEngine(config=cfg)
    cleaned_df = engine.clean(df)

    # Merge engine log into cleaning logger
    for step in engine.get_log():
        logger.info(step["step"], step["detail"], rows_affected=step.get("rows_affected", 0))

    # Validation
    validation_errors = []
    if validation_schema:
        try:
            schema = json.loads(validation_schema)
            validator = SchemaValidator(schema)
            cleaned_df, v_errors = validator.validate(cleaned_df)
            validation_errors = [e.to_dict() for e in v_errors]
            logger.info("validation", f"{len(v_errors)} validation errors found")
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"Invalid validation schema: {e}")

    # Save outputs
    cleaned_csv  = REPORTS_DIR / f"{job_id}_cleaned.csv"
    cleaned_xlsx = REPORTS_DIR / f"{job_id}_cleaned.xlsx"
    cleaned_df.to_csv(cleaned_csv, index=False)
    cleaned_df.to_excel(cleaned_xlsx, index=False, engine="openpyxl")

    log_paths = logger.save()

    report = engine.get_report()
    report["validation_errors"] = validation_errors
    report_path = REPORTS_DIR / f"{job_id}_cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _JOBS[job_id].update({
        "status": "cleaned",
        "cleaned_rows": len(cleaned_df),
        "cleaned_csv": str(cleaned_csv),
        "cleaned_xlsx": str(cleaned_xlsx),
        "cleaning_report": str(report_path),
        "log_json": str(log_paths["json"]),
        "log_txt":  str(log_paths["txt"]),
        "validation_errors": len(validation_errors),
    })

    return {
        "job_id": job_id,
        "original_rows": len(df),
        "cleaned_rows": len(cleaned_df),
        "rows_removed": len(df) - len(cleaned_df),
        "cleaning_steps": len(engine.get_log()),
        "validation_errors": len(validation_errors),
        "status": "cleaned",
    }


# ── DOWNLOAD ──────────────────────────────────────────────────────────────────
@router.get("/download/{job_id}", summary="Download cleaned CSV")
async def download_csv(job_id: str):
    job = _get_job(job_id)
    if "cleaned_csv" not in job:
        raise HTTPException(400, "Dataset not cleaned yet. Call /batch/clean first.")
    return FileResponse(job["cleaned_csv"], media_type="text/csv", filename=f"{job_id}_cleaned.csv")


@router.get("/download-excel/{job_id}", summary="Download cleaned Excel")
async def download_excel(job_id: str):
    job = _get_job(job_id)
    if "cleaned_xlsx" not in job:
        raise HTTPException(400, "Dataset not cleaned yet.")
    return FileResponse(job["cleaned_xlsx"], media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=f"{job_id}_cleaned.xlsx")


@router.get("/report/{job_id}", summary="Download cleaning report JSON")
async def download_report(job_id: str):
    job = _get_job(job_id)
    if "cleaning_report" not in job:
        raise HTTPException(400, "No cleaning report available.")
    return FileResponse(job["cleaning_report"], media_type="application/json", filename=f"{job_id}_report.json")


@router.get("/log/{job_id}", summary="Download log TXT")
async def download_log(job_id: str):
    job = _get_job(job_id)
    if "log_txt" not in job:
        raise HTTPException(400, "No log available.")
    return FileResponse(job["log_txt"], media_type="text/plain", filename=f"{job_id}_log.txt")


@router.get("/profile-html/{job_id}", summary="View profiling report HTML")
async def view_profile_html(job_id: str):
    p = REPORTS_DIR / f"{job_id}_profile.html"
    if not p.exists():
        raise HTTPException(404, "Profile HTML not found. Call /batch/profile first.")
    return HTMLResponse(content=p.read_text(encoding="utf-8"))


# ── JOBS ──────────────────────────────────────────────────────────────────────
@router.get("/jobs", summary="List all cleaning jobs")
async def list_jobs():
    return list(_JOBS.values())


@router.get("/jobs/{job_id}", summary="Get job status")
async def get_job_status(job_id: str):
    return _get_job(job_id)
