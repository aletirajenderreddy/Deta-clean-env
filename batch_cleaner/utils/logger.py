"""Structured logging for batch cleaning jobs."""
from __future__ import annotations
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


LOG_DIR = Path(__file__).parent.parent / "reports"


class CleaningLogger:
    """Maintains a structured log for one cleaning job and writes TXT + JSON output."""

    def __init__(self, job_id: str, log_dir: Optional[Path] = None):
        self.job_id = job_id
        self.log_dir = Path(log_dir or LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.entries: List[Dict[str, Any]] = []
        self._started = datetime.utcnow()

        # Also wire into Python logging
        self._logger = logging.getLogger(f"batch_cleaner.{job_id}")
        if not self._logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
            self._logger.addHandler(h)
        self._logger.setLevel(logging.DEBUG)

    def _entry(self, level: str, event: str, detail: str, rows_affected: int = 0, extra: Optional[Dict] = None) -> Dict:
        e = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": level, "event": event,
            "detail": detail, "rows_affected": rows_affected,
        }
        if extra:
            e.update(extra)
        self.entries.append(e)
        getattr(self._logger, level.lower(), self._logger.info)(f"[{event}] {detail}")
        return e

    def info(self, event: str, detail: str, rows_affected: int = 0, **kwargs):
        return self._entry("INFO", event, detail, rows_affected, kwargs or None)

    def warning(self, event: str, detail: str, rows_affected: int = 0, **kwargs):
        return self._entry("WARNING", event, detail, rows_affected, kwargs or None)

    def error(self, event: str, detail: str, **kwargs):
        return self._entry("ERROR", event, detail, 0, kwargs or None)

    # Convenience helpers
    def rows_removed(self, reason: str, count: int):
        return self.info("rows_removed", f"{count} rows removed: {reason}", rows_affected=count)

    def nulls_filled(self, col: str, count: int, strategy: str):
        return self.info("nulls_filled", f"'{col}': {count} nulls filled via '{strategy}'", rows_affected=count)

    def outliers_detected(self, col: str, count: int, strategy: str):
        return self.info("outliers_detected", f"'{col}': {count} outliers handled via '{strategy}'", rows_affected=count)

    def save(self) -> Dict[str, Path]:
        """Save logs as JSON and TXT. Returns paths."""
        elapsed = (datetime.utcnow() - self._started).total_seconds()
        payload = {
            "job_id": self.job_id,
            "started_at": self._started.isoformat() + "Z",
            "elapsed_seconds": round(elapsed, 3),
            "total_log_entries": len(self.entries),
            "entries": self.entries,
        }
        json_path = self.log_dir / f"{self.job_id}_log.json"
        txt_path  = self.log_dir / f"{self.job_id}_log.txt"

        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        lines = [
            f"DataCleanEnv — Cleaning Job Log",
            f"Job ID   : {self.job_id}",
            f"Started  : {self._started.isoformat()}Z",
            f"Elapsed  : {elapsed:.3f}s",
            "=" * 60,
        ]
        for e in self.entries:
            row_note = f" (rows affected: {e['rows_affected']})" if e.get("rows_affected") else ""
            lines.append(f"[{e['ts'][:19]}] {e['level']:7s} [{e['event']}] {e['detail']}{row_note}")
        txt_path.write_text("\n".join(lines), encoding="utf-8")

        return {"json": json_path, "txt": txt_path}
