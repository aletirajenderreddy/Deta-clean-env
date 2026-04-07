"""Data profiling module — generates statistics before cleaning."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd


def profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a full profiling report as a dict."""
    report: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": {},
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / max(df.size, 1) * 100, 2),
    }
    for col in df.columns:
        s = df[col]
        info: Dict[str, Any] = {
            "dtype": str(s.dtype),
            "null_count": int(s.isnull().sum()),
            "null_pct": round(s.isnull().sum() / max(len(s), 1) * 100, 2),
            "unique_count": int(s.nunique(dropna=True)),
        }
        # sample values (non-null)
        sample = s.dropna().head(5).tolist()
        info["sample_values"] = [str(v) for v in sample]

        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            info.update({
                "mean":   _safe(desc.get("mean")),
                "median": _safe(s.median()),
                "std":    _safe(desc.get("std")),
                "min":    _safe(desc.get("min")),
                "max":    _safe(desc.get("max")),
                "q1":     _safe(s.quantile(0.25)),
                "q3":     _safe(s.quantile(0.75)),
            })
            # IQR outliers
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = int(((s < lo) | (s > hi)).sum())
            info["outlier_count_iqr"] = outliers
            # Distribution (bucketed)
            try:
                counts, edges = np.histogram(s.dropna(), bins=10)
                info["distribution"] = {
                    "counts": counts.tolist(),
                    "edges": [round(e, 4) for e in edges.tolist()],
                }
            except Exception:
                pass
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            vc = s.value_counts().head(5)
            info["top_values"] = {str(k): int(v) for k, v in vc.items()}

        report["columns"][col] = info

    return report


def _safe(val) -> Any:
    if val is None:
        return None
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    return round(float(val), 6) if isinstance(val, (float, np.floating)) else val


def to_json(report: Dict, path: Path | None = None) -> str:
    """Serialize report to JSON string, optionally write to file."""
    s = json.dumps(report, indent=2, default=str)
    if path:
        Path(path).write_text(s, encoding="utf-8")
    return s


def to_html(report: Dict, path: Path | None = None) -> str:
    """Generate a downloadable HTML profiling report."""
    cols_html = ""
    for name, info in report.get("columns", {}).items():
        stat_rows = ""
        for k, v in info.items():
            if k not in ("distribution", "top_values", "sample_values"):
                stat_rows += f"<tr><td>{k}</td><td>{v}</td></tr>"
        cols_html += f"""
        <div class="col-card">
          <h3>{name} <span class="dtype">[{info.get('dtype','?')}]</span>
          {'<span class="badge red">'+str(info.get("null_pct",0))+'% null</span>' if info.get("null_pct",0)>0 else ''}
          {'<span class="badge orange">'+str(info.get("outlier_count_iqr",0))+' outliers</span>' if info.get("outlier_count_iqr",0)>0 else ''}
          </h3>
          <table class="stat-table">{stat_rows}</table>
        </div>"""

    html = f"""<!DOCTYPE html><html lang="en">
<head><meta charset="UTF-8"><title>DataCleanEnv — Profiling Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#0A0E1A;color:#E2E8F0;padding:24px}}
h1{{color:#00D4FF;font-size:24px;margin-bottom:6px}}
.meta{{font-size:12px;color:#888;margin-bottom:24px}}
.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:28px}}
.stat-box{{background:#111827;border:1px solid #1E2D40;padding:16px;text-align:center}}
.stat-box .val{{font-size:28px;font-weight:700;color:#00D4FF}}
.stat-box .lbl{{font-size:11px;color:#888;margin-top:4px}}
.cols{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}}
.col-card{{background:#111827;border:1px solid #1E2D40;padding:16px}}
.col-card h3{{font-size:14px;color:#E2E8F0;margin-bottom:10px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
.dtype{{font-size:10px;color:#888;font-weight:400}}.badge{{font-size:9px;padding:2px 6px;border-radius:2px}}
.badge.red{{background:rgba(255,77,77,.2);color:#FF4D4D}}.badge.orange{{background:rgba(255,176,32,.2);color:#FFB020}}
.stat-table{{width:100%;font-size:11px;border-collapse:collapse}}
.stat-table td{{padding:3px 6px;border-bottom:1px solid #1a2535;color:#aaa}}
.stat-table td:first-child{{color:#00D4FF;width:140px}}
</style></head>
<body>
<h1>DataCleanEnv — Data Profiling Report</h1>
<div class="meta">Generated: {report.get('generated_at','—')}</div>
<div class="summary">
  <div class="stat-box"><div class="val">{report['shape']['rows']}</div><div class="lbl">ROWS</div></div>
  <div class="stat-box"><div class="val">{report['shape']['columns']}</div><div class="lbl">COLUMNS</div></div>
  <div class="stat-box"><div class="val">{report['total_missing']}</div><div class="lbl">MISSING VALUES</div></div>
  <div class="stat-box"><div class="val">{report['missing_pct']}%</div><div class="lbl">MISSING %</div></div>
  <div class="stat-box"><div class="val">{report['duplicate_rows']}</div><div class="lbl">DUPLICATE ROWS</div></div>
</div>
<div class="cols">{cols_html}</div>
</body></html>"""
    if path:
        Path(path).write_text(html, encoding="utf-8")
    return html
