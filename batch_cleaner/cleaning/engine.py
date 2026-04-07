"""Cleaning engine — orchestrates rules from YAML config."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import yaml

from batch_cleaner.cleaning.rules import (
    fill_mean, fill_median, fill_mode, fill_constant,
    drop_missing_rows, drop_missing_cols,
    remove_duplicates, cast_numeric, cast_datetime,
    normalize_text, strip_whitespace, fix_email,
    cap_iqr, remove_iqr, cap_zscore,
    drop_column, drop_low_variance,
)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "cleaning_rules.yaml"


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or DEFAULT_CONFIG_PATH
    if Path(p).exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class CleaningEngine:
    """Orchestrates configurable data cleaning steps with detailed logging."""

    def __init__(self, config: Optional[Dict] = None, config_path: Optional[Path] = None):
        self.config = config or load_config(config_path)
        self.log: List[Dict[str, Any]] = []

    def _log(self, step: str, detail: str, rows_affected: int = 0):
        self.log.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "step": step,
            "detail": detail,
            "rows_affected": rows_affected,
        })

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full cleaning pipeline. Returns cleaned df."""
        self.log = []
        original_rows = len(df)
        original_nulls = int(df.isnull().sum().sum())
        cfg = self.config

        # 1. Drop columns above missing threshold
        col_threshold = cfg.get("drop_columns_missing_above", 1.0)
        before_cols = list(df.columns)
        df = drop_missing_cols(df, threshold=col_threshold)
        dropped_cols = [c for c in before_cols if c not in df.columns]
        if dropped_cols:
            self._log("drop_cols", f"Dropped {len(dropped_cols)} cols >missing threshold {col_threshold}: {dropped_cols}")

        # 2. Drop explicitly listed columns
        for col in cfg.get("drop_columns", []):
            if col in df.columns:
                df = drop_column(df, col)
                self._log("drop_col", f"Dropped column: {col}")

        # 3. Duplicates
        dup_config = cfg.get("duplicates", {})
        if dup_config.get("remove", True):
            before = len(df)
            df = remove_duplicates(df, keep=dup_config.get("keep", "first"))
            removed = before - len(df)
            if removed:
                self._log("duplicates", f"Removed {removed} duplicate rows", rows_affected=removed)

        # 4. Missing values
        fill_rules = cfg.get("fill_missing", {})
        global_strategy = cfg.get("default_fill_strategy", "mode")
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            strategy = fill_rules.get(col, global_strategy)
            null_count = int(df[col].isnull().sum())
            if strategy == "mean":
                df = fill_mean(df, col)
            elif strategy == "median":
                df = fill_median(df, col)
            elif strategy == "mode":
                df = fill_mode(df, col)
            elif strategy == "drop":
                before = len(df)
                df = drop_missing_rows(df, col=col)
                self._log("drop_rows", f"Dropped {before-len(df)} rows with null '{col}'", rows_affected=before-len(df))
                continue
            elif isinstance(strategy, (str, int, float)):
                df = fill_constant(df, col, value=strategy)
            self._log("fill_missing", f"Filled {null_count} nulls in '{col}' using '{strategy}'", rows_affected=null_count)

        # 5. Type corrections
        type_rules = cfg.get("cast_types", {})
        for col, dtype in type_rules.items():
            if col not in df.columns:
                continue
            if dtype in ("int", "float", "numeric"):
                df = cast_numeric(df, col)
                self._log("cast_type", f"Cast '{col}' to numeric")
            elif dtype in ("datetime", "date"):
                df = cast_datetime(df, col)
                self._log("cast_type", f"Cast '{col}' to datetime")

        # 6. Text normalisation
        text_cols = cfg.get("normalize_text", [])
        for col in text_cols:
            if col in df.columns and pd.api.types.is_object_dtype(df[col]):
                df = normalize_text(df, col)
                self._log("normalize_text", f"Normalized text in '{col}'")

        # 7. Outlier handling
        outlier_cfg = cfg.get("outliers", {})
        strategy = outlier_cfg.get("strategy", "cap_iqr")
        for col in df.select_dtypes(include="number").columns:
            if col in outlier_cfg.get("skip_columns", []):
                continue
            before_nulls = df[col].isnull().sum()
            if strategy == "cap_iqr":
                df = cap_iqr(df, col, factor=outlier_cfg.get("iqr_factor", 1.5))
                self._log("outlier_cap_iqr", f"Capped IQR outliers in '{col}'")
            elif strategy == "remove_iqr":
                before = len(df)
                df = remove_iqr(df, col, factor=outlier_cfg.get("iqr_factor", 1.5))
                self._log("outlier_remove_iqr", f"Removed {before-len(df)} outlier rows for '{col}'", rows_affected=before-len(df))
            elif strategy == "cap_zscore":
                df = cap_zscore(df, col, threshold=outlier_cfg.get("zscore_threshold", 3.0))
                self._log("outlier_cap_zscore", f"Capped Z-score outliers in '{col}'")

        # 8. Low-variance columns
        if cfg.get("drop_low_variance", False):
            before_cols = list(df.columns)
            df = drop_low_variance(df, threshold=cfg.get("variance_threshold", 0.01))
            dropped = [c for c in before_cols if c not in df.columns]
            if dropped:
                self._log("low_variance", f"Dropped low-variance columns: {dropped}")

        final_nulls = int(df.isnull().sum().sum())
        self._log("summary", f"Cleaning complete. Rows: {original_rows}→{len(df)}, Nulls: {original_nulls}→{final_nulls}")
        return df

    def get_log(self) -> List[Dict]:
        return self.log

    def get_report(self) -> Dict[str, Any]:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "steps": self.log,
            "total_steps": len(self.log),
        }
