"""Cleaning rules — individual, composable cleaning functions."""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


# ── MISSING VALUES ────────────────────────────────────────────────────────────

def fill_mean(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].mean())
    return df


def fill_median(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    return df


def fill_mode(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    mode = df[col].mode()
    if not mode.empty:
        df[col] = df[col].fillna(mode[0])
    return df


def fill_constant(df: pd.DataFrame, col: str, value: Any = "unknown") -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].fillna(value)
    return df


def drop_missing_rows(df: pd.DataFrame, col: Optional[str] = None, threshold: float = 0.5) -> pd.DataFrame:
    """Drop rows where col is null, or rows missing more than threshold fraction of values."""
    df = df.copy()
    if col:
        df = df.dropna(subset=[col])
    else:
        df = df.dropna(thresh=int(len(df.columns) * (1 - threshold)))
    return df


def drop_missing_cols(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns with missing % above threshold."""
    df = df.copy()
    keep = [c for c in df.columns if df[c].isnull().mean() <= threshold]
    return df[keep]


# ── DUPLICATES ────────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = "first") -> pd.DataFrame:
    df = df.copy()
    return df.drop_duplicates(subset=subset, keep=keep)


# ── DATA TYPE CORRECTION ──────────────────────────────────────────────────────

def cast_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def cast_datetime(df: pd.DataFrame, col: str, fmt: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
    return df


def cast_boolean(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    true_vals = {"true", "1", "yes", "y", "on"}
    df[col] = df[col].astype(str).str.strip().str.lower().map(lambda x: True if x in true_vals else (False if x in {"false","0","no","n","off"} else None))
    return df


# ── TEXT NORMALISATION ────────────────────────────────────────────────────────

def normalize_text(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Lowercase + strip whitespace + collapse internal spaces."""
    df = df.copy()
    df[col] = df[col].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    return df


def strip_whitespace(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].astype(str).str.strip()
    return df


def fix_email(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Flag malformed emails."""
    df = df.copy()
    pattern = r"^[\w\.\+\-]+@[\w\-]+\.[a-z]{2,}$"
    mask = df[col].notna() & ~df[col].astype(str).str.match(pattern, case=False)
    df.loc[mask, col] = np.nan  # flag as missing for re-imputation
    return df


# ── OUTLIER HANDLING ──────────────────────────────────────────────────────────

def cap_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
    """Cap outliers at IQR bounds (Winsorize)."""
    df = df.copy()
    if not pd.api.types.is_numeric_dtype(df[col]):
        return df
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def remove_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
    """Remove rows containing IQR outliers."""
    df = df.copy()
    if not pd.api.types.is_numeric_dtype(df[col]):
        return df
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    return df[(df[col] >= lo) & (df[col] <= hi)]


def cap_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.DataFrame:
    """Cap Z-score outliers."""
    df = df.copy()
    if not pd.api.types.is_numeric_dtype(df[col]):
        return df
    mean = df[col].mean()
    std = df[col].std()
    if std == 0:
        return df
    lo = mean - threshold * std
    hi = mean + threshold * std
    df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# ── COLUMN REMOVAL ────────────────────────────────────────────────────────────

def drop_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    return df.drop(columns=[col], errors="ignore")


def drop_low_variance(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Drop numeric columns with near-zero variance."""
    df = df.copy()
    num = df.select_dtypes(include=[np.number])
    low_var = [c for c in num.columns if num[c].std() < threshold]
    return df.drop(columns=low_var)
