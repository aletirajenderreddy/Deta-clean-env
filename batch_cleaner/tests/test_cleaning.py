"""Unit tests for batch_cleaner modules."""
from __future__ import annotations
import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Make sure we can import from the project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from batch_cleaner.ingestion.loader import load_file, load_dataframe
from batch_cleaner.profiling.profiler import profile
from batch_cleaner.cleaning.rules import (
    fill_median, fill_mode, fill_constant, remove_duplicates,
    cap_iqr, normalize_text, cast_numeric, drop_missing_cols
)
from batch_cleaner.cleaning.engine import CleaningEngine
from batch_cleaner.validation.validator import SchemaValidator


SAMPLE_CSV = Path(__file__).parent / "sample_data.csv"


# ── INGESTION ──────────────────────────────────────────────────────────────────

def test_load_csv():
    df = load_file(str(SAMPLE_CSV))
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "name" in df.columns

def test_load_bytes():
    data = SAMPLE_CSV.read_bytes()
    df = load_file(data, filename="sample_data.csv")
    assert len(df) > 0

def test_load_dataframe():
    df_in = pd.DataFrame({"a ": [1, 2], "b": [3, 4]})
    df_out = load_dataframe(df_in)
    assert "a" in df_out.columns  # stripped


# ── PROFILING ──────────────────────────────────────────────────────────────────

def test_profile_structure():
    df = load_file(str(SAMPLE_CSV))
    report = profile(df)
    assert "shape" in report
    assert "columns" in report
    assert "total_missing" in report
    assert report["shape"]["rows"] == len(df)

def test_profile_numeric_stats():
    df = pd.DataFrame({"age": [20, 30, 40, None, 200], "name": ["a","b","c","d","e"]})
    report = profile(df)
    age_info = report["columns"]["age"]
    assert "mean" in age_info
    assert "outlier_count_iqr" in age_info


# ── CLEANING RULES ─────────────────────────────────────────────────────────────

def test_fill_median():
    df = pd.DataFrame({"salary": [50000.0, None, 80000.0, None, 70000.0]})
    result = fill_median(df, "salary")
    assert result["salary"].isnull().sum() == 0
    assert result["salary"].iloc[1] == 70000.0

def test_fill_mode():
    df = pd.DataFrame({"dept": ["Eng", "Eng", None, "HR", None]})
    result = fill_mode(df, "dept")
    assert result["dept"].isnull().sum() == 0
    assert result["dept"].iloc[2] == "Eng"

def test_fill_constant():
    df = pd.DataFrame({"status": [None, "active", None]})
    result = fill_constant(df, "status", value="unknown")
    assert result["status"].iloc[0] == "unknown"

def test_remove_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    result = remove_duplicates(df)
    assert len(result) == 2

def test_cap_iqr():
    df = pd.DataFrame({"age": [25, 30, 28, 999, 27, 32]})
    result = cap_iqr(df, "age")
    assert result["age"].max() < 999

def test_normalize_text():
    df = pd.DataFrame({"name": ["  Alice  ", "BOB", "carol  "]})
    result = normalize_text(df, "name")
    assert result["name"].iloc[0] == "alice"
    assert result["name"].iloc[1] == "bob"

def test_cast_numeric():
    df = pd.DataFrame({"age": ["25", "30", "abc", None]})
    result = cast_numeric(df, "age")
    assert pd.api.types.is_float_dtype(result["age"])
    assert pd.isnull(result["age"].iloc[2])

def test_drop_missing_cols():
    df = pd.DataFrame({"a": [1, None, None, None], "b": [1, 2, 3, 4]})
    result = drop_missing_cols(df, threshold=0.5)
    assert "a" not in result.columns
    assert "b" in result.columns


# ── CLEANING ENGINE ──────────────────────────────────────────────────────────

def test_engine_basic():
    df = load_file(str(SAMPLE_CSV))
    engine = CleaningEngine()
    cleaned = engine.clean(df)
    assert isinstance(cleaned, pd.DataFrame)
    assert len(cleaned) > 0
    # Duplicates removed
    assert cleaned.duplicated().sum() == 0

def test_engine_log():
    df = pd.DataFrame({"age": [25, None, 30], "name": ["Alice", None, "Bob"]})
    cfg = {"default_fill_strategy": "mode", "duplicates": {"remove": True}}
    engine = CleaningEngine(config=cfg)
    engine.clean(df)
    log = engine.get_log()
    assert len(log) > 0
    assert all("step" in e for e in log)

def test_engine_config_override():
    df = pd.DataFrame({"salary": [50000.0, None, 80000.0]})
    cfg = {"default_fill_strategy": "mean", "duplicates": {"remove": False}}
    engine = CleaningEngine(config=cfg)
    result = engine.clean(df)
    assert result["salary"].isnull().sum() == 0


# ── VALIDATION ────────────────────────────────────────────────────────────────

def test_validation_required():
    schema = {"columns": {"email": {"required": True}}}
    df = pd.DataFrame({"email": ["a@b.com", None, "c@d.com"]})
    validator = SchemaValidator(schema)
    valid_df, errors = validator.validate(df)
    assert len(errors) == 1
    assert len(valid_df) == 2

def test_validation_range():
    schema = {"columns": {"age": {"type": "numeric", "min": 0, "max": 120}}}
    df = pd.DataFrame({"age": [25, 999, 30]})
    validator = SchemaValidator(schema)
    valid_df, errors = validator.validate(df)
    assert any(e.rule == "max" for e in errors)

def test_validation_allowed_values():
    schema = {"columns": {"status": {"allowed_values": ["active", "inactive"]}}}
    df = pd.DataFrame({"status": ["active", "unknown", "inactive"]})
    validator = SchemaValidator(schema)
    valid_df, errors = validator.validate(df)
    assert len(errors) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
