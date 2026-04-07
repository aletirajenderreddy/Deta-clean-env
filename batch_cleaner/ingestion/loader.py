"""Ingestion module — loads CSV, Excel, SQLite databases into pandas DataFrames."""
from __future__ import annotations
import io, sqlite3
from pathlib import Path
from typing import Union
import pandas as pd


SUPPORTED_EXT = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet", ".db", ".sqlite"}


def load_file(path: Union[str, Path, bytes], filename: str = "") -> pd.DataFrame:
    """Load a dataset from a file path, bytes buffer, or filename hint."""
    if isinstance(path, bytes):
        return _from_bytes(path, filename)
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported format: {ext}. Supported: {SUPPORTED_EXT}")
    return _read(p, ext)


def _from_bytes(data: bytes, filename: str) -> pd.DataFrame:
    ext = Path(filename).suffix.lower() if filename else ".csv"
    buf = io.BytesIO(data)
    return _read(buf, ext)


def _read(src, ext: str) -> pd.DataFrame:
    readers = {
        ".csv":  lambda s: pd.read_csv(s, low_memory=False),
        ".tsv":  lambda s: pd.read_csv(s, sep="\t", low_memory=False),
        ".xlsx": lambda s: pd.read_excel(s, engine="openpyxl"),
        ".xls":  lambda s: pd.read_excel(s),
        ".json": lambda s: pd.read_json(s),
        ".parquet": lambda s: pd.read_parquet(s),
        ".db":   lambda s: _load_sqlite(s),
        ".sqlite": lambda s: _load_sqlite(s),
    }
    reader = readers.get(ext)
    if not reader:
        raise ValueError(f"No reader for extension: {ext}")
    df = reader(src)
    # Normalise column names: strip whitespace
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _load_sqlite(path) -> pd.DataFrame:
    """Load the first table from an SQLite database."""
    conn = sqlite3.connect(str(path))
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    if tables.empty:
        raise ValueError("No tables found in SQLite database")
    table = tables.iloc[0]["name"]
    df = pd.read_sql(f"SELECT * FROM [{table}]", conn)
    conn.close()
    return df


def load_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Accept a pre-loaded DataFrame and normalise columns."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df
