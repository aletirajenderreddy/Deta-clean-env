"""Schema validation layer — validates DataFrames against configurable rules."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


class ValidationError:
    def __init__(self, row: Optional[int], column: str, rule: str, value: Any, message: str):
        self.row = row
        self.column = column
        self.rule = rule
        self.value = value
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        return {"row": self.row, "column": self.column, "rule": self.rule, "value": str(self.value), "message": self.message}


class SchemaValidator:
    """
    Validate a DataFrame against a schema dict.

    Schema format:
    {
      "columns": {
        "age": {"type": "numeric", "required": true, "min": 0, "max": 120},
        "email": {"type": "string", "required": true, "pattern": ".*@.*"},
        "status": {"type": "string", "allowed_values": ["active","inactive"]}
      }
    }
    """

    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema

    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[ValidationError]]:
        """
        Returns:
          valid_df: rows that passed all rules
          errors: list of ValidationError objects
        """
        errors: List[ValidationError] = []
        col_schemas = self.schema.get("columns", {})
        invalid_rows = set()

        # Required fields check
        for col, rules in col_schemas.items():
            if col not in df.columns:
                if rules.get("required", False):
                    errors.append(ValidationError(None, col, "missing_column", None, f"Required column '{col}' not found"))
                continue

            for idx, val in enumerate(df[col]):
                row_errors = self._check_value(idx, col, val, rules)
                errors.extend(row_errors)
                if row_errors:
                    invalid_rows.add(idx)

        valid_df = df.drop(index=list(invalid_rows)).reset_index(drop=True)
        return valid_df, errors

    def _check_value(self, row: int, col: str, val: Any, rules: Dict) -> List[ValidationError]:
        errs = []
        is_null = pd.isnull(val) if not isinstance(val, (list, dict)) else False

        # Required
        if rules.get("required", False) and is_null:
            errs.append(ValidationError(row, col, "required", val, f"Row {row}: '{col}' is required but null"))
            return errs  # don't run further checks on null

        if is_null:
            return errs

        # Type check
        expected_type = rules.get("type")
        if expected_type == "numeric":
            try:
                float(val)
            except (TypeError, ValueError):
                errs.append(ValidationError(row, col, "type", val, f"Row {row}: '{col}' expected numeric, got '{val}'"))

        # Range
        if "min" in rules:
            try:
                if float(val) < rules["min"]:
                    errs.append(ValidationError(row, col, "min", val, f"Row {row}: '{col}'={val} < min={rules['min']}"))
            except (TypeError, ValueError):
                pass

        if "max" in rules:
            try:
                if float(val) > rules["max"]:
                    errs.append(ValidationError(row, col, "max", val, f"Row {row}: '{col}'={val} > max={rules['max']}"))
            except (TypeError, ValueError):
                pass

        # Pattern
        if "pattern" in rules:
            import re
            if not re.match(rules["pattern"], str(val)):
                errs.append(ValidationError(row, col, "pattern", val, f"Row {row}: '{col}'='{val}' doesn't match pattern '{rules['pattern']}'"))

        # Allowed values
        if "allowed_values" in rules:
            if str(val) not in [str(v) for v in rules["allowed_values"]]:
                errs.append(ValidationError(row, col, "allowed_values", val, f"Row {row}: '{col}'='{val}' not in {rules['allowed_values']}"))

        return errs
