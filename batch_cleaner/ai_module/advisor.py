"""AI advisor — uses OpenAI API to suggest cleaning actions from dataset statistics."""
from __future__ import annotations
import json
import os
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


SYSTEM_PROMPT = """You are a senior data engineer specializing in data quality.
You receive statistical summaries of datasets (never raw data) and return structured JSON cleaning recommendations.
Always respond with ONLY valid JSON matching the schema provided. No markdown fences, no extra text."""

USER_TEMPLATE = """Analyze this dataset summary and recommend cleaning actions.

Dataset Summary:
{summary}

Respond with JSON only, in this exact schema:
{{
  "drop_columns": ["list of column names to drop (high null %, low info, etc.)"],
  "fill_missing": {{"col_name": "strategy"}} where strategy is one of: mean, median, mode, drop, constant_unknown,
  "outlier_strategy": "cap_iqr | remove_iqr | cap_zscore | none",
  "cast_types": {{"col_name": "numeric | datetime | boolean"}},
  "normalize_text": ["columns to lowercase+strip"],
  "anomalies": ["plain-english descriptions of detected anomalies"],
  "feature_importance_hints": ["which columns seem most informative"],
  "notes": "overall summary and any warnings"
}}"""


class AIAdvisor:
    """Queries an LLM for cleaning suggestions."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self._client = None

    def _get_client(self):
        if not _HAS_OPENAI:
            raise RuntimeError("openai package not installed. pip install openai")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        if not self._client:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def build_summary(self, profile: Dict[str, Any]) -> str:
        """Build a concise statistics summary safe to send to an LLM (no raw data)."""
        lines = [
            f"Rows: {profile['shape']['rows']}, Columns: {profile['shape']['columns']}",
            f"Total missing: {profile['total_missing']} ({profile['missing_pct']}%)",
            f"Duplicate rows: {profile['duplicate_rows']}",
            "",
            "Column statistics:",
        ]
        for col, info in profile.get("columns", {}).items():
            line = (
                f"  {col} [{info['dtype']}]: "
                f"null={info['null_pct']}%, "
                f"unique={info['unique_count']}"
            )
            if "mean" in info:
                line += f", mean={info['mean']}, std={info['std']}, min={info['min']}, max={info['max']}"
                if info.get("outlier_count_iqr"):
                    line += f", IQR_outliers={info['outlier_count_iqr']}"
            if "top_values" in info:
                top = list(info["top_values"].keys())[:3]
                line += f", top_vals={top}"
            lines.append(line)
        return "\n".join(lines)

    def get_suggestions(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Query the LLM and return structured suggestions dict."""
        summary = self.build_summary(profile)
        prompt = USER_TEMPLATE.format(summary=summary)
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
            )
            content = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {e}", "raw": content if "content" in dir() else ""}
        except Exception as e:
            return {"error": str(e), "drop_columns": [], "fill_missing": {}, "outlier_strategy": "cap_iqr", "notes": "AI unavailable — using defaults"}

    def apply_suggestions(self, suggestions: Dict, config: Dict) -> Dict:
        """Merge AI suggestions into an existing cleaning config dict."""
        merged = dict(config)
        if suggestions.get("drop_columns"):
            merged["drop_columns"] = list(set(merged.get("drop_columns", []) + suggestions["drop_columns"]))
        if suggestions.get("fill_missing"):
            merged.setdefault("fill_missing", {}).update(suggestions["fill_missing"])
        if suggestions.get("outlier_strategy") and suggestions["outlier_strategy"] != "none":
            merged.setdefault("outliers", {})["strategy"] = suggestions["outlier_strategy"]
        if suggestions.get("cast_types"):
            merged.setdefault("cast_types", {}).update(suggestions["cast_types"])
        if suggestions.get("normalize_text"):
            existing = set(merged.get("normalize_text", []))
            existing.update(suggestions["normalize_text"])
            merged["normalize_text"] = list(existing)
        return merged
