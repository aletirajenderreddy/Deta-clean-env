================================================================================
 DataCleanEnv — Complete Project Documentation
 Version 1.0.0 | Built with Python 3.11, FastAPI, React-free Vanilla JS
================================================================================

TABLE OF CONTENTS
─────────────────
 1. Project Overview
 2. Architecture Diagram
 3. Module Descriptions
 4. Dashboard UI Guide
 5. Batch Cleaning System API
 6. AI Integration Details
 7. Configuration Reference
 8. Running the Project
 9. Docker Setup
10. Testing
11. API Endpoint Reference
12. File Structure
13. Known Limitations & Future Work

================================================================================
1. PROJECT OVERVIEW
================================================================================

DataCleanEnv is a dual-purpose platform:

  A) RL ENVIRONMENT DASHBOARD
     A reinforcement-learning environment where an AI agent learns to clean
     messy tabular datasets. Hosted as a FastAPI server with a dark industrial
     "Mission Control" web UI. Supports:
       - 3 difficulty levels (Easy / Medium / Hard)
       - Real-time reward tracking
       - Cell-level issue highlighting (missing, format, type, outlier)
       - Auto-cleaning agent (rule-based, client-side)
       - Custom CSV upload & local simulation

  B) BATCH DATA CLEANING SYSTEM WITH AI INTEGRATION
     A production-ready REST API for batch data cleaning:
       - Upload CSV / Excel / JSON datasets
       - Auto-profile: missing%, types, outliers, distribution
       - AI suggestions via OpenAI GPT
       - Rule-based cleaning engine (configurable via YAML)
       - Schema validation layer
       - Download cleaned CSV / Excel
       - Structured logs (JSON + TXT)


================================================================================
2. ARCHITECTURE DIAGRAM
================================================================================

  Browser
    │
    ▼
  GET / or /web  ──────────────►  dashboard.html (Mission Control UI)
    │                                   │
    │                          JS: fetch /reset, /step
    │                                   │
    ▼                                   ▼
  FastAPI Server (server/app.py)  ──────────────────────────────────────
    │                                   │
    ├── /reset, /step ────────► server/environment.py (RL env)
    │                               └── server/dataset_generator.py
    │                               └── server/graders.py
    │                               └── server/reward.py
    │
    └── /batch/* ─────────────► batch_cleaner/ (Batch System)
          │
          ├── /batch/upload    → ingestion/loader.py
          ├── /batch/profile   → profiling/profiler.py
          ├── /batch/ai-suggest→ ai_module/advisor.py (OpenAI API)
          ├── /batch/clean     → cleaning/engine.py + rules.py
          ├── /batch/download  → reports/{job_id}_cleaned.csv
          └── /batch/report    → reports/{job_id}_cleaning_report.json


================================================================================
3. MODULE DESCRIPTIONS
================================================================================

── server/ (RL Environment) ──────────────────────────────────────────────────

  app.py
    FastAPI application entry point. Mounts all routes including batch_cleaner
    router. Serves dashboard.html at /, /web, /dashboard.

  environment.py
    Core RL environment. Implements reset() and step() methods.
    Tracks episode state: current dataset, issues, rewards, steps.

  dataset_generator.py
    Generates synthetic messy datasets with configurable issue density.
    Used by the RL environment for Easy/Medium/Hard episodes.

  graders.py
    Task-specific graders that evaluate how clean a dataset is.
    Returns a quality score 0.0–1.0 used for reward calculation.

  reward.py
    Maps cleaning actions → reward signals (+0.1 to +0.5 for correct fixes,
    -0.1 for wrong/unnecessary actions).

── batch_cleaner/ (Batch System) ────────────────────────────────────────────

  ingestion/loader.py
    Unified file loader. Accepts:
      - File path (str/Path)
      - Bytes buffer + filename hint
      - Pre-loaded pandas DataFrame
    Supports: .csv, .tsv, .xlsx, .xls, .json, .parquet, .db/.sqlite

  profiling/profiler.py
    Generates a detailed statistical profile:
      - Per-column: null%, unique count, mean/median/std/min/max/q1/q3
      - IQR outlier count, value distribution (histogram buckets)
      - Top 5 value frequencies for categorical columns
    Exports: JSON + downloadable HTML report

  cleaning/rules.py
    20+ individual, composable cleaning functions:
      fill_mean, fill_median, fill_mode, fill_constant
      remove_duplicates
      cast_numeric, cast_datetime, cast_boolean
      normalize_text, strip_whitespace, fix_email
      cap_iqr, remove_iqr, cap_zscore
      drop_column, drop_low_variance

  cleaning/engine.py
    Orchestrator that applies rules in pipeline order:
      1. Drop high-null columns
      2. Drop explicitly listed columns
      3. Remove duplicates
      4. Fill missing values (per-column strategy from config)
      5. Cast types
      6. Normalize text
      7. Handle outliers (cap/remove)
      8. Drop low-variance columns
    Generates detailed log for every step applied.

  config/cleaning_rules.yaml
    YAML config controlling all cleaning rules.
    Can be overridden per API call via JSON payload.

  validation/validator.py
    Schema-based row validator:
      - Required fields check
      - Type checks (numeric, datetime)
      - Range checks (min/max)
      - Regex pattern matching
      - Allowed values list
    Invalid rows are separated from valid output with error detail.

  ai_module/advisor.py
    OpenAI GPT integration:
      - Converts dataset profile to a safe, statistics-only prompt
      - Returns structured JSON recommendations
      - Merges suggestions into cleaning config automatically
    Output schema:
      {
        "drop_columns": [],
        "fill_missing": {"age": "median"},
        "outlier_strategy": "cap_iqr",
        "cast_types": {},
        "normalize_text": [],
        "anomalies": [],
        "feature_importance_hints": [],
        "notes": ""
      }

  utils/logger.py
    Structured logger for each cleaning job:
      - timestamped entries with level, event, detail, rows_affected
      - saves JSON log + human-readable TXT log to reports/

  api/routes.py
    FastAPI router mounted at /batch. All endpoints described in Section 5.

  tests/test_cleaning.py
    19 unit tests covering ingestion, profiling, all rule functions,
    the cleaning engine, and the validation layer.

  tests/sample_data.csv
    20-row sample with intentional issues:
      - Missing values (name, age, email, phone, salary)
      - Duplicate rows (rows 4=1, 11=3)
      - Wrong format (date as MM/DD/YYYY, phone no dashes, email no @)
      - Wrong type (age="thirty")
      - Outlier (age=999, salary=-5000)
      - Inconsistent text whitespace


================================================================================
4. DASHBOARD UI GUIDE
================================================================================

Access at: http://127.0.0.1:8000/web

  HOME PAGE
  ──────────
  Clean landing with 3 action cards:

  [📂 UPLOAD CSV]
    Upload any .csv/.tsv/.xlsx file.
    Supports up to 5000 rows. Shows 5-row preview before loading.
    Detects issues automatically using generic heuristics
    (works with ANY column names, not just the built-in schema).

  [🗄️ BUILT-IN DATASET]
    Select Easy / Medium / Hard difficulty.
    Click ▶ START EPISODE to fetch from RL environment server.

  [⬇️ DOWNLOAD]  (enabled after data is loaded)
    Click to open format-choice modal (CSV or Excel).

  DASHBOARD
  ──────────
  Switches to full 5-zone layout after loading data:

  ┌──────────────────────────────────────────────────────┐
  │ HEADER: Logo (clickable→home) | Status | Step | Timer │
  ├─────────────┬────────────────────────────────────────┤
  │ LEFT PANEL  │  TABLE ZONE                            │
  │ - Level btns│  ⬡ LIVE VIEW / ⟺ DIFF VIEW tabs       │
  │ - START btn │  ⬆ UPLOAD CSV | ▣ COMPARE | ⬇ DOWNLOAD│
  │ - PAUSE btn │  Dataset table with issue highlighting  │
  │ - RESET btn │  (drag column edges to resize width)    │
  │ - AUTO CLEAN│                                        │
  │ - Speed sl. │  COMPARISON CHART (▣ COMPARE toggle)   │
  │ - Live Stats│  Bar chart: Original vs Remaining vs Fixed│
  │ - IssuesBars│                                        │
  │ - Score Ring│  ACTION LOG                            │
  ├─────────────┴────────────────────────────────────────┤
  │ REWARD CURVE (Chart.js line + bar chart)              │
  └──────────────────────────────────────────────────────┘

  CELL COLORS:
    Red border  = Missing value (——)
    Amber border = Wrong format (email without @, bad date)
    Orange border= Wrong type (string where number expected)
    Purple border= Outlier (age>120 or salary<0)
    Duplicate rows shown with red row background + DUP badge

  DIFF VIEW:
    Side-by-side: Original Messy vs Current State.
    Changed cells highlighted green (improved) or red (different).
    Horizontal drag: resize left/right pane with the │ divider.
    Vertical drag: resize pane height with the — divider.

  AUTO CLEAN:
    Clicks "⚡ AUTO CLEAN" to start the rule-based agent.
    Agent runs every N milliseconds (controlled by Speed slider).
    Agent priority: fix missing → remove duplicates → fix format → fix type → fix outlier.
    Works in both Built-in mode (server API) and CSV upload mode (local simulation).


================================================================================
5. BATCH CLEANING SYSTEM API
================================================================================

Base URL: http://127.0.0.1:8000/batch

Interactive docs: http://127.0.0.1:8000/docs

  WORKFLOW:
    1. POST /batch/upload          → get job_id
    2. POST /batch/profile/{id}    → inspect data quality
    3. POST /batch/ai-suggest/{id} → get AI recommendations
    4. POST /batch/clean/{id}      → run cleaning pipeline
    5. GET  /batch/download/{id}   → download cleaned CSV
    5b. GET /batch/download-excel/{id} → download XLSX
    6. GET  /batch/report/{id}     → get cleaning report JSON
    7. GET  /batch/log/{id}        → get cleaning log TXT

  EXAMPLE (curl):

    # 1. Upload
    curl -X POST http://localhost:8000/batch/upload \
         -F "file=@sample_data.csv"
    # → {"job_id":"abc12345","rows":20,"columns":[...],"status":"uploaded"}

    # 2. Profile
    curl -X POST "http://localhost:8000/batch/profile/abc12345?format=json"

    # 3. AI Suggestions (needs OPENAI_API_KEY)
    curl -X POST http://localhost:8000/batch/ai-suggest/abc12345

    # 4. Clean
    curl -X POST http://localhost:8000/batch/clean/abc12345

    # 5. Download
    curl -O "http://localhost:8000/batch/download/abc12345"

    # 6. View HTML profile in browser
    http://localhost:8000/batch/profile-html/abc12345

    # All jobs
    curl http://localhost:8000/batch/jobs


================================================================================
6. AI INTEGRATION DETAILS
================================================================================

  Model: gpt-4o-mini (configurable)
  Safety: ONLY dataset statistics are sent. No raw data values.

  What is sent (example):
    Rows: 20, Columns: 9
    Total missing: 8 (4.44%)
    Duplicate rows: 2
    Column statistics:
      age [object]: null=5%, unique=17, top_vals=['28','35','31']
      salary [float64]: null=5%, unique=18, mean=76000, std=12000, IQR_outliers=1

  What is returned:
    {
      "drop_columns": [],
      "fill_missing": {"age": "median", "salary": "median"},
      "outlier_strategy": "cap_iqr",
      "cast_types": {"age": "numeric", "id": "numeric"},
      "normalize_text": ["name", "department"],
      "anomalies": ["salary has negative values", "age has string values"],
      "feature_importance_hints": ["department", "salary"],
      "notes": "Dataset has moderate quality issues..."
    }

  AI Fallback: If OPENAI_API_KEY is not set or API fails,
  the system returns default recommendations and continues.


================================================================================
7. CONFIGURATION REFERENCE
================================================================================

  File: batch_cleaner/config/cleaning_rules.yaml

  duplicates:
    remove: true          # Remove duplicate rows
    keep: first           # Which duplicate to keep: first|last

  fill_missing:           # Per-column strategies
    age: median           # mean | median | mode | drop | constant
    email: "unknown@example.com"

  default_fill_strategy: mode   # Used for columns not listed above

  drop_columns_missing_above: 0.9  # Drop cols with >90% missing

  drop_columns: []        # Column names to always drop

  cast_types:            # Force type conversions
    age: numeric
    salary: numeric

  normalize_text:        # Columns to lowercase+strip
    - name
    - department

  outliers:
    strategy: cap_iqr    # cap_iqr|remove_iqr|cap_zscore|none
    iqr_factor: 1.5
    zscore_threshold: 3.0
    skip_columns: [id]

  drop_low_variance: false
  variance_threshold: 0.01


================================================================================
8. RUNNING THE PROJECT
================================================================================

  PREREQUISITES:
    - Python 3.11+
    - pip packages (auto-installed):
      fastapi, uvicorn, pandas, numpy, openpyxl, scipy, pyyaml,
      aiofiles, python-multipart, pyarrow, openai, faker

  INSTALL:
    cd datacleanenv
    pip install -r requirements.txt

  SET API KEYS (create .env file):
    OPENAI_API_KEY=sk-...
    HUGGING_FACE_HUB_TOKEN=hf_...

  START SERVER:
    python3.11 -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --reload

  ACCESS:
    Dashboard:    http://127.0.0.1:8000/web
    API Docs:     http://127.0.0.1:8000/docs
    Health:       http://127.0.0.1:8000/health
    Batch APIs:   http://127.0.0.1:8000/batch/*


================================================================================
9. DOCKER SETUP
================================================================================

  BUILD & RUN (one command):
    docker-compose up --build

  ENVIRONMENT VARIABLES (in docker-compose.yml or .env):
    OPENAI_API_KEY=sk-...
    HUGGING_FACE_HUB_TOKEN=hf_...

  PORTS:
    8000 → FastAPI server (dashboard + batch API)

  VOLUMES:
    ./batch_cleaner/reports → /app/batch_cleaner/reports
    (cleaned files and logs persist on host)

  MANUAL DOCKER:
    docker build -t datacleanenv .
    docker run -p 8000:8000 -e OPENAI_API_KEY=... datacleanenv


================================================================================
10. TESTING
================================================================================

  RUN ALL TESTS:
    python3.11 -m pytest batch_cleaner/tests/test_cleaning.py -v

  TEST COVERAGE (19 tests):
    ✓ test_load_csv             - CSV file loading
    ✓ test_load_bytes           - Bytes buffer loading
    ✓ test_load_dataframe       - DataFrame column normalization
    ✓ test_profile_structure    - Profile report structure
    ✓ test_profile_numeric_stats- Numeric statistics in profile
    ✓ test_fill_median          - Median imputation
    ✓ test_fill_mode            - Mode imputation
    ✓ test_fill_constant        - Constant fill
    ✓ test_remove_duplicates    - Duplicate removal
    ✓ test_cap_iqr              - IQR outlier capping
    ✓ test_normalize_text       - Text normalization
    ✓ test_cast_numeric         - Type casting
    ✓ test_drop_missing_cols    - High-null column dropping
    ✓ test_engine_basic         - Full pipeline on sample data
    ✓ test_engine_log           - Log generation
    ✓ test_engine_config_override- Config override
    ✓ test_validation_required  - Required field validation
    ✓ test_validation_range     - Min/max range validation
    ✓ test_validation_allowed_values - Allowed values check


================================================================================
11. API ENDPOINT REFERENCE
================================================================================

  RL ENVIRONMENT
  ──────────────
  GET  /              → Dashboard UI
  GET  /web           → Dashboard UI (alias)
  GET  /health        → Health check {"status":"healthy"}
  POST /reset         → Start new episode {"task_level":"easy|medium|hard"}
  POST /step          → Take cleaning action {"action":{...}}
  GET  /state         → Current episode state

  BATCH CLEANING
  ──────────────
  POST /batch/upload                  → Upload file, get job_id
  POST /batch/profile/{job_id}        → Profile dataset (JSON or HTML)
  POST /batch/ai-suggest/{job_id}     → Get AI cleaning suggestions
  POST /batch/clean/{job_id}          → Run cleaning pipeline
  GET  /batch/download/{job_id}       → Download cleaned CSV
  GET  /batch/download-excel/{job_id} → Download cleaned Excel
  GET  /batch/report/{job_id}         → Download cleaning report JSON
  GET  /batch/log/{job_id}            → Download cleaning log TXT
  GET  /batch/profile-html/{job_id}   → View HTML profiling report
  GET  /batch/jobs                    → List all jobs
  GET  /batch/jobs/{job_id}           → Get specific job status

  INTERACTIVE API DOCS:
  http://127.0.0.1:8000/docs


================================================================================
12. FILE STRUCTURE
================================================================================

datacleanenv/
├── server/
│   ├── app.py                    ← FastAPI app (RL + Batch routes)
│   ├── environment.py            ← RL environment logic
│   ├── dataset_generator.py      ← Messy dataset generator
│   ├── graders.py                ← Episode scoring
│   ├── reward.py                 ← Reward computation
│   └── templates/
│       └── dashboard.html        ← Mission Control UI (single file)
│
├── batch_cleaner/
│   ├── ingestion/
│   │   └── loader.py             ← CSV/Excel/SQLite loader
│   ├── profiling/
│   │   └── profiler.py           ← Statistical profiling + HTML report
│   ├── cleaning/
│   │   ├── engine.py             ← Pipeline orchestrator
│   │   └── rules.py              ← 20+ individual cleaning functions
│   ├── validation/
│   │   └── validator.py          ← Schema validation
│   ├── ai_module/
│   │   └── advisor.py            ← OpenAI GPT integration
│   ├── utils/
│   │   └── logger.py             ← Structured logging (JSON + TXT)
│   ├── api/
│   │   └── routes.py             ← FastAPI /batch/* router
│   ├── config/
│   │   └── cleaning_rules.yaml   ← Default cleaning config
│   ├── tests/
│   │   ├── sample_data.csv       ← 20-row messy test dataset
│   │   └── test_cleaning.py      ← 19 unit tests
│   └── reports/                  ← Output: cleaned files, logs, reports
│
├── models.py                     ← Pydantic models (CleaningAction, etc.)
├── requirements.txt              ← Python dependencies
├── Dockerfile                    ← Docker image definition
├── docker-compose.yml            ← One-command Docker startup
├── openenv.yaml                  ← OpenEnv deployment config
├── demo.py                       ← Demo script for RL environment
├── baseline.py                   ← Baseline agent implementation
├── client.py                     ← Python client for the API
├── pyproject.toml                ← Project metadata
└── PROJECT_DOCUMENTATION.txt    ← This file


================================================================================
13. KNOWN LIMITATIONS & FUTURE WORK
================================================================================

  Current Limitations:
  • In-memory job store (jobs lost on server restart → use Redis for production)
  • AI suggestions require OpenAI API key (costs apply — uses gpt-4o-mini)
  • No authentication/authorization on batch endpoints
  • Dashboard RL environment uses server-side state (single user)
  • CSV local-mode cleaning is heuristic-only (no server validation)

  Planned Improvements:
  • Data drift detection (compare profile before vs after)
  • PostgreSQL/SQLite job persistence
  • WebSocket real-time progress for large batch jobs
  • Support for database connections (PostgreSQL, MySQL)
  • Visualization of missing data (heatmap)
  • Scheduled batch cleaning (cron-based)
  • Multi-file batch processing
  • User-defined cleaning rules via UI
  • Export profiling report to PDF

================================================================================
 END OF DOCUMENTATION
 DataCleanEnv v1.0.0 — Built with ❤️ using FastAPI + Vanilla JS
================================================================================
