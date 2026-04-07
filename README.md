# DataCleanEnv — AI-Powered Data Cleaning Platform

> **RL Environment + Batch Data Cleaning System + Mission Control Dashboard**

A production-ready dual platform:
1. **RL Environment** — Train AI agents to clean messy tabular datasets via reinforcement learning
2. **Batch Cleaning System** — Upload any CSV/Excel, auto-profile, get AI suggestions, and download cleaned data
3. **Mission Control UI** — A stunning dark terminal dashboard to visualize and control the full pipeline

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python **3.11+**
- `pip`
- (Optional) Docker for containerized run

---

### Step 1 — Clone & Enter Project

```bash
cd datacleanenv
```

---

### Step 2 — Create Virtual Environment

```bash
# Windows
python3.11 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4 — Configure API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...          # Required for AI cleaning suggestions
HUGGING_FACE_HUB_TOKEN=hf_... # Required for openenv push / HF Spaces
```

> **Note:** The platform runs fine without API keys — AI suggestions will return a fallback response, and all other features work fully.

---

### Step 5 — Start the Server (Backend + Frontend Together)

This project has **no separate frontend server** — the dashboard is served directly by FastAPI as a single HTML file.

```bash
# Windows
python3.11 -m uvicorn server.app:app --host 127.0.0.1 --port 7860 --reload

# macOS / Linux
uvicorn server.app:app --host 127.0.0.1 --port 7860 --reload
```

**That's it — one command starts everything.**

---

### Step 6 — Open in Browser

| URL | What it opens |
|-----|--------------|
| `http://127.0.0.1:7860/` | Mission Control Dashboard |
| `http://127.0.0.1:7860/web` | Dashboard (alias) |
| `http://127.0.0.1:7860/docs` | Interactive API docs (Swagger UI) |
| `http://127.0.0.1:7860/health` | Health check |
| `http://127.0.0.1:7860/batch/jobs` | List batch cleaning jobs |

---

## 🐳 Docker (One-Command Run)

```bash
# Build and start
docker-compose up --build

# OR build and run manually
docker build -t datacleanenv .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=sk-... \
  -e HUGGING_FACE_HUB_TOKEN=hf_... \
  datacleanenv
```

Then open: `http://localhost:7860/`

---

## 📁 Project Structure

```
datacleanenv/
│
├── server/                         ← Backend (FastAPI)
│   ├── app.py                      ← Main app: RL routes + batch routes + dashboard
│   ├── environment.py              ← RL environment logic
│   ├── dataset_generator.py        ← Messy dataset generator (Easy/Medium/Hard)
│   ├── graders.py                  ← Episode quality scoring
│   ├── reward.py                   ← Reward calculation
│   └── templates/
│       └── dashboard.html          ← Mission Control UI (standalone HTML — no build step)
│
├── batch_cleaner/                  ← Batch Data Cleaning System
│   ├── ingestion/loader.py         ← CSV · Excel · JSON · SQLite loader
│   ├── profiling/profiler.py       ← Statistical profiling + HTML report
│   ├── cleaning/
│   │   ├── engine.py               ← Cleaning pipeline orchestrator
│   │   └── rules.py                ← 20+ individual cleaning functions
│   ├── validation/validator.py     ← Schema validation (types, ranges, patterns)
│   ├── ai_module/advisor.py        ← OpenAI GPT integration (metadata-only prompts)
│   ├── utils/logger.py             ← Structured JSON + TXT logging
│   ├── api/routes.py               ← FastAPI /batch/* router (9 endpoints)
│   ├── config/cleaning_rules.yaml  ← Default cleaning configuration
│   ├── tests/
│   │   ├── sample_data.csv         ← 20-row messy test dataset
│   │   └── test_cleaning.py        ← 19 unit tests (all passing)
│   └── reports/                    ← Auto-created: cleaned files, logs, reports
│
├── models.py                       ← Pydantic models (CleaningAction, etc.)
├── requirements.txt                ← All Python dependencies
├── Dockerfile                      ← Production Docker image
├── docker-compose.yml              ← One-command Docker startup
├── demo.py                         ← Demo: run RL agent from Python
├── baseline.py                     ← GPT baseline agent
├── client.py                       ← Python API client
├── openenv.yaml                    ← OpenEnv / HuggingFace Spaces config
└── PROJECT_DOCUMENTATION.md       ← Full technical documentation
```

---

## 🖥️ Dashboard — How to Use

### Home Page
Three action cards greet you:

| Card | Action |
|------|--------|
| 📂 **UPLOAD CSV** | Upload your own messy CSV/Excel file — auto-detects issues |
| 🗄️ **BUILT-IN DATASET** | Choose Easy/Medium/Hard → click ▶ START EPISODE |
| ⬇️ **DOWNLOAD** | Export cleaned data as CSV or Excel (enabled after loading) |

### Main Dashboard Zones

```
┌─────────────────────────────────────────────────────────────────┐
│  HEADER: Logo (→ Home) │ Connected ● │ EP: xxxx │ Step X/30   │
├────────────┬────────────────────────────────────────────────────┤
│ LEFT PANEL │  ⬡ LIVE VIEW  │  ⟺ DIFF VIEW                     │
│            │  ⬆ UPLOAD  │  ▣ COMPARE  │  ⬇ DOWNLOAD           │
│ EASY/MED/  │                                                     │
│ HARD btns  │  Dataset Table (colored issue cells)               │
│            │  ← drag column edges to resize →                   │
│ ▶ START    │                                                     │
│ ⏸ PAUSE   │  ▣ COMPARE: Before vs After grouped bar chart      │
│ ⟳ RESET   │                                                     │
│ ⚡ AUTO    │  Agent Action Log                                   │
│            ├────────────────────────────────────────────────────┤
│ Live Stats │  REWARD CURVE (Chart.js)                           │
│ Issue Bars │                                                     │
│ Score Ring │                                                     │
└────────────┴────────────────────────────────────────────────────┘
```

### Cell Colors
| Color | Meaning |
|-------|---------|
| 🔴 Red border | Missing value |
| 🟡 Amber border | Wrong format (bad email, date) |
| 🟠 Orange border | Wrong type (string instead of number) |
| 🟣 Purple border | Outlier (e.g. age=999) |
| Row red bg + `DUP` badge | Duplicate row |

---

## 📦 Batch Data Cleaning API

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/batch/upload` | Upload CSV / Excel / JSON file |
| `POST` | `/batch/profile/{job_id}` | Get statistical profiling report |
| `POST` | `/batch/ai-suggest/{job_id}` | Get AI cleaning suggestions (OpenAI) |
| `POST` | `/batch/clean/{job_id}` | Run cleaning pipeline |
| `GET`  | `/batch/download/{job_id}` | Download cleaned CSV |
| `GET`  | `/batch/download-excel/{job_id}` | Download cleaned Excel |
| `GET`  | `/batch/report/{job_id}` | Download cleaning report JSON |
| `GET`  | `/batch/log/{job_id}` | Download cleaning log TXT |
| `GET`  | `/batch/profile-html/{job_id}` | View HTML profiling report in browser |
| `GET`  | `/batch/jobs` | List all cleaning jobs |

### Example Workflow (curl)

```bash
# 1. Upload your file
curl -X POST http://localhost:7860/batch/upload \
     -F "file=@my_data.csv"
# Returns: {"job_id": "abc12345", "rows": 500, ...}

# 2. Profile it
curl -X POST "http://localhost:7860/batch/profile/abc12345?format=json"

# 3. Get AI suggestions (needs OPENAI_API_KEY)
curl -X POST http://localhost:7860/batch/ai-suggest/abc12345

# 4. Run cleaning
curl -X POST http://localhost:7860/batch/clean/abc12345

# 5. Download cleaned CSV
curl -O http://localhost:7860/batch/download/abc12345

# 6. View HTML profile report in browser
open http://localhost:7860/batch/profile-html/abc12345
```

### Example Workflow (Python)

```python
import requests

BASE = "http://127.0.0.1:7860"

# Upload
with open("my_data.csv", "rb") as f:
    r = requests.post(f"{BASE}/batch/upload", files={"file": f})
job_id = r.json()["job_id"]

# Profile
profile = requests.post(f"{BASE}/batch/profile/{job_id}").json()
print(f"Missing values: {profile['total_missing']}")

# Clean
result = requests.post(f"{BASE}/batch/clean/{job_id}").json()
print(f"Cleaned {result['cleaned_rows']} rows, removed {result['rows_removed']}")

# Download
csv = requests.get(f"{BASE}/batch/download/{job_id}")
open("cleaned.csv", "wb").write(csv.content)
```

---

## 🤖 AI Integration

The AI module sends **only dataset statistics** to the LLM — never raw data values.

**What gets sent:**
```
Rows: 500, Columns: 12
Total missing: 45 (7.5%)
Column statistics:
  age [float64]: null=10%, mean=34.2, std=12.1, IQR_outliers=3
  email [object]: null=5%, unique=480, top_vals=['unknown']
```

**What AI returns (structured JSON):**
```json
{
  "drop_columns": ["irrelevant_col"],
  "fill_missing": {"age": "median", "salary": "mean"},
  "outlier_strategy": "cap_iqr",
  "cast_types": {"age": "numeric"},
  "normalize_text": ["name", "department"],
  "anomalies": ["salary has 3 negative values"],
  "notes": "Dataset is moderately clean..."
}
```

---

## ⚙️ Cleaning Configuration

Edit `batch_cleaner/config/cleaning_rules.yaml` to customize behavior:

```yaml
# Fill missing values per column
fill_missing:
  age: median
  salary: mean
  department: mode
  email: "unknown@example.com"

# Default strategy for unlisted columns
default_fill_strategy: mode

# Drop high-null columns (threshold: 0.0 – 1.0)
drop_columns_missing_above: 0.9

# Outlier handling
outliers:
  strategy: cap_iqr   # cap_iqr | remove_iqr | cap_zscore | none
  iqr_factor: 1.5

# Text normalization (lowercase + strip spaces)
normalize_text:
  - name
  - department
```

---

## 🧪 Running Tests

```bash
# Run all unit tests
python3.11 -m pytest batch_cleaner/tests/test_cleaning.py -v

# Expected: 19 passed
```

---

## RL Environment

### Action Space

| Action | Description | Reward |
|--------|-------------|--------|
| `fix_missing` | Fill a null/empty cell | `+0.15` |
| `fix_format` | Rewrite bad format (date, phone, email) | `+0.18` |
| `remove_duplicate` | Delete duplicate row | `+0.12` |
| `fix_type` | Convert wrong data type | `+0.15` |
| `fix_outlier` | Replace impossible value | `+0.20` |
| `no_op` | No action | `-0.05` |

### Task Levels

| Level | Issues | Max Steps | Baseline (GPT-4) |
|-------|--------|-----------|-----------------|
| Easy | 5–8 | 15 | ~0.91 |
| Medium | 12–20 | 30 | ~0.73 |
| Hard | 25–40 | 50 | ~0.51 |

### Python Client Example

```python
from client import DataCleanClient

client = DataCleanClient("http://127.0.0.1:7860")
obs = client.reset(task_level="easy")

while not obs["done"]:
    action = {
        "action_type": "fix_missing",
        "row_index": 0,
        "column_name": "age",
        "new_value": "28",
        "reason": "Fill missing age with median"
    }
    obs, reward, done = client.step(action)
    print(f"Reward: {reward}")
```

---

## 🔄 Updating & Re-running After Changes

```bash
# If you edit any Python file, uvicorn --reload auto-restarts the server
# If you edit dashboard.html, just refresh the browser (F5)

# Re-run tests after changes
python3.11 -m pytest batch_cleaner/tests/test_cleaning.py -v

# Run RL demo
python3.11 demo.py

# Run baseline evaluation
python3.11 baseline.py
```

---

## 🐳 Docker — Full Environment

```yaml
# docker-compose.yml — already configured
services:
  datacleanenv:
    build: .
    ports:
      - "7860:7860"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./batch_cleaner/reports:/app/batch_cleaner/reports
```

```bash
# Start
docker-compose up --build

# Stop
docker-compose down

# View logs
docker-compose logs -f
```

---

## 📊 Baseline Results

| Task | GPT-3.5 | GPT-4 |
|------|---------|-------|
| Easy | ~0.72 | ~0.91 |
| Medium | ~0.48 | ~0.73 |
| Hard | ~0.28 | ~0.51 |

> Run `python3.11 baseline.py` against a live server to reproduce.

---

## 🔑 Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Optional | Enables AI cleaning suggestions |
| `HUGGING_FACE_HUB_TOKEN` | Optional | For `openenv push` to HF Spaces |

---

## 📄 License

MIT — see LICENSE file.

---

## 📚 Full Documentation

See [PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md) for:
- Complete architecture diagram
- All module descriptions
- API endpoint reference
- Docker configuration
- Configuration reference
- Known limitations & roadmap

# Deta-clean-env
