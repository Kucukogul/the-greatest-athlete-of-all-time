# The Greatest Athlete of All Time

A production-grade sports analytics platform that quantifies athletic greatness across eras and disciplines using statistical modeling, normalization systems, and interactive dashboards.

---

## Architecture Philosophy

- **Raw data is immutable.** Never modify files inside `data/raw/`.
- **Processed data is reproducible.** Any file in `data/processed/` can be regenerated from `data/raw/` by running the pipeline.
- **Notebooks are for exploration only.** Production logic lives in `src/`.
- **Scoring is configuration-driven.** No hardcoded weights — all scoring configs live in `configs/`.
- **Visualization is modular.** Every chart is a standalone function in `src/visualization/`.
- **Tests validate numerical correctness.** Edge cases for scoring and normalization are non-negotiable.

---

## Folder Responsibilities

```
the-greatest-athlete-of-all-time/
│
├── data/
│   ├── raw/          # Original source data — READ ONLY
│   ├── processed/    # Cleaned, normalized, feature-engineered outputs
│   └── external/     # Third-party reference data (era definitions, sport averages)
│
├── notebooks/
│   ├── exploration/  # EDA, hypothesis testing, one-off analysis
│   └── experiments/  # Model experiments, scoring experiments
│
├── src/
│   ├── data/         # DataLoader, validators — all I/O goes through here
│   ├── features/     # FeatureEngineer — normalization, era adjustment
│   ├── models/       # BaseAthleteModel and ML model implementations
│   ├── scoring/      # AthleteScorer, ScoringConfig, normalizers
│   ├── visualization/# Plotly chart functions — one function per chart type
│   └── utils/        # safe_divide, timer, flatten_dict — lightweight only
│
├── dashboards/
│   └── app.py        # Streamlit interactive ranking dashboard
│
├── tests/
│   ├── unit/         # Fast, isolated tests for src/ modules
│   ├── fixtures/     # Shared test data files
│   └── conftest.py   # Shared pytest fixtures
│
├── configs/
│   └── scoring_nba.yaml   # Sport-specific scoring weight configs
│
├── agents/           # Claude Code multi-agent definitions
├── skills/           # Reusable Claude Code skill templates
├── docs/             # Architecture decisions, conventions, routing
│
├── requirements.txt
├── main.py           # Batch pipeline entry point
└── CLAUDE.md         # AI agent instructions (not for humans)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data processing | Python, Pandas, NumPy |
| Feature engineering | Scikit-learn |
| Scoring engine | Custom (`src/scoring/`) |
| Visualization | Plotly |
| Dashboard | Streamlit |
| API | FastAPI + Uvicorn |
| Config | YAML |
| Testing | pytest + pytest-cov |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Development Workflow

**Recommended order for a new sport/dataset:**

1. Drop raw data into `data/raw/` — never modify it
2. Load and validate in `src/data/loader.py` + `validators.py`
3. Engineer features in `src/features/engineering.py` (normalize, era-adjust)
4. Save processed output to `data/processed/` via `DataLoader.save_processed()`
5. Create a scoring config in `configs/scoring_<sport>.yaml`
6. Load config and score with `AthleteScorer` from `src/scoring/scorer.py`
7. Visualize results with functions from `src/visualization/plots.py`
8. Run the dashboard: `streamlit run dashboards/app.py`
9. Write tests for any new scoring logic in `tests/unit/`

---

## Running the Project

```bash
# Interactive dashboard
streamlit run dashboards/app.py

# API server
uvicorn src.api.main:app --reload

# Run all tests
pytest tests/ -v --cov=src

# Run only scoring tests
pytest tests/unit/test_scoring.py -v

# Pipeline entry point
python main.py
```

---

## Where Future Features Go

| Feature | Location |
|---|---|
| New sport scoring config | `configs/scoring_<sport>.yaml` |
| New normalization method | `src/scoring/normalizer.py` |
| New ML model | `src/models/<model_name>.py` (extend `BaseAthleteModel`) |
| New chart type | `src/visualization/plots.py` |
| New API endpoint | `src/api/routers/<resource>.py` |
| New data source | `src/data/loader.py` + new method |
| Era adjustment coefficients | `data/external/era_definitions.json` |

---

## Common Mistakes to Avoid

- **Never** call `pd.read_csv()` directly in a FastAPI endpoint — route through `DataLoader`
- **Never** hardcode scoring weights — always load from `configs/`
- **Never** use `print()` in production code — use `logging`
- **Never** import `*` — always explicit imports
- **Never** put experiment code in `src/` — that belongs in `notebooks/experiments/`
- **Never** commit files from `data/raw/`, `data/processed/`, or `models/` to git
- **Never** write two agents for the same task — check `docs/routing.md` first

---

## Agent System

This project uses a Claude Code multi-agent system. Before starting any task, read `docs/routing.md` to identify the correct agent. Each agent file in `agents/` has a strict scope — violating these boundaries degrades AI-assisted workflows.
