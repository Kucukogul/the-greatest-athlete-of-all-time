# The Greatest Athlete of All Time

An analytical platform that measures athletic superiority across sports using statistical modeling and normalization systems.
---

## Folder Structure

```
the-greatest-athlete-of-all-time/
│
├── src/
│   ├── data/         # DataLoader, validators
│   ├── features/     # Normalization, era adjustment
│   ├── models/       # BaseAthleteModel
│   ├── scoring/      # AthleteScorer, ScoringConfig
│   ├── normalize/    # Sport-specific normalizers
│   ├── pipelines/    # Data transformation pipelines
│   ├── visualization/# Plotly chart functions
│   └── utils/        # Lightweight helpers
│
├── dashboards/
│   └── app.py        # Streamlit dashboard
│
├── notebooks/
│   ├── exploration/  # EDA, one-off analysis
│   └── experiments/  # Scoring experiments
│
├── tests/
│   ├── unit/
│   ├── regression/
│   ├── fixtures/
│   └── conftest.py
│
├── configs/
│   └── scoring_tennis.yaml
│
├── requirements.txt
└── main.py
```

---

## Setup & Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run dashboards/app.py   # dashboard
uvicorn src.api.main:app --reload  # API
python main.py                     # pipeline
pytest tests/ -v --cov=src         # tests
```

---

## Author

Huseyin Kucukogul · MIT License
