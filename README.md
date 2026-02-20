# KPI Anomaly Detection (Time-Series Sliding Window)

Small, internship-friendly project layout for KPI anomaly detection using:
- **Per-KPI time-series split** (no mixing different KPIs)
- **Sliding windows** (past window length `W`, future horizon `H`)
- **Sklearn classifier** + **threshold selection** to target a minimum precision

## What this repo does
For each KPI series, we create windows of length `W` from past values and label each window as **1** if there is **any anomaly in the next `H` timestamps**.

Then we train a baseline model (Logistic Regression or Random Forest), pick a probability threshold to achieve a minimum precision, and evaluate.

## Quickstart

### 1) Setup
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Put data in `data/raw/`
This repo expects:
- `data/raw/train.csv`
- `data/raw/test.csv`

(These files are gitignored.)

### 3) Run training
```bash
python -m pip install -e .
python scripts/train.py --train_csv data/raw/train.csv --test_csv data/raw/test.csv --W 50 --H 20 --model logreg --min_precision 0.7
```

Pick a specific KPI ID:
```bash
python scripts/train.py --train_csv data/raw/train.csv --kpi_id <YOUR_KPI_ID>
```

## Project structure
- `src/kpi_ad/data.py` — load/split utilities + anomaly interval extraction
- `src/kpi_ad/windows.py` — sliding window creation
- `src/kpi_ad/model.py` — `TimeSeriesModel` wrapper (scaling + threshold selection)
- `scripts/train.py` — runnable entrypoint

## Notes
- **Do not mix windows across different KPI IDs.** Always split per KPI, then window.
- Threshold tuning is done on a validation slice (last 20% of training windows).
