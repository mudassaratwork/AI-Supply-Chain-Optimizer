# AI Supply Chain Optimizer

An end-to-end machine learning system that forecasts product demand and recommends optimal inventory levels for a multi-SKU supply chain. The project combines classical time-series models (SARIMA, Prophet) with a gradient-boosted regressor (XGBoost), feeds the forecasts into an inventory-optimization layer (EOQ + safety stock), and serves predictions through a FastAPI backend, a Streamlit dashboard, and a serverless AWS Lambda endpoint.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Component Details](#component-details)
5. [Tech Stack](#tech-stack)
6. [Getting Started](#getting-started)
7. [Build Roadmap](#build-roadmap)
8. [API Reference](#api-reference)
9. [Data Model](#data-model)
10. [Modeling Approach](#modeling-approach)
11. [Testing](#testing)
12. [Deployment](#deployment)
13. [CI/CD](#cicd)
14. [Roadmap & Future Work](#roadmap--future-work)
15. [License](#license)

---

## Overview

Most retailers and distributors lose margin in two places: stock-outs that miss revenue, and overstock that ties up capital and shelf space. This project tackles both by:

1. **Forecasting demand** at the SKU-day granularity using ensemble ML.
2. **Translating forecasts into inventory decisions** (reorder point, safety stock, economic order quantity).
3. **Exposing the results** via a REST API, an interactive dashboard, and a serverless inference endpoint so business users and downstream systems can consume them.

The full pipeline — ingestion, transformation, training, serving, and deployment — runs locally with Docker Compose and ships to AWS Lambda for production inference.

---

## Architecture

```
                    ┌──────────────────────────────────────────────────────────────┐
       DATA         │  Sales history    External signals    Inventory    Supplier  │
                    │  (CSV/ERP)        (weather/holiday)   (snapshots)  (lead/$)  │
                    └────────┬──────────────────┬────────────────┬──────────┬──────┘
                             │                  │                │          │
                             ▼                  ▼                ▼          ▼
                    ┌──────────────────────────────────────────────────────────────┐
       PROCESS      │   ETL Pipeline  ───►   PostgreSQL                            │
                    │   Clean · Transform · Load   (feature store / history)       │
                    └────────┬─────────────────────────┬───────────────────────────┘
                             │                         │
                             ▼                         ▼
                    ┌──────────────────────────────────────────────────────────────┐
       MODELS       │  Time-series        XGBoost              Inventory optimizer │
                    │  SARIMA · Prophet   Demand · features    EOQ · safety stock  │
                    └────────┬─────────────────┬──────────────────────┬────────────┘
                             │                 │                      │
                             ▼                 ▼                      ▼
                    ┌──────────────────────────────────────────────────────────────┐
       SERVE        │   FastAPI ───► Streamlit dashboard ───► AWS Lambda           │
                    │   REST /predict   Charts · Alerts · KPIs    Serverless infer │
                    └──────────────────────────────────────────────────────────────┘
```

**Layered architecture, four planes:**

| Plane     | Responsibility                                                                 |
| --------- | ------------------------------------------------------------------------------ |
| `DATA`    | Heterogeneous raw inputs — sales, weather/holidays, on-hand stock, suppliers.  |
| `PROCESS` | Cleansing, feature engineering, persistence into a Postgres feature store.     |
| `MODELS`  | Time-series, gradient-boosted, and operations-research models cooperating.     |
| `SERVE`   | REST API, Streamlit dashboard, and serverless inference for production usage.  |

---

## Project Structure

```
supply-chain-optimizer/
├── data/
│   ├── raw/                  # Raw CSVs (sales, inventory, suppliers)
│   └── processed/            # Cleaned, feature-engineered data
├── etl/
│   ├── ingest.py             # Pulls data from sources
│   ├── transform.py          # Cleaning, feature engineering
│   └── load.py               # Loads into PostgreSQL
├── models/
│   ├── forecaster.py         # SARIMA + Prophet time-series models
│   ├── xgboost_model.py      # XGBoost demand forecasting
│   ├── inventory_optimizer.py# EOQ, safety stock calculations
│   └── train.py              # Training pipeline entry point
├── api/
│   ├── main.py               # FastAPI app
│   ├── routes/
│   │   ├── forecast.py       # POST /predict/demand
│   │   ├── inventory.py      # GET /inventory/recommendations
│   │   └── health.py         # GET /health
│   └── schemas.py            # Pydantic request/response models
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── deploy/
│   ├── lambda_handler.py     # AWS Lambda entry point
│   └── Dockerfile            # Container for Lambda
├── tests/
│   ├── test_etl.py
│   ├── test_models.py
│   └── test_api.py
├── .github/workflows/
│   └── ci.yml                # GitHub Actions CI/CD
├── docker-compose.yml        # Local dev (API + PostgreSQL)
├── requirements.txt
└── README.md
```

---

## Component Details

### ETL Pipeline (`etl/`)

Ingests raw sales history CSVs, external signals (holidays, weather via API), and inventory snapshots. Cleans nulls, encodes categoricals, engineers lag features (7-day and 30-day rolling averages, day-of-week, month, promo flags), and loads everything into PostgreSQL using SQLAlchemy. Designed to run on a cron schedule or be triggered manually.

- `ingest.py` — Reads CSVs from `data/raw/` and pulls external data from holiday/weather APIs.
- `transform.py` — Null handling, categorical encoding, lag/rolling features, holiday joins.
- `load.py` — Bulk-inserts cleaned frames into Postgres using SQLAlchemy `to_sql`.

### Models (`models/`)

Three models work together:

- **`forecaster.py`** — SARIMA (statsmodels) and Facebook Prophet for time-series demand forecasting. Handles weekly/yearly seasonality, trend, and holiday effects automatically. Useful as a strong baseline and for SKUs with sparse covariates.
- **`xgboost_model.py`** — XGBoost regression trained on engineered features (lag sales, price, promotions, category, weather, holiday indicators). Outperforms pure time-series approaches when many external features are available.
- **`inventory_optimizer.py`** — Consumes forecasted demand to compute Economic Order Quantity (EOQ) and safety stock per SKU, then derives reorder points and recommended order quantities.
- **`train.py`** — Orchestrator: pulls features from Postgres, trains each model, evaluates against a hold-out (MAE/RMSE/MAPE), and persists artifacts with `joblib`.

### FastAPI Backend (`api/`)

Three endpoints exposed:

- `POST /predict/demand` — Takes `product_id` + `horizon` (days) and returns a forecast curve.
- `GET /inventory/recommendations` — Returns reorder suggestions for all SKUs.
- `GET /health` — Liveness/readiness probe.

All inputs and responses are validated with Pydantic schemas (`schemas.py`). Served by Uvicorn.

### Streamlit Dashboard (`dashboard/`)

Connects to the FastAPI backend (and Postgres directly for raw history). Surfaces:

- Demand forecast charts (Plotly) per SKU.
- Inventory level gauges with on-hand vs. reorder point.
- Low-stock alerts table.
- Model accuracy metrics: MAE, RMSE, MAPE.
- Manual + timer-driven refresh.

### AWS Lambda Deploy (`deploy/`)

Trained model artifacts (`joblib`) are packaged into a Docker container and pushed to ECR, then deployed to Lambda. `lambda_handler.py` loads the model once on cold start and serves inference requests. API Gateway sits in front of Lambda for HTTP routing.

---

## Tech Stack

| Purpose                | Library / Service                          |
| ---------------------- | ------------------------------------------ |
| Time-series forecasting| `statsmodels` (SARIMA), `prophet`          |
| Gradient boosting      | `xgboost`, `scikit-learn`                  |
| Data manipulation      | `pandas`, `numpy`                          |
| Database               | `sqlalchemy`, `psycopg2`, PostgreSQL       |
| API                    | `fastapi`, `uvicorn`, `pydantic`           |
| Dashboard              | `streamlit`, `plotly`                      |
| Model persistence      | `joblib`                                   |
| Testing                | `pytest`, `httpx`                          |
| Cloud                  | `boto3` (AWS SDK), AWS Lambda, ECR, API GW |
| Containers / Dev       | Docker, Docker Compose                     |
| CI/CD                  | GitHub Actions                             |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- (Optional) AWS CLI v2 configured with deploy credentials

### Local Setup

```bash
# 1. Clone and enter the project
git clone <your-fork-url> supply-chain-optimizer
cd supply-chain-optimizer

# 2. Create a virtualenv and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Spin up Postgres + the FastAPI service
docker compose up -d

# 4. Generate synthetic data and run the ETL
python -m etl.ingest
python -m etl.transform
python -m etl.load

# 5. Train models and persist artifacts
python -m models.train

# 6. Launch the dashboard
streamlit run dashboard/app.py
```

The API will be at `http://localhost:8000`, dashboard at `http://localhost:8501`, and Postgres at `localhost:5432`.

---

## Build Roadmap

A step-by-step plan to reproduce the project from scratch:

1. **Generate a synthetic dataset** — 3 years of daily sales for 50 SKUs with weekly + yearly seasonality and promo bursts, using `pandas` + `numpy`.
2. **Build the ETL** locally against SQLite for speed, then swap to PostgreSQL via Docker.
3. **Train the SARIMA and XGBoost models**, evaluate with MAE/RMSE, save artifacts with `joblib`.
4. **Implement the FastAPI endpoints**, validate with `pytest` + `httpx`.
5. **Build the Streamlit dashboard** pointed at local FastAPI.
6. **Containerize with Docker** and deploy to AWS Lambda via ECR.
7. **Wire up GitHub Actions** to run tests on every push and build the Docker image on merge to `main`.

---

## API Reference

### `POST /predict/demand`

Request:
```json
{
  "product_id": "SKU-00042",
  "horizon_days": 14
}
```

Response:
```json
{
  "product_id": "SKU-00042",
  "horizon_days": 14,
  "forecast": [
    {"date": "2026-05-20", "units": 124.3, "lower": 110.1, "upper": 137.8},
    {"date": "2026-05-21", "units": 130.2, "lower": 115.6, "upper": 144.4}
  ],
  "model": "xgboost"
}
```

### `GET /inventory/recommendations`

Returns a list of reorder recommendations across all SKUs:
```json
[
  {
    "product_id": "SKU-00042",
    "on_hand": 312,
    "reorder_point": 480,
    "recommended_order_qty": 600,
    "safety_stock": 90
  }
]
```

### `GET /health`

```json
{"status": "ok", "version": "0.1.0"}
```

---

## Data Model

PostgreSQL schema (simplified):

| Table                | Purpose                                            |
| -------------------- | -------------------------------------------------- |
| `sales_history`      | One row per SKU-day: units sold, price, promo flag |
| `inventory_snapshot` | Daily on-hand stock per SKU/warehouse              |
| `suppliers`          | Lead time, unit cost, MOQ per supplier             |
| `external_signals`   | Holidays, weather features keyed by date           |
| `features`           | Engineered feature store consumed by models        |
| `forecasts`          | Persisted forecast outputs, versioned by run       |

---

## Modeling Approach

- **Baseline:** Naive (last-week) and seasonal-naive to set a floor.
- **Classical:** SARIMA per-SKU for weekly + yearly seasonality; Prophet where holiday effects dominate.
- **ML:** XGBoost over a feature panel — lag sales (1, 7, 14, 30 day), rolling means/stds, calendar features, price, promo, weather, holiday flags.
- **Selection:** Per-SKU MAPE on a hold-out fold; the best of {SARIMA, Prophet, XGBoost} is registered for that SKU.
- **Inventory layer:**
  - Safety stock: `Z × σ_demand × √lead_time`
  - EOQ: `√(2DS / H)` where D = annual demand, S = order cost, H = holding cost.
  - Reorder point: `mean_demand × lead_time + safety_stock`.

Metrics tracked: **MAE**, **RMSE**, **MAPE** per SKU; aggregate weighted by revenue.

---

## Testing

```bash
pytest -q
```

- `tests/test_etl.py` — Schema checks, null handling, feature correctness.
- `tests/test_models.py` — Deterministic training on a fixture, sanity bounds on MAE.
- `tests/test_api.py` — `httpx` calls against the FastAPI app; status codes, schema validation, error paths.

---

## Deployment

### Build the Lambda image

```bash
docker build -f deploy/Dockerfile -t supply-chain-optimizer:latest .
```

### Push to ECR and deploy

```bash
aws ecr create-repository --repository-name supply-chain-optimizer
docker tag supply-chain-optimizer:latest <acct>.dkr.ecr.<region>.amazonaws.com/supply-chain-optimizer:latest
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
docker push <acct>.dkr.ecr.<region>.amazonaws.com/supply-chain-optimizer:latest

aws lambda create-function \
  --function-name supply-chain-optimizer \
  --package-type Image \
  --code ImageUri=<acct>.dkr.ecr.<region>.amazonaws.com/supply-chain-optimizer:latest \
  --role arn:aws:iam::<acct>:role/lambda-exec-role
```

Front it with API Gateway (HTTP API) to expose `/predict/demand` over HTTPS.

---

## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request:

1. Install dependencies and cache pip.
2. Lint (`ruff`) and type-check (`mypy`) the codebase.
3. Run `pytest`.
4. On `main`, build the Docker image and push to ECR.

---

## Roadmap & Future Work

- Replace synthetic data with a real retailer dataset (e.g., M5).
- Add a model registry (MLflow) and experiment tracking.
- Hierarchical reconciliation across SKU → category → region.
- Multi-echelon inventory optimization (warehouse → store).
- Drift detection and automated retraining triggered by feature/concept drift.
- Slack/Email alerting on low-stock and forecast-anomaly events.

---

## License

MIT — see `LICENSE`.
