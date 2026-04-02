# Product Data Challenge: Thanksgiving 2021 Sales Analysis & Taxonomy Classifier

> **Impact.com Data Science & Advanced Analytics -- Technical Challenge**
>
> Sinethemba Makoma | Senior Data Scientist & ML Engineer

---

## Overview

This project delivers a complete analytical pipeline and production-ready product classification system for Thanksgiving week 2021 e-commerce sales data. Every question from the case study is addressed in the interactive Streamlit dashboard and supported by modular, tested Python code.

**What's included:**

- Exploratory analysis of Thanksgiving sales trends, top products, and commission patterns
- Market basket analysis identifying products and categories purchased together
- Black Friday vs Cyber Monday comparison across revenue, categories, and geography
- TF-IDF + LightGBM product taxonomy classifier with 5-fold stratified cross-validation
- FastAPI production API for real-time classification (sub-5ms inference)
- Dockerized deployment architecture with MLflow experiment tracking
- Business recommendations grounded in the data

## Quick Start

```bash
# 1. Setup
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Place raw data in data/raw/
#    - ProductSales-ThanksGivingWeek.csv
#    - ProductTaxonomy.csv (tab-separated)

# 3. Train the classifier + preprocess all data
PYTHONPATH=. python -m src.models.train

# 4. Launch the interactive dashboard
PYTHONPATH=. streamlit run streamlit_app.py

# 5. (Optional) Serve the classification API
PYTHONPATH=. uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Project Structure

```
ProductDataChallenge/
├── configs/
│   ├── data_config.yaml          # Data paths, date ranges, taxonomy settings
│   └── model_config.yaml         # Hyperparameters, MLflow, API config
├── data/
│   ├── raw/                      # Original CSVs (not in git)
│   └── processed/                # Cleaned parquet outputs
├── models/                       # Serialized classifier artifacts
├── src/
│   ├── data/
│   │   ├── loader.py             # Config-driven data loading
│   │   └── preprocessor.py       # Cleaning, taxonomy parsing, merge pipeline
│   ├── features/
│   │   ├── text_features.py      # TF-IDF vectorization + handcrafted features
│   │   └── sales_features.py     # Temporal, commission, trend aggregations
│   ├── models/
│   │   ├── classifier.py         # LightGBM taxonomy classifier pipeline
│   │   ├── basket_analysis.py    # Market basket co-occurrence analysis
│   │   └── train.py              # Training orchestrator with MLflow tracking
│   └── api/
│       ├── main.py               # FastAPI production server
│       └── schemas.py            # Pydantic request/response models
├── tests/
│   └── test_pipeline.py          # 16 unit tests (all passing)
├── streamlit_app.py              # Interactive dashboard (8 analytical pages)
├── Dockerfile                    # Multi-stage production container
├── Dockerfile.streamlit          # Dashboard container
├── docker-compose.yml            # Full stack: API + MLflow + Dashboard
├── Makefile                      # Dev commands (make train, make serve, etc.)
├── requirements.txt
└── .github/workflows/ci.yml      # GitHub Actions CI pipeline
```

## Addressing Each Case Study Question

### 1. Sales Trends During Thanksgiving 2021

**Dashboard page: Overview + Daily Trends**

The pipeline computes daily aggregates across the full week (Nov 22-29). Key temporal features (day of week, Black Friday/Cyber Monday flags, commission rates) are engineered in `src/data/preprocessor.py`. Revenue and transaction volume are visualized by day with key shopping days highlighted.

### 2. Top Selling Products (Overall and by Category)

**Dashboard page: Top Products**

Products are ranked by total revenue and transaction count. An interactive category filter lets users drill into top products within each Google Merchant taxonomy category. Implemented in `src/features/sales_features.py`.

### 3. Highest Commissioned Product Items

**Dashboard page: Commission Analysis**

Two views: total commission leaders and highest commission rate items (filtered to products with 2+ transactions to avoid single-sale outliers). Payout type distribution (PCT_OF_SALEAMOUNT vs FIXED_AMOUNT) is also visualized.

### 4. Products Frequently Purchased Together

**Dashboard page: Basket Analysis > Order-Level Pairs**

The full `action_id` serves as the order/cart key. Items sharing the same action_id were purchased in a single transaction. Co-occurrence is computed via a vectorized pandas self-join (handles 200K+ orders in seconds).

### 5. Categories Frequently Purchased Together + Classification Method

**Dashboard page: Basket Analysis > Category Co-occurrence + Product Classifier**

**Classification approach:** TF-IDF text vectorization (unigrams + bigrams, max 5000 features) paired with LightGBM, validated with 5-fold stratified cross-validation. The taxonomy training file maps item names to Google Merchant categories. Noisy entries (color values, sizes, non-hierarchical labels) are filtered during preprocessing.

**Why TF-IDF + LightGBM over alternatives:**

| Factor | TF-IDF + LightGBM | Fine-tuned BERT | Zero-shot LLM |
|--------|-------------------|-----------------|----------------|
| Training data (~200 samples) | Strong fit | Overfits | No training needed |
| Inference latency | < 5ms | ~50ms | ~500ms |
| Interpretability | SHAP on text features | Attention maps | Prompt-dependent |
| Production cost (1M items) | ~$0.01 | ~$5 (GPU) | ~$500 (API calls) |
| GPU dependency | None | Required | None |

Category co-occurrence uses order-level baskets first, falling back to merchant-day sessions (capped at 50 items) when individual orders don't span multiple categories.

### 6. Black Friday vs Cyber Monday

**Dashboard page: Black Friday vs Cyber Monday**

Side-by-side comparison of revenue, transaction count, average order value, and unique items. Category-level and country-level breakdowns show where shopping behavior shifts between the two days.

### 7. Business Recommendations

**Dashboard page: Deployment Design (includes recommendations)**

Recommendations are data-driven and tied to specific findings from the analysis, covering merchandising strategy, commission optimization, cross-sell opportunities from basket analysis, and inventory planning from the BF vs CM comparison.

### 8. Deployment Pattern for Product Classification Model

**Dashboard page: Deployment Design**

Full production architecture covering:

- **Real-time API:** FastAPI serving the sklearn pipeline, Dockerized, with health checks and batch inference
- **Batch pipeline:** Scheduled classification of new catalog items with low-confidence flagging for human review
- **Model registry:** MLflow experiment tracking, A/B testing framework, automated promotion gates
- **Monitoring:** Data drift detection, prediction distribution monitoring, latency tracking
- **Scaling:** Cloud Run (low volume) to Kubernetes HPA (high volume) with Redis caching
- **Hybrid ML + LLM fallback:** Trained model handles 80-90% of items cheaply, LLM classifies low-confidence edge cases

## Key Technical Decisions

**Taxonomy cleaning:** Aggressive filtering removes noisy labels ("Green", "1M", "Athletic Fit") that would poison the classifier. Only entries with valid hierarchical paths (containing " > ") are retained.

**Sales cleaning:** Conservative approach. Zero-sale and zero-commission transactions are flagged but not removed, preserving complete transaction coverage for all business questions.

**Basket analysis:** Uses vectorized self-joins instead of Python-loop-based FP-Growth. With 500K+ orders (213K multi-item), traditional `iterrows()` approaches cause memory/timeout failures. The self-join runs in seconds.

**Configuration:** All paths, hyperparameters, and thresholds are YAML-driven (`configs/`), making the pipeline reproducible and easy to modify without touching code.

## Running Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

All 16 tests pass, covering data preprocessing, text feature engineering, classifier components, and API schema validation.

## Docker Deployment

```bash
docker compose build
docker compose up -d

# API:       http://localhost:8000/docs  (Swagger UI)
# MLflow:    http://localhost:5000
# Dashboard: http://localhost:8501
```

## Tech Stack

**ML:** scikit-learn, LightGBM, SHAP, mlxtend |
**API:** FastAPI, Pydantic, Uvicorn |
**Tracking:** MLflow |
**Dashboard:** Streamlit, Plotly |
**Infrastructure:** Docker, Docker Compose, GitHub Actions |
**Data:** pandas, PyArrow

---

*Built with a production-first mindset: modular, tested, containerized, and deployable.*