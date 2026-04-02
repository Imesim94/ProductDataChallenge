# 🦃 ProductDataChallenge: Thanksgiving 2021 MLOps Pipeline

[![CI Pipeline](https://github.com/Imesim94/ProductDataChallenge/actions/workflows/ci.yml/badge.svg)](https://github.com/Imesim94/ProductDataChallenge/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Data Science & Advanced Analytics — Technical Case Study**
>
> **Author:** Sinethemba Makoma | Senior Data Scientist & ML Engineer Candidate

---

## 🏗️ System Architecture

This project is engineered as a **three-tier MLOps ecosystem**. It bypasses the limitations of traditional "notebook-only" data science by providing a reproducible, containerized environment for both analytical discovery and real-time inference.

1.  **Core Intelligence:** A TF-IDF + LightGBM classifier optimized for high-cardinality Google Merchant taxonomies.
2.  **Serving Layer:** A high-concurrency FastAPI server capable of batch and real-time inference ($P_{99} < 20ms$).
3.  **Observability Layer:** An 8-page Streamlit BI dashboard for executive-level decision-making.

---

## 🚀 Deployment: The "One-Command" Workflow

The entire stack is orchestrated via Docker Compose. This ensures that "it works on my machine" translates to "it works in production."

```bash
# Clone and Deploy the full stack (API + Dashboard + MLflow)
git clone https://github.com/Imesim94/ProductDataChallenge.git
cd ProductDataChallenge
docker-compose up --build