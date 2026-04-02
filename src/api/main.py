"""
FastAPI Production API
======================
Serves the product taxonomy classifier via REST endpoints.

Endpoints:
    GET  /health         - Health check with model status
    POST /classify       - Batch product classification
    POST /classify/single - Single item classification

Production Features:
    - Request validation via Pydantic
    - Structured logging
    - CORS middleware for frontend integration
    - Model loaded once at startup (not per-request)
    - Graceful error handling

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    ClassificationResult,
    HealthResponse,
    ProductItem,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model state
MODEL_STATE = {
    "pipeline": None,
    "label_encoder": None,
    "version": "v1.0.0",
    "loaded": False,
}

MODEL_PATH = os.getenv(
    "MODEL_PATH", "models/classifier_pipeline.joblib"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    logger.info(f"Loading model from {MODEL_PATH}")
    try:
        from src.models.classifier import load_model

        pipeline, label_encoder = load_model(MODEL_PATH)
        MODEL_STATE["pipeline"] = pipeline
        MODEL_STATE["label_encoder"] = label_encoder
        MODEL_STATE["loaded"] = True
        logger.info(
            f"Model loaded: {len(label_encoder.classes_)} classes"
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        MODEL_STATE["loaded"] = False

    yield  # App runs here

    # Cleanup
    logger.info("Shutting down, releasing model resources")
    MODEL_STATE["pipeline"] = None
    MODEL_STATE["label_encoder"] = None


app = FastAPI(
    title="Product Taxonomy Classifier API",
    description=(
        "Classifies product item names into Google Merchant taxonomy categories. "
        "Built for the Impact.com Data Science technical challenge."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for Streamlit / frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Model and API health status."""
    return HealthResponse(
        status="healthy" if MODEL_STATE["loaded"] else "degraded",
        model_loaded=MODEL_STATE["loaded"],
        model_version=MODEL_STATE["version"],
        n_classes=(
            len(MODEL_STATE["label_encoder"].classes_)
            if MODEL_STATE["loaded"]
            else 0
        ),
    )


@app.post("/classify", response_model=ClassificationResponse)
async def classify_batch(request: ClassificationRequest):
    """
    Classify a batch of product items.

    Accepts 1-100 items per request. Returns predicted category
    and confidence score for each item.
    """
    if not MODEL_STATE["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check /health for status.",
        )

    start_time = time.time()

    pipeline = MODEL_STATE["pipeline"]
    label_encoder = MODEL_STATE["label_encoder"]

    item_names = pd.Series([item.item_name for item in request.items])

    # Predict
    y_pred = pipeline.predict(item_names)
    y_proba = pipeline.predict_proba(item_names)

    predicted_labels = label_encoder.inverse_transform(y_pred)
    max_proba = y_proba.max(axis=1)

    # Build results with top-3 categories
    results = []
    for i, item in enumerate(request.items):
        top_indices = np.argsort(y_proba[i])[::-1][:3]
        top_categories = {
            label_encoder.classes_[idx]: float(y_proba[i][idx])
            for idx in top_indices
        }

        results.append(
            ClassificationResult(
                item_name=item.item_name,
                predicted_category=predicted_labels[i],
                confidence=float(max_proba[i]),
                top_categories=top_categories,
            )
        )

    elapsed = time.time() - start_time
    logger.info(
        f"Classified {len(results)} items in {elapsed:.3f}s "
        f"({elapsed/len(results)*1000:.1f}ms/item)"
    )

    return ClassificationResponse(
        results=results,
        model_version=MODEL_STATE["version"],
        total_items=len(results),
    )


@app.post("/classify/single", response_model=ClassificationResult)
async def classify_single(item: ProductItem):
    """Classify a single product item."""
    if not MODEL_STATE["loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    pipeline = MODEL_STATE["pipeline"]
    label_encoder = MODEL_STATE["label_encoder"]

    item_series = pd.Series([item.item_name])
    y_pred = pipeline.predict(item_series)
    y_proba = pipeline.predict_proba(item_series)

    predicted_label = label_encoder.inverse_transform(y_pred)[0]

    top_indices = np.argsort(y_proba[0])[::-1][:3]
    top_categories = {
        label_encoder.classes_[idx]: float(y_proba[0][idx])
        for idx in top_indices
    }

    return ClassificationResult(
        item_name=item.item_name,
        predicted_category=predicted_label,
        confidence=float(y_proba[0].max()),
        top_categories=top_categories,
    )
