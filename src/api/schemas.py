"""
API Schemas
===========
Pydantic models for request/response validation.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ProductItem(BaseModel):
    """Single product item for classification."""
    item_name: str = Field(
        ...,
        description="Product item name to classify",
        examples=["adidas Ultraboost 21 Shoes Core Black 9.5 Mens"],
    )


class ClassificationRequest(BaseModel):
    """Batch classification request."""
    items: List[ProductItem] = Field(
        ...,
        description="List of product items to classify",
        min_length=1,
        max_length=100,
    )


class ClassificationResult(BaseModel):
    """Single classification result."""
    item_name: str
    predicted_category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_categories: Optional[Dict[str, float]] = Field(
        None, description="Top-3 category probabilities"
    )


class ClassificationResponse(BaseModel):
    """Batch classification response."""
    results: List[ClassificationResult]
    model_version: str
    total_items: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: str
    n_classes: int
