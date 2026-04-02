"""
Unit Tests for ProductDataChallenge
====================================
Run: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_taxonomy():
    return pd.DataFrame(
        {
            "item_name": [
                "adidas Ultraboost 21 Shoes Core Black 9.5 Mens",
                "Nike Air Force 1 Low White",
                "Rugs USA Blue Shaggy Rug 8x10",
                "Leaps & Bounds Feathered Fish Cat Toy",
                "Bad Entry",
            ],
            "original_format_category": [
                "Apparel & Accessories > Shoes",
                "Apparel & Accessories > Shoes",
                "Home & Garden > Decor > Rugs",
                "Animals & Pet Supplies > Pet Supplies > Cat Supplies",
                "Green",  # noise
            ],
        }
    )


@pytest.fixture
def sample_sales():
    return pd.DataFrame(
        {
            "action_id": ["1.2.3", "1.2.4", "1.2.5", "2.3.6"],
            "sqldate": [
                "2021/11/26",
                "2021/11/26",
                "2021/11/29",
                "2021/11/29",
            ],
            "item_name": [
                "adidas Ultraboost 21 Shoes Core Black 9.5 Mens",
                "Nike Air Force 1 Low White",
                "Rugs USA Blue Shaggy Rug 8x10",
                "Leaps & Bounds Feathered Fish Cat Toy",
            ],
            "country": ["US", "US", "US", "CA"],
            "payout_type": [
                "PCT_OF_SALEAMOUNT",
                "FIXED_AMOUNT",
                "PCT_OF_SALEAMOUNT",
                "PCT_OF_SALEAMOUNT",
            ],
            "saleamt": [126.0, 81.0, 165.0, 1.39],
            "commission": [8.82, 1.62, 19.80, 0.03],
        }
    )


# ============================================================================
# Data Preprocessor Tests
# ============================================================================

class TestTaxonomyCleaning:
    def test_removes_noisy_entries(self, sample_taxonomy):
        from src.data.preprocessor import clean_taxonomy

        result = clean_taxonomy(sample_taxonomy)
        # "Green" should be removed (no " > " separator)
        assert len(result) == 4
        assert "Bad Entry" not in result["item_name"].values

    def test_parses_hierarchy(self, sample_taxonomy):
        from src.data.preprocessor import clean_taxonomy

        result = clean_taxonomy(sample_taxonomy)
        assert "category_L0" in result.columns
        assert "category_L1" in result.columns
        assert result.iloc[0]["category_L0"] == "Apparel & Accessories"

    def test_creates_target_column(self, sample_taxonomy):
        from src.data.preprocessor import clean_taxonomy

        result = clean_taxonomy(sample_taxonomy)
        assert "category_target" in result.columns
        assert result["category_target"].notna().all()


class TestSalesCleaning:
    def test_parses_dates(self, sample_sales):
        from src.data.preprocessor import clean_sales

        result = clean_sales(sample_sales)
        assert pd.api.types.is_datetime64_any_dtype(result["sqldate"])

    def test_creates_temporal_features(self, sample_sales):
        from src.data.preprocessor import clean_sales

        result = clean_sales(sample_sales)
        assert "day_of_week" in result.columns
        assert "is_black_friday" in result.columns
        assert "is_cyber_monday" in result.columns

    def test_flags_black_friday(self, sample_sales):
        from src.data.preprocessor import clean_sales

        result = clean_sales(sample_sales)
        bf_rows = result[result["is_black_friday"]]
        assert len(bf_rows) == 2  # Nov 26

    def test_commission_rate(self, sample_sales):
        from src.data.preprocessor import clean_sales

        result = clean_sales(sample_sales)
        assert "commission_rate" in result.columns
        assert (result["commission_rate"] >= 0).all()


# ============================================================================
# Text Feature Tests
# ============================================================================

class TestTextFeatures:
    def test_clean_item_text(self):
        from src.features.text_features import clean_item_text

        result = clean_item_text(
            "adidas Ultraboost 21 Shoes Core Black 9.5 Mens"
        )
        assert "adidas" in result
        assert result == result.lower()

    def test_clean_handles_html_entities(self):
        from src.features.text_features import clean_item_text

        result = clean_item_text("Women&#39;s Fleece Hoodie")
        assert "&#" not in result

    def test_extract_brand(self):
        from src.features.text_features import extract_brand_feature

        assert extract_brand_feature("adidas Ultraboost 21") == "adidas"
        assert extract_brand_feature("Nike Air Force 1") == "nike"
        assert extract_brand_feature("Random Product Name") == "other"

    def test_handcrafted_features(self, sample_sales):
        from src.features.text_features import extract_handcrafted_features

        result = extract_handcrafted_features(sample_sales)
        assert "item_word_count" in result.columns
        assert "has_color" in result.columns
        assert "is_womens" in result.columns


# ============================================================================
# Classifier Tests
# ============================================================================

class TestClassifier:
    def test_text_transformer_fit_transform(self, sample_taxonomy):
        from src.data.preprocessor import clean_taxonomy
        from src.models.classifier import TextFeatureTransformer

        tax = clean_taxonomy(sample_taxonomy)
        transformer = TextFeatureTransformer(max_features=100, min_df=1)
        transformer.fit(tax["item_name"])
        X = transformer.transform(tax["item_name"])

        assert X.shape[0] == len(tax)
        assert X.shape[1] <= 100

    def test_predictions_have_correct_shape(self, sample_taxonomy):
        from src.data.preprocessor import clean_taxonomy
        from src.models.classifier import TextFeatureTransformer

        tax = clean_taxonomy(sample_taxonomy)
        transformer = TextFeatureTransformer(max_features=100, min_df=1)
        X = transformer.fit_transform(tax["item_name"])

        assert X.shape[0] == len(tax)


# ============================================================================
# API Schema Tests
# ============================================================================

class TestAPISchemas:
    def test_product_item_validation(self):
        from src.api.schemas import ProductItem

        item = ProductItem(item_name="Test Product")
        assert item.item_name == "Test Product"

    def test_classification_request_validation(self):
        from src.api.schemas import ClassificationRequest, ProductItem

        request = ClassificationRequest(
            items=[ProductItem(item_name="Test Product")]
        )
        assert len(request.items) == 1

    def test_classification_request_max_items(self):
        from src.api.schemas import ClassificationRequest, ProductItem

        with pytest.raises(Exception):
            ClassificationRequest(items=[])  # min_length=1
