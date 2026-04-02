"""
Text Feature Engineering
========================
Transforms product item names into features for classification.

Design Rationale
----------------
We use TF-IDF rather than transformer embeddings for several reasons:
1. Interpretability: SHAP values on TF-IDF features produce human-readable
   explanations (e.g., "the word 'shoe' increased P(Apparel) by 0.3").
2. Latency: TF-IDF + LightGBM inference is <5ms per item vs ~50ms for BERT.
3. Data size: With ~100-200 labeled taxonomy items, a fine-tuned transformer
   would overfit. TF-IDF + shallow model generalizes better at this scale.
4. Production simplicity: No GPU dependency for serving.

If the taxonomy training set grows to 10K+ items, revisiting sentence
transformers (e.g., all-MiniLM-L6-v2) with a simple classifier head would
be worth benchmarking.
"""

import re
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_item_text(text: str) -> str:
    """
    Clean a single product item name for vectorization.

    Steps:
    1. Lowercase
    2. Remove HTML entities (&#39, &#174, etc.)
    3. Remove size/color suffixes that add noise (e.g., "Black M", "Size 7.5")
    4. Remove special characters but keep hyphens and apostrophes
    5. Collapse whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove HTML entities
    text = re.sub(r"&#?\w+;?", " ", text)

    # Remove common size patterns: "S", "M", "L", "XL", "2X", "36x30", "Size 7.5"
    text = re.sub(
        r"\b(size\s*)?\d+(\.\d+)?\s*(x\s*\d+)?\b", " ", text, flags=re.IGNORECASE
    )
    text = re.sub(r"\b[XSML]{1,3}\b", " ", text)

    # Remove color words that appear as suffixes (common noise in product names)
    # Keeping them as features actually helps classification, so we only remove
    # standalone color-only entries
    text = re.sub(r"[^a-z0-9\s\-']", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def batch_clean_text(series: pd.Series) -> pd.Series:
    """Apply text cleaning to a pandas Series of item names."""
    return series.apply(clean_item_text)


# ---------------------------------------------------------------------------
# TF-IDF Vectorizer
# ---------------------------------------------------------------------------

def build_tfidf_vectorizer(
    texts: pd.Series,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> Tuple[TfidfVectorizer, csr_matrix]:
    """
    Fit a TF-IDF vectorizer on product item names and return the
    fitted vectorizer + transformed matrix.

    Parameters
    ----------
    texts : pd.Series
        Cleaned item name text
    max_features : int
        Maximum vocabulary size
    ngram_range : tuple
        (min_n, max_n) for character/word n-grams
    min_df : int
        Minimum document frequency
    max_df : float
        Maximum document frequency
    sublinear_tf : bool
        Apply sublinear TF scaling (1 + log(tf))

    Returns
    -------
    tuple of (TfidfVectorizer, csr_matrix)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w[\w'-]*\b",
    )

    X = vectorizer.fit_transform(texts)

    logger.info(
        f"TF-IDF vectorizer: {X.shape[0]} docs x {X.shape[1]} features "
        f"(vocab size: {len(vectorizer.vocabulary_)})"
    )

    return vectorizer, X


def extract_brand_feature(item_name: str) -> str:
    """
    Heuristic brand extraction from product names.

    Many product names follow patterns like:
    - "adidas Ultraboost 21 Shoes..." -> "adidas"
    - "Nike Air Force 1..." -> "Nike"
    - "Women's Gabby Platform Heels - Universal Thread..." -> ""

    This is a simple first-word heuristic. A production system would
    use a brand dictionary lookup or NER model.
    """
    if not isinstance(item_name, str):
        return "unknown"

    # Known brand patterns (expandable)
    known_brands = {
        "adidas", "nike", "jordan", "puma", "reebok", "ugg", "lego",
        "disney", "marvel", "rugs", "leaps", "kong", "hill's", "nutro",
        "dyson", "dior", "fresh", "supreme", "skechers",
    }

    first_word = item_name.lower().split()[0] if item_name.strip() else "unknown"
    return first_word if first_word in known_brands else "other"


def extract_handcrafted_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract additional handcrafted features from item names.

    These complement TF-IDF by capturing structural patterns:
    - item_word_count: longer names often indicate specific product types
    - has_size: indicates wearable/apparel items
    - has_color: common in fashion/home decor
    - has_brand: whether a known brand is detected
    - name_length: character count
    """
    result = pd.DataFrame(index=df.index)

    names = df["item_name"].fillna("")

    result["item_word_count"] = names.str.split().str.len()
    result["name_length"] = names.str.len()

    # Size indicators
    result["has_size"] = names.str.contains(
        r"\b(?:XS|S|M|L|XL|XXL|\d+x\d+|Size\s+\d+)\b", case=False, regex=True
    ).astype(int)

    # Color indicators
    color_pattern = (
        r"\b(?:black|white|blue|red|green|pink|gray|grey|brown|navy|"
        r"cream|ivory|purple|yellow|orange|gold|silver|beige)\b"
    )
    result["has_color"] = names.str.contains(
        color_pattern, case=False, regex=True
    ).astype(int)

    # Gender indicators
    result["is_womens"] = names.str.contains(
        r"\b(?:women'?s|woman|female|girls'?)\b", case=False, regex=True
    ).astype(int)
    result["is_mens"] = names.str.contains(
        r"\b(?:men'?s|man|male|boys'?)\b", case=False, regex=True
    ).astype(int)

    # Product type signals
    result["is_kids"] = names.str.contains(
        r"\b(?:kids'?|toddler|baby|infant|newborn)\b", case=False, regex=True
    ).astype(int)

    # Brand feature
    result["brand"] = names.apply(extract_brand_feature)

    return result
