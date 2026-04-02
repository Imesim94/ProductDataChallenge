"""
Data Preprocessor Module
========================
Cleans raw sales and taxonomy data, handles missing values,
parses taxonomy hierarchy, and produces analysis-ready datasets.

Design Rationale
----------------
- Taxonomy cleaning is aggressive because noisy labels (e.g., "1M", "Green",
  "Athletic Fit") would poison the classifier. We filter to rows with valid
  Google Merchant hierarchical paths (containing " > ").
- Sales cleaning is conservative: we keep all transactions but flag anomalies
  (zero-sale, missing commission) rather than dropping them, because the
  business questions require complete transaction coverage.
"""

import logging
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Taxonomy Cleaning
# ---------------------------------------------------------------------------

def clean_taxonomy(df: pd.DataFrame, separator: str = " > ", min_depth: int = 2) -> pd.DataFrame:
    """
    Clean taxonomy data by filtering to valid Google Merchant hierarchical categories.

    Steps:
    1. Drop rows with null or empty category values
    2. Filter to rows where category contains the hierarchy separator (" > ")
    3. Strip whitespace from item names and categories
    4. Parse hierarchy into levels (L0, L1, L2, ...)
    5. Flag and remove obviously noisy entries (color values, sizes, etc.)

    Parameters
    ----------
    df : pd.DataFrame
        Raw taxonomy with [item_name, original_format_category]
    separator : str
        Hierarchy delimiter in Google Merchant taxonomy
    min_depth : int
        Minimum number of hierarchy levels to keep a record

    Returns
    -------
    pd.DataFrame
        Cleaned taxonomy with parsed hierarchy columns
    """
    logger.info(f"Starting taxonomy cleaning: {len(df):,} records")
    initial_count = len(df)

    # Step 1: Drop nulls
    df = df.dropna(subset=["item_name", "original_format_category"]).copy()
    logger.info(f"After dropping nulls: {len(df):,} ({initial_count - len(df)} removed)")

    # Step 2: Strip whitespace
    df["item_name"] = df["item_name"].str.strip()
    df["original_format_category"] = df["original_format_category"].str.strip()

    # Step 3: Filter to valid hierarchical categories
    # Valid categories contain " > " (e.g., "Apparel & Accessories > Clothing > Shorts")
    # Invalid entries are noise: "1M", "Green", "Athletic Fit", "Ivory", etc.
    has_hierarchy = df["original_format_category"].str.contains(
        re.escape(separator), na=False
    )
    noise_removed = (~has_hierarchy).sum()
    df = df[has_hierarchy].copy()
    logger.info(
        f"After hierarchy filter: {len(df):,} ({noise_removed} noisy entries removed)"
    )

    # Step 4: Parse hierarchy into level columns
    split_cats = df["original_format_category"].str.split(separator, expand=True)
    for i in range(split_cats.shape[1]):
        col_name = f"category_L{i}"
        df[col_name] = split_cats[i].str.strip()

    df["taxonomy_depth"] = df["original_format_category"].str.count(
        re.escape(separator)
    ) + 1

    # Step 5: Filter by minimum depth
    df = df[df["taxonomy_depth"] >= min_depth].copy()
    logger.info(f"After min_depth={min_depth} filter: {len(df):,} records")

    # Step 6: Create primary classification target (L0 = root category)
    df["category_target"] = df["category_L0"]
    logger.info(
        f"Unique root categories: {df['category_target'].nunique()} -> "
        f"{df['category_target'].value_counts().head(10).to_dict()}"
    )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sales Cleaning
# ---------------------------------------------------------------------------

def clean_sales(df: pd.DataFrame, date_format: str = "%Y/%m/%d") -> pd.DataFrame:
    """
    Clean sales transaction data.

    Steps:
    1. Parse dates and create temporal features
    2. Handle missing/malformed numeric columns
    3. Standardize item names
    4. Create derived features (commission_rate, day_of_week, is_weekend)
    5. Flag Black Friday and Cyber Monday transactions

    Parameters
    ----------
    df : pd.DataFrame
        Raw sales data
    date_format : str
        Expected date format string

    Returns
    -------
    pd.DataFrame
        Cleaned sales with temporal and derived features
    """
    logger.info(f"Starting sales cleaning: {len(df):,} records")
    df = df.copy()

    # Step 1: Parse dates
    df["sqldate"] = pd.to_datetime(df["sqldate"], format=date_format, errors="coerce")
    null_dates = df["sqldate"].isna().sum()
    if null_dates > 0:
        logger.warning(f"{null_dates} records with unparseable dates")

    # Step 2: Clean numerics
    for col in ["saleamt", "commission"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Step 3: Standardize item names
    df["item_name"] = df["item_name"].str.strip()
    df["item_name_clean"] = (
        df["item_name"]
        .str.replace(r"[^\w\s\-\./&']", "", regex=True)
        .str.strip()
        .str.lower()
    )

    # Step 4: Temporal features
    df["day_of_week"] = df["sqldate"].dt.day_name()
    df["day_num"] = df["sqldate"].dt.dayofweek  # Monday=0, Sunday=6
    df["date_str"] = df["sqldate"].dt.strftime("%Y-%m-%d")

    # Step 5: Commission rate
    df["commission_rate"] = np.where(
        df["saleamt"] > 0,
        df["commission"] / df["saleamt"],
        0.0,
    )

    # Step 6: Flag key shopping days
    df["is_black_friday"] = df["sqldate"].dt.strftime("%Y-%m-%d") == "2021-11-26"
    df["is_cyber_monday"] = df["sqldate"].dt.strftime("%Y-%m-%d") == "2021-11-29"
    df["is_thanksgiving"] = df["sqldate"].dt.strftime("%Y-%m-%d") == "2021-11-25"

    # Step 7: Flag anomalies (don't remove, just mark)
    df["is_zero_sale"] = df["saleamt"] == 0
    df["is_zero_commission"] = df["commission"] == 0

    logger.info(f"Cleaned sales: {len(df):,} records")
    logger.info(f"Date range: {df['sqldate'].min()} to {df['sqldate'].max()}")
    logger.info(f"Countries: {df['country'].nunique()} unique")
    logger.info(
        f"Zero-sale transactions: {df['is_zero_sale'].sum()} "
        f"({df['is_zero_sale'].mean():.1%})"
    )

    return df


# ---------------------------------------------------------------------------
# Merge Pipeline
# ---------------------------------------------------------------------------

def merge_sales_taxonomy(
    sales_df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Left-join sales with taxonomy on item_name.

    Returns both the merged dataframe and the unmatched sales records
    (which need classification by the ML model).

    Parameters
    ----------
    sales_df : pd.DataFrame
        Cleaned sales data
    taxonomy_df : pd.DataFrame
        Cleaned taxonomy data

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (merged_df, unmatched_df)
    """
    # Prepare taxonomy for join (keep only needed columns)
    tax_cols = [c for c in taxonomy_df.columns if c.startswith("category_")] + [
        "item_name",
        "original_format_category",
        "taxonomy_depth",
    ]
    tax_join = taxonomy_df[tax_cols].drop_duplicates(subset=["item_name"], keep="first")

    merged = sales_df.merge(tax_join, on="item_name", how="left")

    matched = merged["original_format_category"].notna().sum()
    unmatched = merged["original_format_category"].isna().sum()
    match_rate = matched / len(merged)

    logger.info(
        f"Merge results: {matched:,} matched ({match_rate:.1%}), "
        f"{unmatched:,} unmatched ({1 - match_rate:.1%})"
    )

    unmatched_df = merged[merged["original_format_category"].isna()].copy()

    return merged, unmatched_df


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(
    sales_path: str,
    taxonomy_path: str,
    output_dir: str = "data/processed",
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline.

    Parameters
    ----------
    sales_path : str
        Path to raw sales CSV
    taxonomy_path : str
        Path to raw taxonomy CSV
    output_dir : str
        Directory for processed outputs
    config : dict, optional
        Data config

    Returns
    -------
    pd.DataFrame
        Merged, analysis-ready dataframe
    """
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if config is None:
        from src.data.loader import load_config
        config = load_config()

    # Load raw data
    sales_raw = pd.read_csv(sales_path, sep=";", encoding="utf-8", on_bad_lines="warn")
    taxonomy_raw = pd.read_csv(
        taxonomy_path, sep=";", encoding="utf-8", on_bad_lines="warn"
    )

    # Clean
    taxonomy_clean = clean_taxonomy(
        taxonomy_raw,
        separator=config["taxonomy"]["separator"],
        min_depth=config["taxonomy"]["min_depth"],
    )
    sales_clean = clean_sales(
        sales_raw, date_format=config["sales"]["date_format"]
    )

    # Save cleaned intermediates
    taxonomy_clean.to_parquet(f"{output_dir}/taxonomy_cleaned.parquet", index=False)
    sales_clean.to_parquet(f"{output_dir}/sales_cleaned.parquet", index=False)

    # Merge
    merged, unmatched = merge_sales_taxonomy(sales_clean, taxonomy_clean)
    merged.to_parquet(f"{output_dir}/sales_with_taxonomy.parquet", index=False)
    unmatched.to_parquet(f"{output_dir}/unmatched_sales.parquet", index=False)

    logger.info(f"Pipeline complete. Outputs saved to {output_dir}/")
    return merged
