"""
Data Loader Module
==================
Handles loading raw CSV data and configuration files.
Designed for reproducibility: every load path is config-driven.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/data_config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_config(config_path: str = "configs/model_config.yaml") -> dict:
    """Load model configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_sales_data(
    path: Optional[str] = None, config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Load raw product sales data.

    Parameters
    ----------
    path : str, optional
        Direct path to CSV file. Overrides config path.
    config : dict, optional
        Data config dict. Loaded from default if not provided.

    Returns
    -------
    pd.DataFrame
        Raw sales dataframe with columns:
        [action_id, sqldate, item_name, country, payout_type, saleamt, commission]
    """
    if config is None:
        config = load_config()

    filepath = path or config["paths"]["raw_sales"]
    logger.info(f"Loading sales data from: {filepath}")

    df = pd.read_csv(
        filepath,
        sep="\t" if filepath.endswith(".tsv") else ",",
        encoding="utf-8",
        on_bad_lines="warn",
    )

    logger.info(f"Loaded {len(df):,} sales records with {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Date range: {df['sqldate'].min()} to {df['sqldate'].max()}")

    return df


def load_taxonomy_data(
    path: Optional[str] = None, config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Load product taxonomy mapping data (Google Merchant categories).

    Parameters
    ----------
    path : str, optional
        Direct path to CSV file.
    config : dict, optional
        Data config dict.

    Returns
    -------
    pd.DataFrame
        Taxonomy dataframe with columns: [item_name, original_format_category]
    """
    if config is None:
        config = load_config()

    filepath = path or config["paths"]["raw_taxonomy"]
    logger.info(f"Loading taxonomy data from: {filepath}")

    df = pd.read_csv(filepath, sep="\t", encoding="utf-8", on_bad_lines="warn")

    logger.info(f"Loaded {len(df):,} taxonomy records")
    logger.info(
        f"Unique categories (raw): {df['original_format_category'].nunique()}"
    )

    return df


def load_processed_data(config: Optional[dict] = None) -> pd.DataFrame:
    """Load the merged, cleaned dataset from parquet."""
    if config is None:
        config = load_config()

    filepath = config["paths"]["merged"]
    logger.info(f"Loading processed data from: {filepath}")
    return pd.read_parquet(filepath)
