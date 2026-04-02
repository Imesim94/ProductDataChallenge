"""
Sales Feature Engineering
=========================
Aggregation, trend computation, and analytical features
for Thanksgiving week sales analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sales Trend Analysis
# ---------------------------------------------------------------------------

def daily_sales_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily sales aggregates across Thanksgiving week.

    Returns
    -------
    pd.DataFrame with columns:
        date, day_name, total_revenue, total_commission, txn_count,
        avg_order_value, avg_commission_rate, unique_items, unique_countries
    """
    daily = (
        df.groupby(["date_str", "day_of_week"])
        .agg(
            total_revenue=("saleamt", "sum"),
            total_commission=("commission", "sum"),
            txn_count=("action_id", "count"),
            avg_order_value=("saleamt", "mean"),
            avg_commission_rate=("commission_rate", "mean"),
            unique_items=("item_name", "nunique"),
            unique_countries=("country", "nunique"),
        )
        .reset_index()
        .sort_values("date_str")
    )

    return daily


def top_selling_products(
    df: pd.DataFrame,
    n: int = 20,
    by: str = "revenue",
    category_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Rank top-selling products by revenue or transaction count.

    Parameters
    ----------
    df : pd.DataFrame
    n : int
        Number of top products to return
    by : str
        Ranking metric: "revenue" or "count"
    category_col : str, optional
        If provided, compute top products within each category

    Returns
    -------
    pd.DataFrame
    """
    metric_col = "saleamt" if by == "revenue" else "action_id"
    agg_func = "sum" if by == "revenue" else "count"

    if category_col and category_col in df.columns:
        result = (
            df.groupby([category_col, "item_name"])
            .agg(
                total_revenue=("saleamt", "sum"),
                total_commission=("commission", "sum"),
                txn_count=("action_id", "count"),
            )
            .reset_index()
            .sort_values(
                [category_col, "total_revenue" if by == "revenue" else "txn_count"],
                ascending=[True, False],
            )
        )
        # Top n per category
        result = result.groupby(category_col).head(n).reset_index(drop=True)
    else:
        result = (
            df.groupby("item_name")
            .agg(
                total_revenue=("saleamt", "sum"),
                total_commission=("commission", "sum"),
                txn_count=("action_id", "count"),
                avg_commission_rate=("commission_rate", "mean"),
            )
            .reset_index()
            .sort_values(
                "total_revenue" if by == "revenue" else "txn_count", ascending=False
            )
            .head(n)
        )

    return result


def highest_commissioned_items(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Find products with highest total and rate-based commissions.

    Returns both total commission leaders and highest commission rate items.
    """
    by_total = (
        df.groupby("item_name")
        .agg(
            total_commission=("commission", "sum"),
            total_revenue=("saleamt", "sum"),
            txn_count=("action_id", "count"),
            avg_commission_rate=("commission_rate", "mean"),
        )
        .reset_index()
        .sort_values("total_commission", ascending=False)
        .head(n)
    )

    # Filter to items with meaningful volume for rate-based ranking
    min_txns = 2
    by_rate = (
        df[df["saleamt"] > 0]
        .groupby("item_name")
        .agg(
            avg_commission_rate=("commission_rate", "mean"),
            total_commission=("commission", "sum"),
            total_revenue=("saleamt", "sum"),
            txn_count=("action_id", "count"),
        )
        .reset_index()
        .query(f"txn_count >= {min_txns}")
        .sort_values("avg_commission_rate", ascending=False)
        .head(n)
    )

    return by_total, by_rate


def black_friday_vs_cyber_monday(df: pd.DataFrame) -> Dict:
    """
    Compare sales patterns between Black Friday and Cyber Monday.

    Returns
    -------
    dict with keys:
        bf_summary, cm_summary, comparison, category_comparison
    """
    bf = df[df["is_black_friday"]].copy()
    cm = df[df["is_cyber_monday"]].copy()

    def summarize(subset: pd.DataFrame, label: str) -> Dict:
        return {
            "label": label,
            "total_revenue": subset["saleamt"].sum(),
            "total_commission": subset["commission"].sum(),
            "txn_count": len(subset),
            "avg_order_value": subset["saleamt"].mean(),
            "avg_commission_rate": subset["commission_rate"].mean(),
            "unique_items": subset["item_name"].nunique(),
            "unique_countries": subset["country"].nunique(),
            "top_country": subset["country"].value_counts().index[0]
            if len(subset) > 0
            else None,
        }

    bf_summary = summarize(bf, "Black Friday")
    cm_summary = summarize(cm, "Cyber Monday")

    # Category-level comparison (if taxonomy exists)
    cat_comparison = None
    if "category_L0" in df.columns:
        cat_bf = (
            bf.dropna(subset=["category_L0"])
            .groupby("category_L0")
            .agg(revenue=("saleamt", "sum"), count=("action_id", "count"))
            .reset_index()
            .rename(columns={"revenue": "bf_revenue", "count": "bf_count"})
        )
        cat_cm = (
            cm.dropna(subset=["category_L0"])
            .groupby("category_L0")
            .agg(revenue=("saleamt", "sum"), count=("action_id", "count"))
            .reset_index()
            .rename(columns={"revenue": "cm_revenue", "count": "cm_count"})
        )
        cat_comparison = cat_bf.merge(cat_cm, on="category_L0", how="outer").fillna(0)
        cat_comparison["revenue_shift"] = (
            cat_comparison["cm_revenue"] - cat_comparison["bf_revenue"]
        )

    return {
        "bf_summary": bf_summary,
        "cm_summary": cm_summary,
        "category_comparison": cat_comparison,
    }


def country_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize sales by country."""
    return (
        df.groupby("country")
        .agg(
            total_revenue=("saleamt", "sum"),
            total_commission=("commission", "sum"),
            txn_count=("action_id", "count"),
            unique_items=("item_name", "nunique"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
