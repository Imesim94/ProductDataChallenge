"""
Product Data Challenge - Interactive Dashboard
===============================================
Streamlit dashboard presenting Thanksgiving 2021 sales analysis,
product classification, and market basket insights.

Usage:
    streamlit run streamlit_app.py
"""


import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from src.models.basket_analysis import get_fast_cooccurrence


# ============================================================================
# Page Config
# ============================================================================
st.set_page_config(
    page_title="Thanksgiving 2021 Sales Analysis",
    page_icon="🦃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Custom CSS
# ============================================================================
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #e0e0e0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 4px;
    }
    .section-header {
        border-left: 4px solid #6366f1;
        padding-left: 12px;
        margin: 2rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Data Loading
# ============================================================================
@st.cache_data
def load_data():
    """Load processed data. Falls back to raw data with inline cleaning."""
    processed_path = "data/processed/sales_with_taxonomy.parquet"
    if os.path.exists(processed_path):
        return pd.read_parquet(processed_path)

    # Fallback: load and clean raw data inline
    st.info("Processed data not found. Loading raw data...")
    from src.data.preprocessor import clean_sales, clean_taxonomy, merge_sales_taxonomy

    sales = pd.read_csv("data/raw/ProductSales-ThanksGivingWeek.csv")
    taxonomy = pd.read_csv("data/raw/ProductTaxonomy.csv", sep="\t")

    sales_clean = clean_sales(sales)
    taxonomy_clean = clean_taxonomy(taxonomy)
    merged, _ = merge_sales_taxonomy(sales_clean, taxonomy_clean)
    return merged


@st.cache_data
def load_classifications():
    """Load model classification results for unmatched items."""
    path = "data/processed/classified_items.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


# ============================================================================
# Helper Functions
# ============================================================================
def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, icon: str = ""):
    st.markdown(
        f'<div class="section-header"><h3>{icon} {title}</h3></div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# Load Data
# ============================================================================
try:
    df = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.info(
        "Run the preprocessing pipeline first:\n"
        "```python -m src.models.train```"
    )
    data_loaded = False

if not data_loaded:
    st.stop()

classifications = load_classifications()

# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.title("🦃 Thanksgiving 2021")
    st.caption("Product Sales Analysis")
    st.divider()

    # Date filter
    dates = sorted(df["date_str"].dropna().unique())
    selected_dates = st.multiselect(
        "Filter Dates",
        options=dates,
        default=dates,
    )

    # Country filter
    countries = sorted(df["country"].dropna().unique())
    selected_countries = st.multiselect(
        "Filter Countries",
        options=countries,
        default=countries,
    )

    st.divider()
    st.markdown("**Navigation**")
    page = st.radio(
        "Section",
        [
            "Overview",
            "Daily Trends",
            "Top Products",
            "Commission Analysis",
            "Black Friday vs Cyber Monday",
            "Basket Analysis",
            "Product Classifier",
            "Deployment Design",
        ],
        label_visibility="collapsed",
    )

# Apply filters
mask = df["date_str"].isin(selected_dates) & df["country"].isin(selected_countries)
filtered = df[mask].copy()


# ============================================================================
# Page: Overview
# ============================================================================
if page == "Overview":
    st.title("Thanksgiving Week 2021 - Sales Overview")
    st.caption(
        "Product sales analysis for Nov 22-29, 2021 (Thanksgiving through Cyber Monday)"
    )

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        metric_card("Total Revenue", f"${filtered['saleamt'].sum():,.0f}")
    with col2:
        metric_card("Total Commission", f"${filtered['commission'].sum():,.0f}")
    with col3:
        metric_card("Transactions", f"{len(filtered):,}")
    with col4:
        metric_card("Avg Order Value", f"${filtered['saleamt'].mean():,.2f}")
    with col5:
        metric_card("Countries", f"{filtered['country'].nunique()}")

    st.markdown("---")

    # Revenue by day
    daily = (
        filtered.groupby("date_str")
        .agg(revenue=("saleamt", "sum"), count=("action_id", "count"))
        .reset_index()
        .sort_values("date_str")
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=daily["date_str"],
            y=daily["revenue"],
            name="Revenue ($)",
            marker_color="#6366f1",
            opacity=0.85,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date_str"],
            y=daily["count"],
            name="Transaction Count",
            mode="lines+markers",
            line=dict(color="#f59e0b", width=3),
            marker=dict(size=8),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Daily Revenue & Transaction Volume",
        template="plotly_dark",
        height=420,
        margin=dict(t=60, b=40),
    )
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
    fig.update_yaxes(title_text="Transactions", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # Country breakdown
    col_left, col_right = st.columns(2)

    with col_left:
        country_rev = (
            filtered.groupby("country")["saleamt"]
            .sum()
            .reset_index()
            .sort_values("saleamt", ascending=True)
        )
        fig_country = px.bar(
            country_rev,
            x="saleamt",
            y="country",
            orientation="h",
            title="Revenue by Country",
            color="saleamt",
            color_continuous_scale="Viridis",
        )
        fig_country.update_layout(
            template="plotly_dark", height=400, showlegend=False
        )
        st.plotly_chart(fig_country, use_container_width=True)

    with col_right:
        if "category_L0" in filtered.columns:
            cat_rev = (
                filtered.dropna(subset=["category_L0"])
                .groupby("category_L0")["saleamt"]
                .sum()
                .reset_index()
                .sort_values("saleamt", ascending=False)
                .head(10)
            )
            fig_cat = px.bar(
                cat_rev,
                x="saleamt",
                y="category_L0",
                orientation="h",
                title="Top 10 Categories by Revenue",
                color="saleamt",
                color_continuous_scale="Plasma",
            )
            fig_cat.update_layout(
                template="plotly_dark",
                height=400,
                showlegend=False,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_cat, use_container_width=True)


# ============================================================================
# Page: Daily Trends
# ============================================================================
elif page == "Daily Trends":
    st.title("Daily Sales Trends")

    day_labels = {
        "2021-11-22": "Mon (Nov 22)",
        "2021-11-23": "Tue (Nov 23)",
        "2021-11-24": "Wed (Nov 24)",
        "2021-11-25": "Thu - Thanksgiving",
        "2021-11-26": "Fri - Black Friday",
        "2021-11-27": "Sat (Nov 27)",
        "2021-11-28": "Sun (Nov 28)",
        "2021-11-29": "Mon - Cyber Monday",
    }

    daily = (
        filtered.groupby("date_str")
        .agg(
            revenue=("saleamt", "sum"),
            commission=("commission", "sum"),
            count=("action_id", "count"),
            avg_order=("saleamt", "mean"),
            unique_items=("item_name", "nunique"),
        )
        .reset_index()
        .sort_values("date_str")
    )
    daily["day_label"] = daily["date_str"].map(day_labels)
    daily["commission_rate"] = daily["commission"] / daily["revenue"].replace(0, 1)

    # Revenue trend with key day annotations
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=daily["day_label"],
            y=daily["revenue"],
            marker_color=[
                "#6366f1" if d not in ["2021-11-26", "2021-11-29"] else "#f59e0b"
                for d in daily["date_str"]
            ],
            text=[f"${r:,.0f}" for r in daily["revenue"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Revenue by Day (Black Friday & Cyber Monday highlighted)",
        template="plotly_dark",
        height=450,
        yaxis_title="Revenue ($)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    st.dataframe(
        daily[["day_label", "revenue", "commission", "count", "avg_order", "unique_items"]]
        .rename(
            columns={
                "day_label": "Day",
                "revenue": "Revenue ($)",
                "commission": "Commission ($)",
                "count": "Transactions",
                "avg_order": "Avg Order ($)",
                "unique_items": "Unique Items",
            }
        )
        .style.format(
            {
                "Revenue ($)": "${:,.2f}",
                "Commission ($)": "${:,.2f}",
                "Avg Order ($)": "${:,.2f}",
            }
        ),
        use_container_width=True,
    )


# ============================================================================
# Page: Top Products
# ============================================================================
elif page == "Top Products":
    st.title("Top Selling Products")

    col1, col2 = st.columns(2)
    with col1:
        n_products = st.slider("Number of products", 5, 50, 20)
    with col2:
        rank_by = st.selectbox("Rank by", ["Revenue", "Transaction Count"])

    sort_col = "total_revenue" if rank_by == "Revenue" else "txn_count"

    top = (
        filtered.groupby("item_name")
        .agg(
            total_revenue=("saleamt", "sum"),
            total_commission=("commission", "sum"),
            txn_count=("action_id", "count"),
        )
        .reset_index()
        .sort_values(sort_col, ascending=False)
        .head(n_products)
    )

    fig = px.bar(
        top.sort_values(sort_col, ascending=True).tail(20),
        x=sort_col,
        y="item_name",
        orientation="h",
        title=f"Top {n_products} Products by {rank_by}",
        color=sort_col,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        template="plotly_dark",
        height=max(400, n_products * 25),
        yaxis=dict(tickfont=dict(size=10)),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # By category
    if "category_L0" in filtered.columns:
        section_header("Top Products by Category", "📊")
        selected_cat = st.selectbox(
            "Select Category",
            sorted(filtered["category_L0"].dropna().unique()),
        )
        cat_top = (
            filtered[filtered["category_L0"] == selected_cat]
            .groupby("item_name")
            .agg(total_revenue=("saleamt", "sum"), txn_count=("action_id", "count"))
            .reset_index()
            .sort_values("total_revenue", ascending=False)
            .head(10)
        )
        st.dataframe(cat_top, use_container_width=True)


# ============================================================================
# Page: Commission Analysis
# ============================================================================
elif page == "Commission Analysis":
    st.title("Highest Commissioned Products")

    tab1, tab2 = st.tabs(["By Total Commission", "By Commission Rate"])

    with tab1:
        by_total = (
            filtered.groupby("item_name")
            .agg(
                total_commission=("commission", "sum"),
                total_revenue=("saleamt", "sum"),
                txn_count=("action_id", "count"),
            )
            .reset_index()
            .sort_values("total_commission", ascending=False)
            .head(20)
        )
        fig = px.bar(
            by_total.sort_values("total_commission", ascending=True),
            x="total_commission",
            y="item_name",
            orientation="h",
            title="Top 20 Products by Total Commission",
            color="total_commission",
            color_continuous_scale="Turbo",
        )
        fig.update_layout(template="plotly_dark", height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        by_rate = (
            filtered[filtered["saleamt"] > 0]
            .groupby("item_name")
            .agg(
                avg_rate=("commission_rate", "mean"),
                total_commission=("commission", "sum"),
                txn_count=("action_id", "count"),
            )
            .reset_index()
            .query("txn_count >= 2")
            .sort_values("avg_rate", ascending=False)
            .head(20)
        )
        by_rate["avg_rate_pct"] = by_rate["avg_rate"] * 100

        fig = px.bar(
            by_rate.sort_values("avg_rate_pct", ascending=True),
            x="avg_rate_pct",
            y="item_name",
            orientation="h",
            title="Highest Commission Rate Products (min 2 transactions)",
            color="avg_rate_pct",
            color_continuous_scale="Hot",
        )
        fig.update_layout(
            template="plotly_dark",
            height=600,
            showlegend=False,
            xaxis_title="Commission Rate (%)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Payout type distribution
    section_header("Payout Type Distribution", "💰")
    payout_dist = filtered["payout_type"].value_counts().reset_index()
    payout_dist.columns = ["payout_type", "count"]
    fig = px.pie(
        payout_dist,
        values="count",
        names="payout_type",
        title="Transaction Distribution by Payout Type",
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Page: Black Friday vs Cyber Monday
# ============================================================================
elif page == "Black Friday vs Cyber Monday":
    st.title("Black Friday vs Cyber Monday Comparison")

    bf = filtered[filtered["is_black_friday"]]
    cm = filtered[filtered["is_cyber_monday"]]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖤 Black Friday")
        metric_card("Revenue", f"${bf['saleamt'].sum():,.0f}")
        st.markdown("")
        metric_card("Transactions", f"{len(bf):,}")
        st.markdown("")
        metric_card("Avg Order", f"${bf['saleamt'].mean():,.2f}")

    with col2:
        st.subheader("💻 Cyber Monday")
        metric_card("Revenue", f"${cm['saleamt'].sum():,.0f}")
        st.markdown("")
        metric_card("Transactions", f"{len(cm):,}")
        st.markdown("")
        metric_card("Avg Order", f"${cm['saleamt'].mean():,.2f}")

    st.markdown("---")

    # Category comparison
    if "category_L0" in filtered.columns:
        bf_cat = (
            bf.dropna(subset=["category_L0"])
            .groupby("category_L0")["saleamt"]
            .sum()
            .reset_index()
            .rename(columns={"saleamt": "Black Friday"})
        )
        cm_cat = (
            cm.dropna(subset=["category_L0"])
            .groupby("category_L0")["saleamt"]
            .sum()
            .reset_index()
            .rename(columns={"saleamt": "Cyber Monday"})
        )
        comparison = bf_cat.merge(cm_cat, on="category_L0", how="outer").fillna(0)
        comparison = comparison.sort_values("Black Friday", ascending=False).head(10)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=comparison["category_L0"],
                y=comparison["Black Friday"],
                name="Black Friday",
                marker_color="#1e1e2e",
            )
        )
        fig.add_trace(
            go.Bar(
                x=comparison["category_L0"],
                y=comparison["Cyber Monday"],
                name="Cyber Monday",
                marker_color="#6366f1",
            )
        )
        fig.update_layout(
            title="Revenue by Category: Black Friday vs Cyber Monday",
            barmode="group",
            template="plotly_dark",
            height=500,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Country comparison
    section_header("Geographic Comparison", "🌍")
    bf_country = bf.groupby("country")["saleamt"].sum().reset_index().rename(columns={"saleamt": "BF Revenue"})
    cm_country = cm.groupby("country")["saleamt"].sum().reset_index().rename(columns={"saleamt": "CM Revenue"})
    geo_comp = bf_country.merge(cm_country, on="country", how="outer").fillna(0)
    geo_comp["shift"] = geo_comp["CM Revenue"] - geo_comp["BF Revenue"]
    geo_comp = geo_comp.sort_values("BF Revenue", ascending=False)
    st.dataframe(
        geo_comp.style.format({"BF Revenue": "${:,.2f}", "CM Revenue": "${:,.2f}", "shift": "${:,.2f}"}),
        use_container_width=True,
    )


# ============================================================================
# Page: Basket Analysis
# ============================================================================
elif page == "Basket Analysis":
    st.title("Market Basket Analysis")
    st.caption("High-performance association mining using Sparse Matrix Vectorization.")

    # 1. Clean Data Sub-selection
    basket_df = filtered[['action_id', 'item_name', 'saleamt', 'date_str']].copy()
    
    # 2. Compute Metrics (Vectorized)
    counts = basket_df.groupby('action_id').size()
    multi_orders = counts[counts >= 2]
    
    col1, col2, col3 = st.columns(3)
    with col1: metric_card("Multi-Item Orders", f"{len(multi_orders):,}")
    with col2: metric_card("Avg Cart Depth", f"{counts.mean():.2f}")
    with col3: metric_card("Max Cart Size", f"{counts.max()}")

    st.divider()

    tab_strict, tab_context, tab_samples = st.tabs(["Order-Level Pairs", "Merchant-Level Context", "Cart Explorer"])

    with tab_strict:
        section_header("Top Product Affinities (Strict)", "🛒")
        pairs_df = get_fast_cooccurrence(basket_df, 'item_name', 'action_id', top_n=50)
        if not pairs_df.empty:
            st.dataframe(pairs_df.rename(columns={'co_occurrences': 'Frequency'}), use_container_width=True, height=500)
        else:
            st.info("No significant co-occurrences found.")

    with tab_context:
        section_header("Retail Context (Same Advertiser/Day)", "🏪")
        # Logic to create merchant session keys
        basket_df['session_key'] = basket_df['action_id'].str.split('.').str[0] + "_" + basket_df['date_str']
        merchant_pairs = get_fast_cooccurrence(basket_df, 'item_name', 'session_key', top_n=50)
        st.dataframe(merchant_pairs, use_container_width=True, height=500)

    with tab_samples:
        section_header("Example Baskets", "🧺")
        top_ids = multi_orders.nlargest(15).index
        samples = basket_df[basket_df['action_id'].isin(top_ids)].groupby('action_id')['item_name'].apply(list)
        for aid, items in samples.items():
            with st.expander(f"Order {aid} | {len(items)} items"):
                for item in sorted(items): 
                    st.write(f"- {item}")

# ============================================================================
# Page: Product Classifier
# ============================================================================
elif page == "Product Classifier":
    st.title("Product Taxonomy Classifier")
    st.caption("ML-powered classification of products into Google Merchant taxonomy categories")

    tab1, tab2, tab3 = st.tabs(["Try the Classifier", "Model Performance", "Classification Results"])

    with tab1:
        st.subheader("Classify a Product")
        user_input = st.text_input(
            "Enter a product name:",
            value="adidas Ultraboost 21 Running Shoes Black Size 10",
        )

        if st.button("Classify", type="primary"):
            try:
                from src.models.classifier import load_model, predict_categories

                pipeline, label_encoder = load_model()
                result = predict_categories(
                    pipeline, label_encoder, pd.Series([user_input])
                )

                st.success(
                    f"**Predicted Category:** {result['predicted_category'].iloc[0]}"
                )
                st.metric("Confidence", f"{result['confidence'].iloc[0]:.1%}")

            except Exception as e:
                st.error(f"Model not loaded: {e}")
                st.info("Train the model first: `python -m src.models.train`")

        # API integration
        st.markdown("---")
        st.subheader("API Integration")
        st.code(
            """
# Classify via API
import requests

response = requests.post(
    "http://localhost:8000/classify/single",
    json={"item_name": "adidas Ultraboost 21 Running Shoes Black Size 10"}
)
print(response.json())
# {"predicted_category": "Apparel & Accessories", "confidence": 0.92, ...}
            """,
            language="python",
        )

    with tab2:
        st.subheader("Model Performance")
        st.markdown(
            """
            | Metric | Value |
            |--------|-------|
            | Algorithm | TF-IDF + LightGBM |
            | Cross-Validation | 5-fold Stratified |
            | Scoring | F1 Macro |
            | Text Features | Unigrams + Bigrams (max 5000) |
            | Explainability | SHAP |

            **Why TF-IDF + LightGBM over Transformers?**
            - Training set is ~100-200 labeled items (transformers would overfit)
            - Inference latency: <5ms vs ~50ms for BERT
            - Fully interpretable via SHAP feature attributions
            - No GPU required for serving
            """
        )

    with tab3:
        if classifications is not None:
            st.subheader(f"Classified {len(classifications):,} Unmatched Products")

            confidence_threshold = st.slider(
                "Minimum Confidence", 0.0, 1.0, 0.5, 0.05
            )
            filtered_class = classifications[
                classifications["confidence"] >= confidence_threshold
            ]

            # Confidence distribution
            fig = px.histogram(
                classifications,
                x="confidence",
                nbins=30,
                title="Prediction Confidence Distribution",
                color_discrete_sequence=["#6366f1"],
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            # Results table
            st.dataframe(
                filtered_class.sort_values("confidence", ascending=False).head(50),
                use_container_width=True,
            )
        else:
            st.info("Run the training pipeline to generate classification results.")


# ============================================================================
# Page: Deployment Design
# ============================================================================
elif page == "Deployment Design":
    st.title("Production Deployment Architecture")
    st.caption("Design for productionalizing the product taxonomy classifier")

    st.markdown(
        """
        ## System Architecture

        The deployment follows a **three-tier pattern** optimized for Impact.com's
        client-facing product features:

        ### Tier 1: Real-time Classification API
        - **FastAPI** serving the trained TF-IDF + LightGBM pipeline
        - Sub-5ms per-item latency (P99 < 20ms with batching)
        - Containerized with Docker, deployed to Kubernetes/Cloud Run
        - Auto-scaling based on request volume

        ### Tier 2: Batch Classification Pipeline
        - Scheduled Airflow/Prefect DAG for bulk product ingestion
        - Processes new catalog items nightly
        - Writes predictions + confidence scores to data warehouse
        - Flags low-confidence predictions for human review

        ### Tier 3: Model Registry & Monitoring
        - **MLflow** for experiment tracking and model versioning
        - A/B testing framework for model updates via feature flags
        - Data drift detection on incoming product name distributions
        - Prediction drift monitoring (category distribution shifts)

        ---

        ## Deployment Flow

        ```
        [Product Catalog] --> [Batch Pipeline]  --> [Data Warehouse]
                                    |                     |
                          [MLflow Registry]         [Analytics]
                                    |
        [Client App] -----> [FastAPI Service] --> [Response Cache]
                                    |
                          [Monitoring/Alerts]
        ```

        ---

        ## Scaling Considerations

        | Scenario | Approach |
        |----------|----------|
        | < 100 req/s | Single container, Cloud Run |
        | 100-1000 req/s | Horizontal pod autoscaling on K8s |
        | > 1000 req/s | Add Redis prediction cache (TTL: 24h) |
        | New product types | Retrain trigger when drift score > threshold |

        ---

        ## Hybrid ML + LLM Strategy

        For items where the trained model confidence is below a threshold (e.g., 0.6),
        we fall back to an **LLM prompt-based classifier**:

        ```python
        if prediction.confidence < CONFIDENCE_THRESHOLD:
            # Fall back to LLM
            category = llm_classify(item_name, taxonomy_tree)
        ```

        This gives us the cost efficiency of the ML model for 80-90% of items,
        with the generalization capability of the LLM for edge cases.

        ---

        ## CI/CD Pipeline

        1. **Code push** triggers GitHub Actions
        2. Run unit tests + integration tests
        3. Train model on latest taxonomy data
        4. Compare CV F1 against production model (promotion gate)
        5. Build Docker image, push to registry
        6. Deploy to staging, run smoke tests
        7. Blue/green deployment to production
        8. Monitor prediction distribution for 24h before full rollout
        """
    )