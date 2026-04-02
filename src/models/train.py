"""
Training Pipeline
=================
Orchestrates the full model training workflow with MLflow tracking.

Usage:
    python -m src.models.train
    python -m src.models.train --config configs/model_config.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_config, load_model_config
from src.data.preprocessor import run_preprocessing_pipeline
from src.models.classifier import (
    train_classifier,
    predict_categories,
    save_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def run_training_pipeline(
    data_config_path: str = "configs/data_config.yaml",
    model_config_path: str = "configs/model_config.yaml",
    skip_preprocessing: bool = False,
):
    """
    Execute the full training pipeline:
    1. Preprocess data
    2. Train classifier with MLflow tracking
    3. Classify unmatched sales items
    4. Save artifacts

    Parameters
    ----------
    data_config_path : str
    model_config_path : str
    skip_preprocessing : bool
        If True, load pre-processed data instead of re-running pipeline
    """
    data_config = load_config(data_config_path)
    model_config = load_model_config(model_config_path)

    # ------------------------------------------------------------------
    # Step 1: Data Preprocessing
    # ------------------------------------------------------------------
    if skip_preprocessing:
        logger.info("Skipping preprocessing, loading existing processed data...")
        taxonomy_df = pd.read_parquet(data_config["paths"]["processed_taxonomy"])
        merged_df = pd.read_parquet(data_config["paths"]["merged"])
    else:
        logger.info("Running preprocessing pipeline...")
        merged_df = run_preprocessing_pipeline(
            sales_path=data_config["paths"]["raw_sales"],
            taxonomy_path=data_config["paths"]["raw_taxonomy"],
            config=data_config,
        )
        taxonomy_df = pd.read_parquet(data_config["paths"]["processed_taxonomy"])

    # ------------------------------------------------------------------
    # Step 2: Train Classifier with MLflow
    # ------------------------------------------------------------------
    mlflow_config = model_config["mlflow"]

    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
        mlflow.set_experiment(mlflow_config["experiment_name"])

        with mlflow.start_run(run_name="taxonomy_classifier_v1") as run:
            logger.info(f"MLflow run: {run.info.run_id}")

            # Log parameters
            mlflow.log_params(
                {
                    "vectorizer_type": model_config["classifier"]["vectorizer"]["type"],
                    "max_features": model_config["classifier"]["vectorizer"]["max_features"],
                    "ngram_range": str(model_config["classifier"]["vectorizer"]["ngram_range"]),
                    "n_estimators": model_config["classifier"]["lgbm"]["n_estimators"],
                    "learning_rate": model_config["classifier"]["lgbm"]["learning_rate"],
                    "cv_splits": model_config["classifier"]["cv"]["n_splits"],
                }
            )

            # Train
            results = train_classifier(taxonomy_df, model_config)

            # Log metrics
            mlflow.log_metrics(
                {
                    "cv_f1_mean": results["cv_f1_mean"],
                    "cv_f1_std": results["cv_f1_std"],
                    "n_classes": results["n_classes"],
                    "n_samples": results["n_samples"],
                }
            )

            # Log model
            mlflow.sklearn.log_model(
                results["pipeline"],
                "classifier_pipeline",
                registered_model_name="product-taxonomy-classifier",
            )

            logger.info(
                f"Training complete. CV F1: {results['cv_f1_mean']:.3f} "
                f"+/- {results['cv_f1_std']:.3f}"
            )

    except ImportError:
        logger.warning("MLflow not installed. Training without experiment tracking.")
        results = train_classifier(taxonomy_df, model_config)
        logger.info(
            f"Training complete. CV F1: {results['cv_f1_mean']:.3f} "
            f"+/- {results['cv_f1_std']:.3f}"
        )

    # ------------------------------------------------------------------
    # Step 3: Save Model Artifacts
    # ------------------------------------------------------------------
    model_path = data_config["paths"]["model_artifacts"] + "classifier_pipeline.joblib"
    save_model(results["pipeline"], results["label_encoder"], model_path)

    # ------------------------------------------------------------------
    # Step 4: Classify Unmatched Sales Items
    # ------------------------------------------------------------------
    unmatched = merged_df[merged_df["category_target"].isna()].copy()
    if len(unmatched) > 0:
        logger.info(f"Classifying {len(unmatched):,} unmatched sales items...")
        predictions = predict_categories(
            results["pipeline"],
            results["label_encoder"],
            unmatched["item_name"],
        )
        predictions.to_parquet("data/processed/classified_items.parquet", index=False)
        logger.info(
            f"Classification complete. "
            f"Mean confidence: {predictions['confidence'].mean():.3f}"
        )

    # ------------------------------------------------------------------
    # Step 5: Print Summary Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Samples:        {results['n_samples']}")
    print(f"Classes:        {results['n_classes']}")
    print(f"CV F1 (macro):  {results['cv_f1_mean']:.3f} +/- {results['cv_f1_std']:.3f}")
    print(f"Model saved:    {model_path}")
    if len(unmatched) > 0:
        print(f"Items classified: {len(unmatched):,}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train product taxonomy classifier")
    parser.add_argument(
        "--data-config", default="configs/data_config.yaml", help="Data config path"
    )
    parser.add_argument(
        "--model-config", default="configs/model_config.yaml", help="Model config path"
    )
    parser.add_argument(
        "--skip-preprocessing", action="store_true", help="Skip data preprocessing"
    )
    args = parser.parse_args()

    run_training_pipeline(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        skip_preprocessing=args.skip_preprocessing,
    )
