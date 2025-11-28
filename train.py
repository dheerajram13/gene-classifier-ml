"""
Training Script for Gene Classifier

Separate script for model training with experiment tracking and model persistence.
Follows ML best practices for reproducibility and versioning.

Author: Data Mining Project
License: MIT
"""

import argparse
import yaml
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold

from main import (
    preprocess_data,
    get_final_preprocessing,
    get_final_model
)
from logger import setup_logger
from model_utils import save_model
from experiment_tracker import ExperimentTracker


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_model(config_path: str = "config.yaml", experiment_name: str = None):
    """
    Train gene classifier model with tracking and persistence.

    Args:
        config_path: Path to configuration file
        experiment_name: Optional name for experiment run
    """
    # Load configuration
    config = load_config(config_path)

    # Setup logger
    logger = setup_logger(
        level=config['logging']['level'],
        log_file=config['logging']['file'],
        console=config['logging']['console']
    )

    logger.info("=" * 60)
    logger.info("Gene Function Classification - Training Pipeline")
    logger.info("=" * 60)

    # Initialize experiment tracker
    tracker = None
    if config['experiment']['tracking_enabled']:
        tracker = ExperimentTracker(config['experiment']['tracking_uri'])
        run_name = experiment_name or None
        tracker.start_run(run_name)
        logger.info(f"Experiment tracking enabled: {tracker.current_run}")

    try:
        # Step 1: Load training data
        logger.info("\n[1/6] Loading training data...")
        train_file = config['data']['train_file']
        train_data = pd.read_csv(train_file)
        logger.info(f"  Training data: {train_data.shape[0]} samples, {train_data.shape[1]} features")

        # Step 2: Split features and target
        logger.info("\n[2/6] Splitting features and target...")
        x_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]
        logger.info(f"  Features: {x_train.shape[1]} columns")
        logger.info(f"  Target distribution: {dict(y_train.value_counts())}")

        # Step 3: Get preprocessing configuration
        logger.info("\n[3/6] Initializing preprocessing...")
        preprocessing_config = get_final_preprocessing()
        logger.info(f"  Imputation: {preprocessing_config['imputation'][0]}")
        logger.info(f"  Scaling: {preprocessing_config['scaling'][0]}")
        logger.info(f"  Outlier handling: {preprocessing_config['outlier'][0]}")

        # Log preprocessing parameters
        if tracker:
            tracker.log_params({
                "imputation_method": preprocessing_config['imputation'][0],
                "scaling_method": preprocessing_config['scaling'][0],
                "outlier_method": preprocessing_config['outlier'][0],
                "n_samples": x_train.shape[0],
                "n_features": x_train.shape[1]
            })

        # Step 4: Preprocess data
        logger.info("\n[4/6] Preprocessing training data...")
        x_train_processed, fitted_preprocessors = preprocess_data(
            x_train, preprocessing_config
        )
        logger.info("  Preprocessing completed")

        # Step 5: Initialize and configure model
        logger.info("\n[5/6] Training model...")
        model = get_final_model()

        # Log model parameters
        if tracker:
            model_params = {}
            for name, estimator in model.estimators:
                params = estimator.get_params()
                model_params[f"{name}_params"] = str(params)
            tracker.log_params(model_params)

        # Train model
        model.fit(x_train_processed, y_train)
        logger.info("  Model training completed")

        # Step 6: Cross-validation evaluation
        logger.info("\n[6/6] Evaluating model performance (CV)...")
        cv_config = config['cross_validation']
        kf = KFold(
            n_splits=cv_config['n_splits'],
            shuffle=cv_config['shuffle'],
            random_state=cv_config['random_state']
        )

        # Calculate metrics
        accuracy_scores = cross_val_score(
            model, x_train_processed, y_train,
            cv=kf, scoring="accuracy"
        )
        f1_scores = cross_val_score(
            model, x_train_processed, y_train,
            cv=kf, scoring="f1"
        )

        accuracy_mean = float(np.mean(accuracy_scores))
        accuracy_std = float(np.std(accuracy_scores))
        f1_mean = float(np.mean(f1_scores))
        f1_std = float(np.std(f1_scores))

        logger.info(f"  Accuracy: {accuracy_mean:.4f} (±{accuracy_std:.4f})")
        logger.info(f"  F1-Score: {f1_mean:.4f} (±{f1_std:.4f})")

        # Log metrics
        if tracker:
            tracker.log_metrics({
                "cv_accuracy_mean": accuracy_mean,
                "cv_accuracy_std": accuracy_std,
                "cv_f1_mean": f1_mean,
                "cv_f1_std": f1_std
            })

        # Save model
        if config['model_persistence']['save_model']:
            logger.info("\nSaving model...")
            metadata = {
                "accuracy": round(accuracy_mean, 4),
                "f1_score": round(f1_mean, 4),
                "accuracy_std": round(accuracy_std, 4),
                "f1_std": round(f1_std, 4),
                "n_samples": x_train.shape[0],
                "n_features": x_train.shape[1]
            }

            version_dir = save_model(
                model,
                fitted_preprocessors,
                model_dir=config['model_persistence']['model_dir'],
                metadata=metadata
            )
            logger.info(f"  Model saved to: {version_dir}")

            if tracker:
                tracker.log_artifact(version_dir, "Trained model")

        # End experiment
        if tracker:
            tracker.end_run(status="completed")
            logger.info(f"\nExperiment tracking completed")

        logger.info("\n" + "=" * 60)
        logger.info("Training completed successfully!")
        logger.info("=" * 60)

        return model, fitted_preprocessors, metadata

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        if tracker:
            tracker.log_metadata({"error": str(e)})
            tracker.end_run(status="failed")
        raise


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train gene classifier model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for experiment run"
    )

    args = parser.parse_args()

    try:
        train_model(args.config, args.experiment_name)
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
