"""
Prediction Script for Gene Classifier

Separate script for making predictions with trained models.
Supports loading specific model versions and batch prediction.

Author: Data Mining Project
License: MIT
"""

import argparse
import yaml
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from main import preprocess_data
from logger import setup_logger
from model_utils import load_model, list_model_versions


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def make_predictions(
    test_file: str,
    output_file: str,
    config_path: str = "config.yaml",
    model_version: str = None
):
    """
    Make predictions on test data using trained model.

    Args:
        test_file: Path to test data CSV
        output_file: Path to save predictions
        config_path: Path to configuration file
        model_version: Specific model version to use (latest if None)
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
    logger.info("Gene Function Classification - Prediction Pipeline")
    logger.info("=" * 60)

    try:
        # Step 1: Load model
        logger.info("\n[1/4] Loading trained model...")
        model_dir = config['model_persistence']['model_dir']

        if model_version:
            logger.info(f"  Loading model version: {model_version}")
        else:
            logger.info("  Loading latest model version")

        model, preprocessors, metadata = load_model(
            model_dir=model_dir,
            version=model_version
        )

        logger.info(f"  Model version: {metadata['version']}")
        logger.info(f"  Model accuracy: {metadata.get('accuracy', 'N/A')}")
        logger.info(f"  Model F1-score: {metadata.get('f1_score', 'N/A')}")

        # Step 2: Load test data
        logger.info(f"\n[2/4] Loading test data from {test_file}...")
        test_data = pd.read_csv(test_file)
        logger.info(f"  Test data: {test_data.shape[0]} samples, {test_data.shape[1]} features")

        # Step 3: Preprocess test data
        logger.info("\n[3/4] Preprocessing test data...")

        # Reconstruct preprocessing config from metadata
        from main import get_final_preprocessing
        preprocessing_config = get_final_preprocessing()

        test_data_processed, _ = preprocess_data(
            test_data,
            preprocessing_config,
            preprocessors
        )
        logger.info("  Preprocessing completed")

        # Step 4: Make predictions
        logger.info("\n[4/4] Generating predictions...")
        predictions = model.predict(test_data_processed)
        logger.info(f"  Predictions generated for {len(predictions)} samples")

        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(test_data_processed)
            logger.info("  Prediction probabilities calculated")

        # Step 5: Save predictions
        logger.info(f"\nSaving predictions to {output_file}...")

        # Save in specified format
        with open(output_file, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(f"{int(pred)},\n")

            # Add model metadata
            accuracy = metadata.get('accuracy', 0)
            f1_score = metadata.get('f1_score', 0)
            f.write(f"{accuracy},{f1_score},\n")

        logger.info(f"  Predictions saved successfully")

        # Optional: Save detailed predictions with probabilities
        if hasattr(model, 'predict_proba'):
            detailed_file = output_file.replace('.txt', '_detailed.csv')
            detailed_df = pd.DataFrame({
                'prediction': predictions,
                'probability_class_0': probabilities[:, 0],
                'probability_class_1': probabilities[:, 1]
            })
            detailed_df.to_csv(detailed_file, index=False)
            logger.info(f"  Detailed predictions saved to {detailed_file}")

        logger.info("\n" + "=" * 60)
        logger.info("Prediction completed successfully!")
        logger.info("=" * 60)

        # Print summary statistics
        logger.info("\nPrediction Summary:")
        logger.info(f"  Class 0: {np.sum(predictions == 0)} samples")
        logger.info(f"  Class 1: {np.sum(predictions == 1)} samples")
        logger.info(f"  Ratio: {np.sum(predictions == 1) / len(predictions):.2%}")

        return predictions

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise


def main():
    """Main entry point for prediction script."""
    parser = argparse.ArgumentParser(
        description="Make predictions with gene classifier model"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test data CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.txt",
        help="Path to save predictions"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Specific model version to use (uses latest if not specified)"
    )
    parser.add_argument(
        "--list-versions",
        action="store_true",
        help="List available model versions and exit"
    )

    args = parser.parse_args()

    # Handle list versions
    if args.list_versions:
        config = load_config(args.config)
        model_dir = config['model_persistence']['model_dir']
        versions = list_model_versions(model_dir)

        if versions:
            print("Available model versions:")
            for version in versions:
                print(f"  - {version}")
        else:
            print("No model versions found.")
        sys.exit(0)

    # Make predictions
    try:
        make_predictions(
            args.test_file,
            args.output,
            args.config,
            args.model_version
        )
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
