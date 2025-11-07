"""
Gene Function Classification Pipeline

This module implements a machine learning pipeline for binary classification
of gene functions, specifically predicting whether genes have cell communication
capabilities. It includes automated preprocessing optimization, ensemble learning,
and comprehensive model evaluation.

Author: Data Mining Project
Date: 2024
License: MIT
"""

import random
import sys
from typing import Dict, Tuple, Optional, Any, List

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_iterative_imputer

from constants import model_config, pre_processing_config, final_preprocessing_config

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)


def get_preprocessing_methods() -> List[Dict[str, Tuple]]:
    """
    Generate all possible preprocessing method combinations.

    Combines different imputation, scaling, and outlier handling methods
    from the configuration to create a comprehensive set of preprocessing
    pipelines for optimization.

    Returns:
        List[Dict[str, Tuple]]: List of dictionaries, each containing a unique
            combination of preprocessing methods with keys:
            - 'imputation': (name, imputer_instance)
            - 'scaling': (name, scaler_instance)
            - 'outlier': (name, outlier_config)

    Example:
        >>> methods = get_preprocessing_methods()
        >>> len(methods)
        36  # 3 imputation × 4 scaling × 3 outlier methods
    """
    # Fetch the preprocessing methods from config
    imputation = pre_processing_config["imputation_methods"]
    scaling = pre_processing_config["scaling_methods"]
    outlier = pre_processing_config["outlier_methods"]

    preprocessing_methods = []
    for imp in imputation.items():
        for scale in scaling.items():
            for out in outlier.items():
                preprocessing_methods.append(
                    {"imputation": imp, "scaling": scale, "outlier": out}
                )

    return preprocessing_methods


def handle_outliers(
    data: pd.DataFrame,
    method: Optional[Dict[str, Any]],
    num_columns: pd.Index
) -> pd.DataFrame:
    """
    Detect and handle outliers in numerical columns.

    Implements two outlier detection methods:
    1. Z-score: Identifies values beyond a threshold of standard deviations
    2. IQR (Interquartile Range): Identifies values outside whisker range

    Detected outliers are replaced with the column median.

    Args:
        data: Input DataFrame containing the data
        method: Dictionary with outlier detection config containing:
            - 'method': Either 'zscore' or 'iqr'
            - 'threshold': Z-score threshold (for zscore method)
            - 'factor': IQR multiplier factor (for iqr method)
            Set to None to skip outlier handling
        num_columns: Index of numerical column names to process

    Returns:
        pd.DataFrame: Copy of data with outliers replaced by median values

    Examples:
        >>> method = {"method": "zscore", "threshold": 3}
        >>> cleaned_data = handle_outliers(data, method, numeric_cols)

        >>> method = {"method": "iqr", "factor": 1.5}
        >>> cleaned_data = handle_outliers(data, method, numeric_cols)
    """
    if method is None:
        return data

    data_copy = data.copy()

    if method["method"] == "zscore":
        # Z-score method: |z| > threshold
        for col in num_columns:
            z_scores = np.abs(
                (data_copy[col] - data_copy[col].mean()) / data_copy[col].std()
            )
            outliers = z_scores > method["threshold"]
            data_copy.loc[outliers, col] = data_copy[col].median()

    elif method["method"] == "iqr":
        # IQR method: Q1 - factor*IQR, Q3 + factor*IQR
        for col in num_columns:
            q1 = data_copy[col].quantile(0.25)
            q3 = data_copy[col].quantile(0.75)
            iqr_value = q3 - q1

            lower_bound = q1 - method["factor"] * iqr_value
            upper_bound = q3 + method["factor"] * iqr_value

            outliers = (data_copy[col] < lower_bound) | (data_copy[col] > upper_bound)
            data_copy.loc[outliers, col] = data_copy[col].median()

    return data_copy


def preprocess_data(
    data: pd.DataFrame,
    preprocessing_config: Dict[str, Tuple],
    preprocessors: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply comprehensive preprocessing pipeline to the data.

    Performs the following steps in order:
    1. Imputation of missing values (numeric features)
    2. Mode imputation for nominal/categorical features
    3. Outlier detection and handling
    4. Feature scaling/normalization

    Args:
        data: Input DataFrame to preprocess
        preprocessing_config: Dictionary containing preprocessing specifications:
            - 'imputation': (name, imputer_instance)
            - 'scaling': (name, scaler_instance or None)
            - 'outlier': (name, outlier_config_dict or None)
        preprocessors: Optional dictionary of fitted preprocessors for test data.
            If None, fits new preprocessors (training mode).
            If provided, uses existing preprocessors (test mode).

    Returns:
        Tuple containing:
            - pd.DataFrame: Preprocessed data
            - Dict[str, Any]: Dictionary of fitted preprocessors with keys:
                - 'numeric_imputer': Fitted imputer for numeric features
                - 'scaler': Fitted scaler (if scaling is applied)

    Example:
        >>> # Training mode
        >>> X_train_processed, fitted_prep = preprocess_data(
        ...     X_train, config, preprocessors=None
        ... )
        >>> # Test mode
        >>> X_test_processed, _ = preprocess_data(
        ...     X_test, config, preprocessors=fitted_prep
        ... )
    """
    data_copy = data.copy()

    num_columns = data_copy.select_dtypes(include=[np.number]).columns
    nominal_columns = data_copy.select_dtypes(include=["object"]).columns

    # Initialize preprocessors dict if not provided (training mode)
    if preprocessors is None:
        preprocessors = {}
        is_training = True
    else:
        is_training = False

    # Step 1: Handle numeric features - imputation
    _, imputer = preprocessing_config["imputation"]
    if is_training:
        preprocessors["numeric_imputer"] = imputer.fit(data_copy[num_columns])
    data_copy[num_columns] = preprocessors["numeric_imputer"].transform(
        data_copy[num_columns]
    )

    # Step 2: Handle nominal/categorical features - mode imputation
    for col in nominal_columns:
        if len(data_copy[col].mode()) > 0:
            data_copy[col] = data_copy[col].fillna(data_copy[col].mode()[0])

    # Step 3: Outlier handling (only in training mode)
    outlier_config = preprocessing_config["outlier"][1]
    if is_training:
        data_copy = handle_outliers(data_copy, outlier_config, num_columns)

    # Step 4: Feature scaling
    _, scaler = preprocessing_config["scaling"]
    if scaler is not None:
        if is_training:
            preprocessors["scaler"] = scaler.fit(data_copy[num_columns])
        data_copy[num_columns] = preprocessors["scaler"].transform(
            data_copy[num_columns]
        )

    return data_copy, preprocessors


def optimize_preprocessing(
    x_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[Dict[str, Tuple], Dict[str, Any]]:
    """
    Find optimal preprocessing configuration using cross-validation.

    Tests all possible combinations of imputation, scaling, and outlier handling
    methods to identify the configuration that maximizes F1-score.

    Args:
        x_train: Training features DataFrame
        y_train: Training labels Series

    Returns:
        Tuple containing:
            - Dict: Best preprocessing configuration
            - Dict: Fitted preprocessors for the best configuration

    Notes:
        - Uses Random Forest as evaluation model
        - Employs 5-fold cross-validation
        - Optimizes for F1-score (suitable for imbalanced data)
        - Silently skips configurations that cause errors

    Example:
        >>> best_config, fitted_prep = optimize_preprocessing(X_train, y_train)
        >>> print(f"Best F1-score: {best_score:.3f}")
    """
    # Initialize variables to save best preprocessing
    best_score = 0
    best_preprocessing = None
    best_preprocessors = None

    # Initialize 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluate all preprocessing combinations
    for prep_config in get_preprocessing_methods():
        try:
            # Preprocess data with current configuration
            x_processed, fitted_preprocessors = preprocess_data(
                x_train, prep_config
            )

            # Evaluate with Random Forest baseline
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            scores = cross_val_score(
                model, x_processed, y_train, cv=kf, scoring="f1"
            )
            avg_score = np.mean(scores)

            # Update best configuration if improved
            if avg_score > best_score:
                best_score = avg_score
                best_preprocessing = prep_config
                best_preprocessors = fitted_preprocessors

        except Exception as e:
            # Skip configurations that cause errors
            print(f"Warning: Preprocessing configuration failed - {e}")
            continue

    print(f"Best preprocessing F1-score: {best_score:.4f}")
    return best_preprocessing, best_preprocessors


def get_models_config() -> Dict[str, Dict[str, Any]]:
    """
    Retrieve model configurations from constants.

    Returns:
        Dict: Model configuration dictionary with model instances and
            hyperparameter grids for grid search
    """
    return model_config


def train_and_evaluate(x_train, y_train):
    '''
        Method to train the model and find the best model
    '''
    # Optimize preprocessing
    best_preprocessing, fitted_preprocessors = optimize_preprocessing(
        x_train, y_train)
    # Preprocess data
    x_train_processed, _ = preprocess_data(
        x_train, best_preprocessing, fitted_preprocessors
    )

    best_model = None
    best_params = None
    best_f1_score = 0
    final_accuracy = 0
    f1_final = 0

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Loop through all the models and find the best model
    for model_name, model_info in get_models_config().items():
        model = model_info["model"]
        print("Training model: ", model_name)
        # check if the model has params
        if model_info["params"]:
            # Grid search to find the best params
            grid_search = GridSearchCV(
                model, model_info["params"], cv=kf, scoring="f1", n_jobs=-1
            )
            grid_search.fit(x_train_processed, y_train)
            model = grid_search.best_estimator_
            current_params = grid_search.best_params_
        else:
            model.fit(x_train_processed, y_train)
            current_params = {}
        # Cross-validation and calculate the f1 score, accuracy
        accuracy_scores = cross_val_score(
            model, x_train_processed, y_train, cv=kf, scoring="accuracy"
        )
        f1_scores = cross_val_score(
            model, x_train_processed, y_train, cv=kf, scoring="f1"
        )
        avg_f1_score = np.mean(f1_scores)
        # check if the average f1 score is greater than the best f1 score
        if avg_f1_score > best_f1_score:
            best_f1_score = avg_f1_score
            best_model = model
            best_params = current_params
            final_accuracy = round(float(np.mean(accuracy_scores)), 3)
            f1_final = round(float(avg_f1_score), 3)

    # Final training on full dataset
    best_model.fit(x_train_processed, y_train)

    print(f"Best Model: {best_model}")


    return (
        best_model,
        best_params,
        best_preprocessing,
        fitted_preprocessors,
        final_accuracy,
        f1_final,
    )


def get_final_model() -> VotingClassifier:
    """
    Create the final ensemble model with optimized hyperparameters.

    Combines K-Nearest Neighbors and Random Forest classifiers using
    soft voting (probability averaging) for improved performance on
    the gene function classification task.

    Returns:
        VotingClassifier: Ensemble model with the following components:
            - KNN: n_neighbors=8, euclidean distance, distance weighting
            - Random Forest: 100 trees, max_depth=10, tuned split parameters

    Notes:
        These models and hyperparameters were selected based on
        cross-validation performance during model development.

    Example:
        >>> model = get_final_model()
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    # K-Nearest Neighbors configuration
    knn = KNeighborsClassifier(
        n_neighbors=8,
        metric="euclidean",
        weights="distance"
    )

    # Random Forest configuration
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
    )

    # Ensemble with soft voting (probability averaging)
    return VotingClassifier(
        estimators=[("knn", knn), ("rf", rf)],
        voting="soft"
    )


def get_final_preprocessing() -> Dict[str, Tuple]:
    """
    Retrieve the final preprocessing configuration.

    Returns the pre-determined optimal preprocessing configuration
    based on prior experimentation and validation.

    Returns:
        Dict[str, Tuple]: Preprocessing configuration with:
            - 'imputation': Mean imputation for numeric features
            - 'scaling': StandardScaler normalization
            - 'outlier': No outlier handling

    Example:
        >>> config = get_final_preprocessing()
        >>> X_processed, preprocessors = preprocess_data(X, config)
    """
    return final_preprocessing_config


def main() -> None:
    """
    Main execution pipeline for gene function classification.

    Workflow:
        1. Load training and test data from CSV files
        2. Split training data into features and labels
        3. Apply preprocessing pipeline (imputation, scaling)
        4. Train ensemble model (KNN + Random Forest)
        5. Evaluate model performance using 5-fold cross-validation
        6. Generate predictions on test data
        7. Save predictions and metrics to output file

    Outputs:
        - predictions.txt: File containing test predictions and performance metrics

    Raises:
        FileNotFoundError: If data files are missing
        SystemExit: If critical errors occur during execution

    Notes:
        - Uses pre-configured optimal preprocessing and model settings
        - For full preprocessing optimization, uncomment lines 450-452
    """
    print("=" * 60)
    print("Gene Function Classification Pipeline")
    print("=" * 60)

    # Step 1: Load data files
    print("\n[1/7] Loading data files...")
    try:
        train_data = pd.read_csv("DM_project_24.csv")
        test_data = pd.read_csv("test_data.csv")
        print(f"  ✓ Training data: {train_data.shape[0]} samples, {train_data.shape[1]} features")
        print(f"  ✓ Test data: {test_data.shape[0]} samples, {test_data.shape[1]} features")
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}")
        print("  Please ensure DM_project_24.csv and test_data.csv are in the directory")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ Unexpected error occurred: {e}")
        sys.exit(1)

    # Step 2: Split features and target
    print("\n[2/7] Splitting features and target...")
    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    print(f"  ✓ Features: {x_train.shape[1]} columns")
    print(f"  ✓ Target distribution: {dict(y_train.value_counts())}")

    # Step 3: Get preprocessing configuration and model
    print("\n[3/7] Initializing preprocessing and model...")
    best_preprocessing = get_final_preprocessing()
    best_model = get_final_model()
    print("  ✓ Preprocessing: Mean imputation + StandardScaler")
    print("  ✓ Model: Ensemble (KNN + Random Forest)")

    # Step 4: Preprocess training data
    print("\n[4/7] Preprocessing training data...")
    x_train_processed, fitted_preprocessors = preprocess_data(
        x_train, best_preprocessing
    )
    print("  ✓ Imputation and scaling completed")

    # Step 5: Train model
    print("\n[5/7] Training ensemble model...")
    best_model.fit(x_train_processed, y_train)
    print("  ✓ Model training completed")

    # Step 6: Cross-validation evaluation
    print("\n[6/7] Evaluating model performance (5-fold CV)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    accuracy_scores = cross_val_score(
        best_model, x_train_processed, y_train, cv=kf, scoring="accuracy"
    )
    f1_scores = cross_val_score(
        best_model, x_train_processed, y_train, cv=kf, scoring="f1"
    )

    model_accuracy = round(float(np.mean(accuracy_scores)), 3)
    f1 = round(float(np.mean(f1_scores)), 3)

    print(f"  ✓ Accuracy: {model_accuracy:.3f} (±{np.std(accuracy_scores):.3f})")
    print(f"  ✓ F1-Score: {f1:.3f} (±{np.std(f1_scores):.3f})")

    # Step 7: Generate test predictions
    print("\n[7/7] Generating test predictions...")
    x_test_data, _ = preprocess_data(
        test_data, best_preprocessing, fitted_preprocessors
    )
    test_predictions = best_model.predict(x_test_data)
    print(f"  ✓ Predictions generated for {len(test_predictions)} samples")

    # Alternative: Full preprocessing optimization and model selection
    # Uncomment the following lines for comprehensive experimentation:
    # best_preprocessing, fitted_preprocessors = optimize_preprocessing(x_train, y_train)
    # best_model, _, _, _, model_accuracy, f1 = train_and_evaluate(x_train, y_train)
    # test_predictions = best_model.predict(x_test_data)

    # Save results to file
    output_file = "predictions.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in test_predictions:
            f.write(f"{int(pred)},\n")
        f.write(f"{model_accuracy},{f1},\n")

    print(f"\n✓ Results saved to '{output_file}'")
    print("=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
