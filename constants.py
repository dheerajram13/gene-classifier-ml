"""
Configuration Module for Gene Function Classification

This module defines all model configurations, preprocessing methods, and
hyperparameter grids used in the machine learning pipeline.

Configuration Categories:
    - model_config: Classifier configurations with hyperparameter grids
    - pre_processing_config: Available preprocessing methods
    - final_preprocessing_config: Optimized preprocessing pipeline

Author: Data Mining Project
License: MIT
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# =============================================================================
# Model Configurations
# =============================================================================

model_config = {
    "decision_tree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 2, 5, 10, 15, 20],
        },
    },
    "random_forest": {
        "model": RandomForestClassifier(),
        "params": {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 2, 5, 10, 15, 20],
            "n_estimators": [100, 200],
        },
    },
    "knn": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": range(1, 18),
            "metric": ["euclidean", "manhattan"]
        },
    },
    "naive_bayes": {
        "model": GaussianNB(),
        "params": {}
    },
}

# =============================================================================
# Preprocessing Configurations
# =============================================================================

pre_processing_config = {
    # Missing value imputation methods
    "imputation_methods": {
        "median": SimpleImputer(strategy="median"),
        "mean": SimpleImputer(strategy="mean"),
        "iterative": IterativeImputer(random_state=42),
    },
    # Feature scaling methods
    "scaling_methods": {
        "standard": StandardScaler(),  # Zero mean, unit variance
        "minmax": MinMaxScaler(),      # Scale to [0, 1] range
        "robust": RobustScaler(),      # Uses median and IQR
        "none": None,                  # No scaling
    },
    # Outlier detection and handling methods
    "outlier_methods": {
        "zscore": {"method": "zscore", "threshold": 3},      # ±3 std devs
        "iqr": {"method": "iqr", "factor": 1.5},             # 1.5 × IQR
        "none": None,                                         # No outlier handling
    },
}

# =============================================================================
# Final Optimized Configuration
# =============================================================================

final_preprocessing_config = {
    "imputation": ("mean", SimpleImputer()),
    "scaling": ("standard", StandardScaler()),
    "outlier": ("none", None),
}
"""
The final configuration was selected based on cross-validation experiments:
- Mean imputation: Best balance of performance and simplicity
- StandardScaler: Effective normalization for distance-based algorithms
- No outlier removal: Preserves biological variability in gene data
"""
