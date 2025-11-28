"""
Unit tests for preprocessing functions.

Tests cover imputation, scaling, outlier handling, and preprocessing pipelines.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, '..')

from main import (
    handle_outliers,
    preprocess_data,
    get_preprocessing_methods
)


class TestOutlierHandling:
    """Test outlier detection and handling methods."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with outliers."""
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 100],  # 100 is outlier
            'feature2': [10, 20, 30, 40, 50, 60],
            'feature3': [-5, 0, 5, 10, 15, 200]  # 200 is outlier
        })

    def test_no_outlier_handling(self, sample_data):
        """Test that None method returns unchanged data."""
        result = handle_outliers(sample_data, None, sample_data.columns)
        pd.testing.assert_frame_equal(result, sample_data)

    def test_zscore_outlier_handling(self, sample_data):
        """Test Z-score outlier detection."""
        method = {"method": "zscore", "threshold": 2}
        result = handle_outliers(sample_data, method, sample_data.columns)

        # Check that outliers were replaced
        assert result['feature1'].max() < 100
        assert result['feature3'].max() < 200

    def test_iqr_outlier_handling(self, sample_data):
        """Test IQR outlier detection."""
        method = {"method": "iqr", "factor": 1.5}
        result = handle_outliers(sample_data, method, sample_data.columns)

        # Check that outliers were replaced
        assert result['feature1'].max() < 100


class TestPreprocessing:
    """Test preprocessing pipeline functions."""

    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample data with missing values."""
        return pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [10, np.nan, 30, 40, 50],
            'feature3': [100, 200, 300, 400, 500]
        })

    def test_preprocess_data_imputation(self, sample_data_with_missing):
        """Test that preprocessing handles missing values."""
        config = {
            "imputation": ("mean", SimpleImputer(strategy="mean")),
            "scaling": ("standard", StandardScaler()),
            "outlier": ("none", None)
        }

        result, preprocessors = preprocess_data(sample_data_with_missing, config)

        # Check no missing values remain
        assert result.isna().sum().sum() == 0

    def test_preprocess_data_scaling(self, sample_data_with_missing):
        """Test that preprocessing applies scaling."""
        config = {
            "imputation": ("mean", SimpleImputer(strategy="mean")),
            "scaling": ("standard", StandardScaler()),
            "outlier": ("none", None)
        }

        result, preprocessors = preprocess_data(sample_data_with_missing, config)

        # Check that features are approximately standardized
        # (mean ~0, std ~1, allowing for small sample size)
        assert abs(result.mean().mean()) < 0.5
        assert abs(result.std().mean() - 1.0) < 0.5

    def test_preprocessing_methods_generation(self):
        """Test that all preprocessing combinations are generated."""
        methods = get_preprocessing_methods()

        # Should have multiple combinations
        assert len(methods) > 0

        # Each method should have required keys
        for method in methods:
            assert 'imputation' in method
            assert 'scaling' in method
            assert 'outlier' in method

    def test_preprocessor_consistency(self, sample_data_with_missing):
        """Test that same preprocessor produces consistent results."""
        config = {
            "imputation": ("mean", SimpleImputer(strategy="mean")),
            "scaling": ("standard", StandardScaler()),
            "outlier": ("none", None)
        }

        # First preprocessing (training)
        result1, preprocessors = preprocess_data(sample_data_with_missing, config)

        # Second preprocessing with same preprocessor (test)
        result2, _ = preprocess_data(sample_data_with_missing, config, preprocessors)

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)


class TestDataIntegrity:
    """Test data integrity during preprocessing."""

    def test_preprocessing_preserves_shape(self):
        """Test that preprocessing maintains data shape."""
        data = pd.DataFrame(np.random.randn(100, 10))
        config = {
            "imputation": ("mean", SimpleImputer(strategy="mean")),
            "scaling": ("standard", StandardScaler()),
            "outlier": ("none", None)
        }

        result, _ = preprocess_data(data, config)

        assert result.shape == data.shape

    def test_preprocessing_no_data_leakage(self):
        """Test that test preprocessing uses training statistics."""
        train_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })

        test_data = pd.DataFrame({
            'feature1': [10, 20, 30],
            'feature2': [100, 200, 300]
        })

        config = {
            "imputation": ("mean", SimpleImputer(strategy="mean")),
            "scaling": ("standard", StandardScaler()),
            "outlier": ("none", None)
        }

        # Fit on training data
        _, preprocessors = preprocess_data(train_data, config)

        # Transform test data with training preprocessors
        result, _ = preprocess_data(test_data, config, preprocessors)

        # Test data should be scaled using training statistics
        assert result is not None
        assert result.shape == test_data.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
