"""
Unit tests for model utilities.

Tests cover model saving, loading, versioning, and comparison.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

import sys
sys.path.insert(0, '..')

from model_utils import (
    save_model,
    load_model,
    list_model_versions,
    compare_models
)


class TestModelPersistence:
    """Test model saving and loading functionality."""

    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for models."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_model(self):
        """Create sample model and preprocessors."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        preprocessors = {
            'numeric_imputer': SimpleImputer(strategy='mean'),
            'scaler': None
        }
        return model, preprocessors

    def test_save_model(self, temp_model_dir, sample_model):
        """Test model saving functionality."""
        model, preprocessors = sample_model
        metadata = {"accuracy": 0.89, "f1_score": 0.75}

        version_dir = save_model(
            model,
            preprocessors,
            model_dir=temp_model_dir,
            version="test_v1",
            metadata=metadata
        )

        # Check that directory was created
        assert Path(version_dir).exists()

        # Check that all files exist
        assert (Path(version_dir) / "model.joblib").exists()
        assert (Path(version_dir) / "preprocessors.joblib").exists()
        assert (Path(version_dir) / "metadata.json").exists()

    def test_load_model(self, temp_model_dir, sample_model):
        """Test model loading functionality."""
        model, preprocessors = sample_model
        metadata = {"accuracy": 0.89}

        # Save model first
        save_model(
            model,
            preprocessors,
            model_dir=temp_model_dir,
            version="test_v1",
            metadata=metadata
        )

        # Load model
        loaded_model, loaded_prep, loaded_meta = load_model(
            model_dir=temp_model_dir,
            version="test_v1"
        )

        # Verify loaded objects
        assert loaded_model is not None
        assert loaded_prep is not None
        assert loaded_meta["accuracy"] == 0.89
        assert loaded_meta["version"] == "test_v1"

    def test_load_latest_model(self, temp_model_dir, sample_model):
        """Test loading latest model version."""
        model, preprocessors = sample_model

        # Save multiple versions
        save_model(model, preprocessors, model_dir=temp_model_dir, version="v1")
        save_model(model, preprocessors, model_dir=temp_model_dir, version="v2")
        save_model(model, preprocessors, model_dir=temp_model_dir, version="v3")

        # Load latest (should be v3)
        _, _, metadata = load_model(model_dir=temp_model_dir)

        assert metadata["version"] == "v3"

    def test_list_model_versions(self, temp_model_dir, sample_model):
        """Test listing all model versions."""
        model, preprocessors = sample_model

        # Save multiple versions
        save_model(model, preprocessors, model_dir=temp_model_dir, version="v1")
        save_model(model, preprocessors, model_dir=temp_model_dir, version="v2")
        save_model(model, preprocessors, model_dir=temp_model_dir, version="v3")

        versions = list_model_versions(model_dir=temp_model_dir)

        assert len(versions) == 3
        assert "v1" in versions
        assert "v2" in versions
        assert "v3" in versions

    def test_compare_models(self, temp_model_dir, sample_model):
        """Test model comparison functionality."""
        model, preprocessors = sample_model

        # Save models with different metrics
        save_model(
            model, preprocessors,
            model_dir=temp_model_dir,
            version="v1",
            metadata={"accuracy": 0.85, "f1_score": 0.70}
        )
        save_model(
            model, preprocessors,
            model_dir=temp_model_dir,
            version="v2",
            metadata={"accuracy": 0.89, "f1_score": 0.75}
        )

        comparison = compare_models(model_dir=temp_model_dir)

        assert len(comparison) == 2
        assert comparison["v1"]["accuracy"] == 0.85
        assert comparison["v2"]["accuracy"] == 0.89

    def test_auto_version_generation(self, temp_model_dir, sample_model):
        """Test automatic version generation."""
        model, preprocessors = sample_model

        # Save without specifying version
        version_dir = save_model(model, preprocessors, model_dir=temp_model_dir)

        # Check that version was auto-generated
        assert Path(version_dir).exists()

        # Load and check version format (should be timestamp)
        _, _, metadata = load_model(model_dir=temp_model_dir)
        assert len(metadata["version"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
