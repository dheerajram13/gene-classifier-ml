"""
Model Utilities Module

Provides functions for model serialization, versioning, and persistence.
Ensures reproducibility and proper model lifecycle management.

Author: Data Mining Project
License: MIT
"""

import json
import joblib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from logger import get_logger

logger = get_logger(__name__)


def save_model(
    model: Any,
    preprocessors: Dict[str, Any],
    model_dir: str = "models",
    model_name: str = "gene_classifier",
    version: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> str:
    """
    Save trained model and preprocessors with versioning.

    Args:
        model: Trained sklearn model
        preprocessors: Dictionary of fitted preprocessors
        model_dir: Directory to save models
        model_name: Base name for model files
        version: Version string (auto-generated if None)
        metadata: Optional metadata dictionary (metrics, params, etc.)

    Returns:
        Path to saved model directory

    Example:
        >>> metadata = {"accuracy": 0.89, "f1_score": 0.75}
        >>> save_model(model, preprocessors, metadata=metadata)
    """
    # Create model directory
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    # Generate version if not provided
    if version is None:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create versioned subdirectory
    version_dir = model_path / f"{model_name}_v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_file = version_dir / "model.joblib"
    joblib.dump(model, model_file)
    logger.info(f"Model saved to {model_file}")

    # Save preprocessors
    preprocessor_file = version_dir / "preprocessors.joblib"
    joblib.dump(preprocessors, preprocessor_file)
    logger.info(f"Preprocessors saved to {preprocessor_file}")

    # Save metadata
    if metadata is None:
        metadata = {}

    metadata.update({
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "model_type": type(model).__name__,
    })

    metadata_file = version_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_file}")

    # Create/update latest symlink
    latest_link = model_path / f"{model_name}_latest"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(version_dir.name)

    logger.info(f"Model saved successfully with version {version}")
    return str(version_dir)


def load_model(
    model_dir: str = "models",
    model_name: str = "gene_classifier",
    version: Optional[str] = None
) -> Tuple[Any, Dict[str, Any], Dict]:
    """
    Load model, preprocessors, and metadata.

    Args:
        model_dir: Directory containing models
        model_name: Base name for model files
        version: Specific version to load (loads latest if None)

    Returns:
        Tuple of (model, preprocessors, metadata)

    Example:
        >>> model, preprocessors, metadata = load_model()
        >>> print(f"Loaded model version: {metadata['version']}")
    """
    model_path = Path(model_dir)

    # Determine which version to load
    if version is None:
        # Load latest version
        latest_link = model_path / f"{model_name}_latest"
        if not latest_link.exists():
            raise FileNotFoundError(f"No model found at {latest_link}")
        version_dir = model_path / latest_link.readlink()
    else:
        version_dir = model_path / f"{model_name}_v{version}"

    if not version_dir.exists():
        raise FileNotFoundError(f"Model version not found: {version_dir}")

    # Load model
    model_file = version_dir / "model.joblib"
    model = joblib.load(model_file)
    logger.info(f"Model loaded from {model_file}")

    # Load preprocessors
    preprocessor_file = version_dir / "preprocessors.joblib"
    preprocessors = joblib.load(preprocessor_file)
    logger.info(f"Preprocessors loaded from {preprocessor_file}")

    # Load metadata
    metadata_file = version_dir / "metadata.json"
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    logger.info(f"Metadata loaded from {metadata_file}")

    return model, preprocessors, metadata


def list_model_versions(
    model_dir: str = "models",
    model_name: str = "gene_classifier"
) -> list:
    """
    List all available model versions.

    Args:
        model_dir: Directory containing models
        model_name: Base name for model files

    Returns:
        List of version strings

    Example:
        >>> versions = list_model_versions()
        >>> print(f"Available versions: {versions}")
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        return []

    versions = []
    for item in model_path.iterdir():
        if item.is_dir() and item.name.startswith(f"{model_name}_v"):
            version = item.name.replace(f"{model_name}_v", "")
            versions.append(version)

    return sorted(versions, reverse=True)


def compare_models(
    model_dir: str = "models",
    model_name: str = "gene_classifier"
) -> Dict[str, Dict]:
    """
    Compare metrics across all model versions.

    Args:
        model_dir: Directory containing models
        model_name: Base name for model files

    Returns:
        Dictionary mapping versions to their metadata

    Example:
        >>> comparison = compare_models()
        >>> for version, data in comparison.items():
        ...     print(f"{version}: F1={data.get('f1_score', 'N/A')}")
    """
    versions = list_model_versions(model_dir, model_name)
    comparison = {}

    for version in versions:
        metadata_file = Path(model_dir) / f"{model_name}_v{version}" / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                comparison[version] = json.load(f)

    return comparison
