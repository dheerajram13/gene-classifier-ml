"""
Model Performance Monitoring Module

Provides utilities for monitoring model performance in production,
detecting data drift, and tracking prediction quality over time.

Author: Data Mining Project
License: MIT
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from logger import get_logger

logger = get_logger(__name__)


class ModelMonitor:
    """
    Monitor model performance and data quality in production.
    """

    def __init__(self, monitor_dir: str = "monitoring"):
        """
        Initialize model monitor.

        Args:
            monitor_dir: Directory to store monitoring data
        """
        self.monitor_dir = Path(monitor_dir)
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.monitor_dir / "metrics_history.jsonl"
        self.drift_file = self.monitor_dir / "drift_history.jsonl"

    def log_prediction_batch(
        self,
        predictions: np.ndarray,
        features: pd.DataFrame,
        true_labels: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Log a batch of predictions for monitoring.

        Args:
            predictions: Model predictions
            features: Input features
            true_labels: Ground truth labels (if available)
            metadata: Additional metadata (model version, timestamp, etc.)
        """
        timestamp = datetime.now().isoformat()

        batch_data = {
            "timestamp": timestamp,
            "n_predictions": len(predictions),
            "prediction_distribution": {
                "class_0": int(np.sum(predictions == 0)),
                "class_1": int(np.sum(predictions == 1)),
            },
            "feature_stats": self._compute_feature_stats(features),
        }

        # Add performance metrics if labels are available
        if true_labels is not None:
            batch_data["metrics"] = self._compute_metrics(predictions, true_labels)

        # Add metadata
        if metadata:
            batch_data["metadata"] = metadata

        # Append to metrics file
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(batch_data) + "\n")

        logger.info(f"Logged prediction batch: {len(predictions)} predictions")

    def _compute_feature_stats(self, features: pd.DataFrame) -> Dict:
        """Compute feature statistics."""
        numeric_cols = features.select_dtypes(include=[np.number]).columns

        stats = {}
        for col in numeric_cols:
            stats[col] = {
                "mean": float(features[col].mean()),
                "std": float(features[col].std()),
                "min": float(features[col].min()),
                "max": float(features[col].max()),
                "missing_ratio": float(features[col].isna().mean()),
            }

        return stats

    def _compute_metrics(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray
    ) -> Dict:
        """Compute performance metrics."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        metrics = {
            "accuracy": float(accuracy_score(true_labels, predictions)),
            "f1_score": float(f1_score(true_labels, predictions)),
            "precision": float(precision_score(true_labels, predictions, zero_division=0)),
            "recall": float(recall_score(true_labels, predictions, zero_division=0)),
        }

        return metrics

    def detect_data_drift(
        self,
        reference_features: pd.DataFrame,
        current_features: pd.DataFrame,
        threshold: float = 0.1
    ) -> Dict:
        """
        Detect data drift between reference and current data.

        Args:
            reference_features: Reference (training) data
            current_features: Current (production) data
            threshold: Drift threshold for statistical tests

        Returns:
            Dictionary with drift detection results
        """
        numeric_cols = reference_features.select_dtypes(include=[np.number]).columns

        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "drifted_features": [],
            "feature_drift_scores": {},
        }

        for col in numeric_cols:
            # Compute distribution statistics
            ref_mean = reference_features[col].mean()
            ref_std = reference_features[col].std()
            curr_mean = current_features[col].mean()
            curr_std = current_features[col].std()

            # Simple drift detection: check if means differ significantly
            # (normalized by reference std dev)
            if ref_std > 0:
                drift_score = abs(curr_mean - ref_mean) / ref_std
            else:
                drift_score = 0.0

            drift_results["feature_drift_scores"][col] = float(drift_score)

            if drift_score > threshold:
                drift_results["drifted_features"].append(col)
                logger.warning(
                    f"Data drift detected in feature '{col}': "
                    f"score={drift_score:.4f}"
                )

        # Log drift results
        with open(self.drift_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(drift_results) + "\n")

        return drift_results

    def get_performance_over_time(
        self,
        metric: str = "f1_score",
        n_recent: Optional[int] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Get performance metric over time.

        Args:
            metric: Metric name to retrieve
            n_recent: Number of recent entries to retrieve (all if None)

        Returns:
            Tuple of (timestamps, metric_values)
        """
        if not self.metrics_file.exists():
            return [], []

        timestamps = []
        values = []

        with open(self.metrics_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

            # Get recent entries
            if n_recent:
                lines = lines[-n_recent:]

            for line in lines:
                data = json.loads(line)
                if "metrics" in data and metric in data["metrics"]:
                    timestamps.append(data["timestamp"])
                    values.append(data["metrics"][metric])

        return timestamps, values

    def get_prediction_distribution_over_time(
        self,
        n_recent: Optional[int] = None
    ) -> Dict:
        """
        Get prediction class distribution over time.

        Args:
            n_recent: Number of recent entries to retrieve

        Returns:
            Dictionary with timestamps and class distributions
        """
        if not self.metrics_file.exists():
            return {"timestamps": [], "class_0": [], "class_1": []}

        result = defaultdict(list)

        with open(self.metrics_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

            if n_recent:
                lines = lines[-n_recent:]

            for line in lines:
                data = json.loads(line)
                result["timestamps"].append(data["timestamp"])

                dist = data["prediction_distribution"]
                total = dist["class_0"] + dist["class_1"]
                result["class_0"].append(dist["class_0"] / total if total > 0 else 0)
                result["class_1"].append(dist["class_1"] / total if total > 0 else 0)

        return dict(result)

    def generate_monitoring_report(self) -> str:
        """
        Generate comprehensive monitoring report.

        Returns:
            Report as formatted string
        """
        report_lines = [
            "=" * 60,
            "MODEL MONITORING REPORT",
            "=" * 60,
            f"\nGenerated at: {datetime.now().isoformat()}",
        ]

        # Performance metrics
        report_lines.append("\n## Performance Over Time")

        timestamps, f1_scores = self.get_performance_over_time("f1_score")
        if f1_scores:
            report_lines.append(f"  Recent F1-scores: {len(f1_scores)} batches")
            report_lines.append(f"  Latest F1-score: {f1_scores[-1]:.4f}")
            report_lines.append(f"  Average F1-score: {np.mean(f1_scores):.4f}")
            report_lines.append(f"  Std F1-score: {np.std(f1_scores):.4f}")

        timestamps, accuracy = self.get_performance_over_time("accuracy")
        if accuracy:
            report_lines.append(f"\n  Recent Accuracy: {len(accuracy)} batches")
            report_lines.append(f"  Latest Accuracy: {accuracy[-1]:.4f}")
            report_lines.append(f"  Average Accuracy: {np.mean(accuracy):.4f}")

        # Prediction distribution
        report_lines.append("\n## Prediction Distribution")
        dist = self.get_prediction_distribution_over_time(n_recent=10)
        if dist["timestamps"]:
            avg_class_1 = np.mean(dist["class_1"])
            report_lines.append(f"  Average Class 1 ratio: {avg_class_1:.2%}")

        # Data drift
        report_lines.append("\n## Data Drift Status")
        if self.drift_file.exists():
            with open(self.drift_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if lines:
                    latest_drift = json.loads(lines[-1])
                    n_drifted = len(latest_drift["drifted_features"])
                    report_lines.append(f"  Drifted features: {n_drifted}")
                    if n_drifted > 0:
                        report_lines.append(f"  Features: {latest_drift['drifted_features']}")
                else:
                    report_lines.append("  No drift data available")
        else:
            report_lines.append("  No drift monitoring data")

        report_lines.append("\n" + "=" * 60)

        return "\n".join(report_lines)

    def check_performance_degradation(
        self,
        metric: str = "f1_score",
        threshold: float = 0.05,
        window: int = 10
    ) -> Tuple[bool, float]:
        """
        Check if model performance has degraded.

        Args:
            metric: Metric to check
            threshold: Degradation threshold
            window: Number of recent batches to compare

        Returns:
            Tuple of (is_degraded, degradation_amount)
        """
        timestamps, values = self.get_performance_over_time(metric, n_recent=window*2)

        if len(values) < window * 2:
            logger.warning("Not enough data for degradation check")
            return False, 0.0

        # Compare recent window to previous window
        recent_avg = np.mean(values[-window:])
        previous_avg = np.mean(values[-window*2:-window])

        degradation = previous_avg - recent_avg

        is_degraded = degradation > threshold

        if is_degraded:
            logger.warning(
                f"Performance degradation detected: "
                f"{metric} dropped by {degradation:.4f}"
            )

        return is_degraded, degradation


# Example usage
if __name__ == "__main__":
    # Create monitor
    monitor = ModelMonitor()

    # Simulate logging predictions
    predictions = np.random.randint(0, 2, 100)
    features = pd.DataFrame(np.random.randn(100, 10))
    true_labels = np.random.randint(0, 2, 100)

    monitor.log_prediction_batch(
        predictions,
        features,
        true_labels,
        metadata={"model_version": "v1"}
    )

    # Generate report
    report = monitor.generate_monitoring_report()
    print(report)
