"""
Experiment Tracking Module

Provides lightweight experiment tracking functionality for ML experiments.
Can be extended to integrate with MLflow, Weights & Biases, or other platforms.

Author: Data Mining Project
License: MIT
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from logger import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """
    Lightweight experiment tracker for ML experiments.

    Tracks hyperparameters, metrics, and metadata for each experiment run.
    """

    def __init__(self, experiment_dir: str = "experiments"):
        """
        Initialize experiment tracker.

        Args:
            experiment_dir: Directory to store experiment logs
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.current_run = None
        self.run_data = {}

    def start_run(self, run_name: Optional[str] = None) -> str:
        """
        Start a new experiment run.

        Args:
            run_name: Optional name for the run (auto-generated if None)

        Returns:
            Run ID
        """
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.current_run = run_name
        self.run_data = {
            "run_id": run_name,
            "start_time": datetime.now().isoformat(),
            "parameters": {},
            "metrics": {},
            "artifacts": [],
            "metadata": {}
        }

        logger.info(f"Started experiment run: {run_name}")
        return run_name

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log experiment parameters.

        Args:
            params: Dictionary of parameter names and values
        """
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        self.run_data["parameters"].update(params)
        logger.debug(f"Logged parameters: {params}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a single metric value.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number for tracking over time
        """
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        if key not in self.run_data["metrics"]:
            self.run_data["metrics"][key] = []

        metric_entry = {"value": value}
        if step is not None:
            metric_entry["step"] = step

        self.run_data["metrics"][key].append(metric_entry)
        logger.debug(f"Logged metric: {key}={value}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for tracking over time
        """
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(self, artifact_path: str, description: str = "") -> None:
        """
        Log an artifact (file) associated with the experiment.

        Args:
            artifact_path: Path to the artifact file
            description: Optional description of the artifact
        """
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        self.run_data["artifacts"].append({
            "path": artifact_path,
            "description": description,
            "timestamp": datetime.now().isoformat()
        })
        logger.debug(f"Logged artifact: {artifact_path}")

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Log additional metadata.

        Args:
            metadata: Dictionary of metadata
        """
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        self.run_data["metadata"].update(metadata)
        logger.debug(f"Logged metadata: {metadata}")

    def end_run(self, status: str = "completed") -> str:
        """
        End the current experiment run and save results.

        Args:
            status: Run status (completed, failed, etc.)

        Returns:
            Path to saved experiment file
        """
        if self.current_run is None:
            raise RuntimeError("No active run to end.")

        self.run_data["end_time"] = datetime.now().isoformat()
        self.run_data["status"] = status

        # Save run data
        run_file = self.experiment_dir / f"{self.current_run}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(self.run_data, f, indent=2)

        logger.info(f"Ended experiment run: {self.current_run}")
        logger.info(f"Experiment data saved to: {run_file}")

        self.current_run = None
        return str(run_file)

    def load_run(self, run_id: str) -> Dict[str, Any]:
        """
        Load data from a previous experiment run.

        Args:
            run_id: ID of the run to load

        Returns:
            Dictionary containing run data
        """
        run_file = self.experiment_dir / f"{run_id}.json"
        if not run_file.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")

        with open(run_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_runs(self) -> list:
        """
        List all experiment runs.

        Returns:
            List of run IDs
        """
        runs = []
        for run_file in self.experiment_dir.glob("*.json"):
            runs.append(run_file.stem)
        return sorted(runs, reverse=True)

    def compare_runs(self, run_ids: Optional[list] = None) -> Dict[str, Dict]:
        """
        Compare metrics across multiple runs.

        Args:
            run_ids: List of run IDs to compare (all runs if None)

        Returns:
            Dictionary mapping run IDs to their metrics
        """
        if run_ids is None:
            run_ids = self.list_runs()

        comparison = {}
        for run_id in run_ids:
            try:
                run_data = self.load_run(run_id)
                comparison[run_id] = {
                    "parameters": run_data.get("parameters", {}),
                    "metrics": run_data.get("metrics", {}),
                    "status": run_data.get("status", "unknown")
                }
            except FileNotFoundError:
                logger.warning(f"Run not found: {run_id}")

        return comparison

    def get_best_run(self, metric: str, mode: str = "max") -> tuple:
        """
        Find the best run based on a specific metric.

        Args:
            metric: Metric name to optimize
            mode: 'max' or 'min' for optimization direction

        Returns:
            Tuple of (run_id, metric_value)
        """
        runs = self.list_runs()
        best_run = None
        best_value = float('-inf') if mode == 'max' else float('inf')

        for run_id in runs:
            try:
                run_data = self.load_run(run_id)
                metrics = run_data.get("metrics", {})

                if metric in metrics:
                    # Get the latest value if metric has multiple entries
                    values = metrics[metric]
                    if isinstance(values, list):
                        value = values[-1].get("value") if isinstance(values[-1], dict) else values[-1]
                    else:
                        value = values

                    if mode == 'max' and value > best_value:
                        best_value = value
                        best_run = run_id
                    elif mode == 'min' and value < best_value:
                        best_value = value
                        best_run = run_id

            except (FileNotFoundError, KeyError):
                continue

        return best_run, best_value


# Example usage
if __name__ == "__main__":
    tracker = ExperimentTracker()

    # Start a new run
    tracker.start_run("example_run")

    # Log parameters
    tracker.log_params({
        "learning_rate": 0.01,
        "n_estimators": 100,
        "max_depth": 10
    })

    # Log metrics
    tracker.log_metrics({
        "accuracy": 0.89,
        "f1_score": 0.75,
        "precision": 0.80,
        "recall": 0.71
    })

    # End the run
    tracker.end_run()

    print(f"Available runs: {tracker.list_runs()}")
