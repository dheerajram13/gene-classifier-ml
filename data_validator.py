"""
Data Validation Module

Provides comprehensive data validation and quality checks for ML pipelines.
Ensures data integrity and catches issues early in the pipeline.

Author: Data Mining Project
License: MIT
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Results from data validation checks."""
    is_valid: bool
    warnings: List[str]
    errors: List[str]
    statistics: Dict


class DataValidator:
    """
    Validate data quality and schema for ML pipelines.
    """

    def __init__(
        self,
        max_missing_ratio: float = 0.5,
        check_missing: bool = True,
        check_types: bool = True,
        check_ranges: bool = True
    ):
        """
        Initialize data validator.

        Args:
            max_missing_ratio: Maximum allowed ratio of missing values per column
            check_missing: Whether to check for missing values
            check_types: Whether to check data types
            check_ranges: Whether to check value ranges
        """
        self.max_missing_ratio = max_missing_ratio
        self.check_missing = check_missing
        self.check_types = check_types
        self.check_ranges = check_ranges

    def validate(self, data: pd.DataFrame, schema: Optional[Dict] = None) -> ValidationResult:
        """
        Perform comprehensive data validation.

        Args:
            data: DataFrame to validate
            schema: Optional schema specification

        Returns:
            ValidationResult with validation details
        """
        warnings = []
        errors = []
        statistics = {}

        logger.info("Starting data validation...")

        # Basic statistics
        statistics['n_rows'] = len(data)
        statistics['n_columns'] = len(data.columns)
        statistics['memory_usage'] = data.memory_usage(deep=True).sum() / 1024**2  # MB

        # Check for empty data
        if len(data) == 0:
            errors.append("Dataset is empty")
            return ValidationResult(False, warnings, errors, statistics)

        # Check missing values
        if self.check_missing:
            missing_warnings, missing_errors, missing_stats = self._check_missing_values(data)
            warnings.extend(missing_warnings)
            errors.extend(missing_errors)
            statistics.update(missing_stats)

        # Check data types
        if self.check_types:
            type_warnings, type_stats = self._check_data_types(data)
            warnings.extend(type_warnings)
            statistics.update(type_stats)

        # Check value ranges
        if self.check_ranges:
            range_warnings, range_stats = self._check_value_ranges(data)
            warnings.extend(range_warnings)
            statistics.update(range_stats)

        # Check for duplicates
        dup_warnings, dup_stats = self._check_duplicates(data)
        warnings.extend(dup_warnings)
        statistics.update(dup_stats)

        # Schema validation if provided
        if schema:
            schema_warnings, schema_errors = self._validate_schema(data, schema)
            warnings.extend(schema_warnings)
            errors.extend(schema_errors)

        is_valid = len(errors) == 0

        if is_valid:
            logger.info(f"Validation passed with {len(warnings)} warnings")
        else:
            logger.error(f"Validation failed with {len(errors)} errors and {len(warnings)} warnings")

        return ValidationResult(is_valid, warnings, errors, statistics)

    def _check_missing_values(self, data: pd.DataFrame) -> Tuple[List[str], List[str], Dict]:
        """Check for missing values."""
        warnings = []
        errors = []
        stats = {}

        missing_counts = data.isna().sum()
        missing_ratios = missing_counts / len(data)

        stats['total_missing'] = int(missing_counts.sum())
        stats['columns_with_missing'] = int((missing_counts > 0).sum())
        stats['missing_ratio_per_column'] = missing_ratios.to_dict()

        for col, ratio in missing_ratios.items():
            if ratio > self.max_missing_ratio:
                errors.append(
                    f"Column '{col}' has {ratio:.2%} missing values "
                    f"(exceeds threshold of {self.max_missing_ratio:.2%})"
                )
            elif ratio > 0.1:  # Warn if > 10% missing
                warnings.append(
                    f"Column '{col}' has {ratio:.2%} missing values"
                )

        return warnings, errors, stats

    def _check_data_types(self, data: pd.DataFrame) -> Tuple[List[str], Dict]:
        """Check data types."""
        warnings = []
        stats = {}

        dtypes = data.dtypes.value_counts().to_dict()
        stats['data_types'] = {str(k): int(v) for k, v in dtypes.items()}

        # Check for object columns (might need encoding)
        object_cols = data.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            warnings.append(
                f"Found {len(object_cols)} object/categorical columns: {list(object_cols)}"
            )

        # Check for mixed types
        for col in data.columns:
            if data[col].dtype == 'object':
                # Try to identify mixed types
                non_null = data[col].dropna()
                if len(non_null) > 0:
                    types = non_null.apply(type).unique()
                    if len(types) > 1:
                        warnings.append(
                            f"Column '{col}' has mixed types: {[t.__name__ for t in types]}"
                        )

        return warnings, stats

    def _check_value_ranges(self, data: pd.DataFrame) -> Tuple[List[str], Dict]:
        """Check value ranges for numerical columns."""
        warnings = []
        stats = {}

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        stats['numeric_columns'] = len(numeric_cols)

        range_info = {}
        for col in numeric_cols:
            col_min = float(data[col].min())
            col_max = float(data[col].max())
            col_mean = float(data[col].mean())
            col_std = float(data[col].std())

            range_info[col] = {
                'min': col_min,
                'max': col_max,
                'mean': col_mean,
                'std': col_std
            }

            # Check for infinite values
            if np.isinf(data[col]).any():
                warnings.append(f"Column '{col}' contains infinite values")

            # Check for constant columns
            if col_std == 0:
                warnings.append(f"Column '{col}' has zero variance (constant value)")

            # Check for extreme values (beyond 5 standard deviations)
            if col_std > 0:
                z_scores = np.abs((data[col] - col_mean) / col_std)
                n_extreme = (z_scores > 5).sum()
                if n_extreme > 0:
                    warnings.append(
                        f"Column '{col}' has {n_extreme} extreme values (>5 std devs)"
                    )

        stats['value_ranges'] = range_info

        return warnings, stats

    def _check_duplicates(self, data: pd.DataFrame) -> Tuple[List[str], Dict]:
        """Check for duplicate rows."""
        warnings = []
        stats = {}

        n_duplicates = data.duplicated().sum()
        stats['n_duplicates'] = int(n_duplicates)
        stats['duplicate_ratio'] = float(n_duplicates / len(data))

        if n_duplicates > 0:
            warnings.append(
                f"Found {n_duplicates} duplicate rows ({n_duplicates/len(data):.2%})"
            )

        return warnings, stats

    def _validate_schema(
        self,
        data: pd.DataFrame,
        schema: Dict
    ) -> Tuple[List[str], List[str]]:
        """Validate data against schema."""
        warnings = []
        errors = []

        # Check required columns
        if 'required_columns' in schema:
            required = set(schema['required_columns'])
            actual = set(data.columns)
            missing = required - actual
            extra = actual - required

            if missing:
                errors.append(f"Missing required columns: {missing}")
            if extra:
                warnings.append(f"Extra columns not in schema: {extra}")

        # Check column types
        if 'column_types' in schema:
            for col, expected_type in schema['column_types'].items():
                if col in data.columns:
                    actual_type = str(data[col].dtype)
                    if actual_type != expected_type:
                        warnings.append(
                            f"Column '{col}' has type '{actual_type}', "
                            f"expected '{expected_type}'"
                        )

        # Check number of rows
        if 'min_rows' in schema:
            if len(data) < schema['min_rows']:
                errors.append(
                    f"Dataset has {len(data)} rows, "
                    f"minimum required is {schema['min_rows']}"
                )

        return warnings, errors

    def generate_schema(self, data: pd.DataFrame) -> Dict:
        """
        Generate schema from data.

        Args:
            data: DataFrame to generate schema from

        Returns:
            Schema dictionary
        """
        schema = {
            'required_columns': list(data.columns),
            'column_types': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'n_rows': len(data),
            'n_columns': len(data.columns),
            'generated_at': pd.Timestamp.now().isoformat()
        }

        return schema

    def print_report(self, result: ValidationResult) -> None:
        """
        Print validation report.

        Args:
            result: ValidationResult to print
        """
        print("\n" + "=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)

        print(f"\nStatus: {'PASSED' if result.is_valid else 'FAILED'}")

        print(f"\nDataset Statistics:")
        print(f"  Rows: {result.statistics.get('n_rows', 'N/A')}")
        print(f"  Columns: {result.statistics.get('n_columns', 'N/A')}")
        print(f"  Memory: {result.statistics.get('memory_usage', 0):.2f} MB")

        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for i, error in enumerate(result.errors, 1):
                print(f"  {i}. {error}")

        if result.warnings:
            print(f"\nWarnings ({len(result.warnings)}):")
            for i, warning in enumerate(result.warnings, 1):
                print(f"  {i}. {warning}")

        print("\n" + "=" * 60)


# Example usage
if __name__ == "__main__":
    # Create sample data
    data = pd.DataFrame({
        'feature1': [1, 2, 3, np.nan, 5],
        'feature2': [10, 20, 30, 40, 50],
        'feature3': ['a', 'b', 'c', 'd', 'e']
    })

    # Validate
    validator = DataValidator()
    result = validator.validate(data)
    validator.print_report(result)

    # Generate schema
    schema = validator.generate_schema(data)
    print("\nGenerated Schema:")
    print(schema)
