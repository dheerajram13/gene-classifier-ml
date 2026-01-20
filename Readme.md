# Gene Function Classification

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange.svg)](https://scikit-learn.org/)

## Overview

A machine learning pipeline for binary classification of gene functions, specifically predicting whether genes have cell communication capabilities. This project implements a comprehensive data preprocessing pipeline, automated model selection, and ensemble learning techniques to handle imbalanced biological datasets.

### Key Features

- **Automated Preprocessing Optimization**: Tests multiple combinations of imputation, scaling, and outlier handling methods
- **Ensemble Learning**: Combines K-Nearest Neighbors and Random Forest classifiers using soft voting
- **Cross-Validation**: 5-fold stratified cross-validation for robust model evaluation
- **Hyperparameter Tuning**: Grid search optimization for model parameters
- **Imbalanced Data Handling**: F1-score optimization for datasets with class imbalances

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Configuration](#configuration)
- [Contributing](#contributing)

## Quick Start

```bash
# Clone and setup
git clone git@github.com:dheerajram13/gene-classifier-ml.git
cd gene-classifier-ml
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Add your data files in the Data dir (DM_project_24.csv and test_data.csv)

# Run the pipeline
python main.py

# View results in predictions.txt
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Data\ Mining
   ```

2. **Create a virtual environment**

   Using conda:
   ```bash
   conda create -n gene-classifier python=3.9
   conda activate gene-classifier
   ```

   Using venv:
   ```bash
   python3 -m venv venv

   # On macOS/Linux:
   source venv/bin/activate

   # On Windows:
   .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data files**

   Ensure the following CSV files are in the project directory:
   - `DM_project_24.csv` - Training dataset (1,600 samples, 106 features)
   - `test_data.csv` - Test dataset (817 samples, 105 features)

## Usage

### Basic Usage

Run the classification pipeline:

```bash
python main.py
```

This will:
1. Load and preprocess the training data
2. Train the ensemble model
3. Perform cross-validation
4. Generate predictions on test data
5. Save results to `predictions.txt`

### Output Format

The `predictions.txt` file contains:
- One prediction per line (0 or 1) for each test sample
- Final line with model accuracy and F1-score

Example:
```
0,
1,
0,
...
0.892,0.745,
```

### Advanced Usage

To enable full preprocessing optimization and model selection (slower but more thorough):

1. Uncomment lines 312-314 in `main.py`
2. This will test all preprocessing combinations and models

## Project Structure

```
.
├── main.py                 # Main pipeline and execution
├── constants.py            # Model and preprocessing configurations
├── requirements.txt        # Python dependencies
├── Readme.md              # Project documentation
├── DM_project_24.csv      # Training data (not in repo)
├── test_data.csv          # Test data (not in repo)
└── predictions.txt        # Output predictions (generated)
```

## Methodology

### Data Preprocessing

The pipeline implements multiple preprocessing strategies:

#### 1. Missing Value Imputation
- **Mean Imputation** (selected): Replaces missing values with column mean
- Median Imputation: Uses median for robust handling of outliers
- Iterative Imputation: MICE-based multivariate imputation

#### 2. Feature Scaling
- **StandardScaler** (selected): Zero mean, unit variance normalization
- MinMaxScaler: Scales features to [0, 1] range
- RobustScaler: Uses median and IQR for outlier resistance

#### 3. Outlier Handling
- **None** (selected): No outlier removal for final model
- Z-score method: Removes values >3 standard deviations
- IQR method: Removes values outside 1.5×IQR range

### Classification Models

#### Final Ensemble Model

A **VotingClassifier** with soft voting combining:

**1. K-Nearest Neighbors (KNN)**
- `n_neighbors`: 8
- `metric`: Euclidean distance
- `weights`: Distance-weighted voting

**2. Random Forest**
- `n_estimators`: 100 trees
- `max_depth`: 10
- `min_samples_split`: 5
- `min_samples_leaf`: 2
- `random_state`: 42 (reproducibility)

#### Voting Strategy
Soft voting (probability averaging) for improved calibration and performance.

### Model Selection Process

1. **Preprocessing Optimization**: 5-fold CV tests all preprocessing combinations
2. **Hyperparameter Tuning**: GridSearchCV for each model type
3. **Model Evaluation**: F1-score and accuracy on validation folds
4. **Ensemble Construction**: Combines best-performing models

### Evaluation Metrics

- **Primary Metric**: F1-Score (harmonic mean of precision and recall)
- **Secondary Metric**: Accuracy
- **Validation**: 5-fold stratified cross-validation

F1-score is prioritized due to class imbalance (88% negative, 12% positive).

## Model Performance

Expected performance (5-fold cross-validation):
- **Accuracy**: ~89-91%
- **F1-Score**: ~74-76%

Note: Performance may vary slightly due to cross-validation randomness.

## Configuration

### Modifying Model Parameters

Edit `constants.py` to customize:

**Model configurations** (`model_config`):
```python
model_config = {
    "knn": {
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": range(1, 18), ...}
    },
    ...
}
```

**Preprocessing methods** (`pre_processing_config`):
```python
pre_processing_config = {
    "imputation_methods": {...},
    "scaling_methods": {...},
    "outlier_methods": {...}
}
```

**Final configuration** (`final_preprocessing_config`):
```python
final_preprocessing_config = {
    "imputation": ("mean", SimpleImputer()),
    "scaling": ("standard", StandardScaler()),
    "outlier": ("none", None),
}
```

## Dependencies

Core libraries:
- `pandas >= 2.2.3` - Data manipulation
- `numpy >= 2.1.2` - Numerical computing
- `scikit-learn >= 1.5.2` - Machine learning algorithms

See `requirements.txt` for complete list.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes with clear commit messages
4. Add tests if applicable
5. Update documentation as needed
6. Submit a pull request

### Code Style

This project follows:
- PEP 8 style guidelines
- Type hints for function signatures
- Comprehensive docstrings (Google style)


## Acknowledgments

- Dataset: Gene function classification dataset
- Built with scikit-learn machine learning library
- Developed as part of a Data Mining course project

## Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Note**: This is an academic project demonstrating machine learning best practices for biological data classification.




