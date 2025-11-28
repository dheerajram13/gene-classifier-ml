# Contributing to Gene Classifier ML

Thank you for your interest in contributing to the Gene Classifier ML project! This document provides guidelines and best practices for contributing.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [ML-Specific Guidelines](#ml-specific-guidelines)

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone git@github.com:your-username/gene-classifier-ml.git
   cd gene-classifier-ml
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Quick Setup

```bash
# Using Make (recommended)
make setup

# Or manually
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

### Using Conda

```bash
conda env create -f environment.yml
conda activate gene-classifier
pre-commit install
```

### Verify Installation

```bash
make test
```

## Code Standards

### Style Guide

We follow PEP 8 with these specifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Sorted with isort
- **Docstrings**: Google style
- **Type hints**: Encouraged for function signatures

### Code Formatting

Format your code before committing:

```bash
make format
```

Or run individually:

```bash
black .
isort .
```

### Linting

Check code quality:

```bash
make lint
```

This runs:
- Black (formatting check)
- isort (import sorting check)
- pylint (static analysis)
- mypy (type checking)

### Pre-commit Hooks

Pre-commit hooks run automatically before each commit:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Follow Arrange-Act-Assert pattern

Example:

```python
def test_preprocessing_handles_missing_values():
    # Arrange
    data = pd.DataFrame({'feature1': [1, 2, np.nan, 4, 5]})
    config = get_test_config()

    # Act
    result, _ = preprocess_data(data, config)

    # Assert
    assert result.isna().sum().sum() == 0
```

### Running Tests

```bash
# All tests with coverage
make test

# Fast test run (no coverage)
make test-fast

# Specific test file
pytest tests/test_preprocessing.py -v

# Specific test function
pytest tests/test_preprocessing.py::test_preprocessing_handles_missing_values -v
```

### Test Coverage

Maintain test coverage above 80%:

```bash
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html  # View coverage report
```

## Commit Guidelines

### Commit Message Format

Follow the Conventional Commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```bash
feat(preprocessing): add robust scaler option

Add RobustScaler as a preprocessing option for handling outliers
in feature scaling pipeline.

Closes #123
```

```bash
fix(model): correct KNN parameter validation

Fix validation logic for KNN n_neighbors parameter to ensure
it's always positive and odd.
```

### Atomic Commits

- Make small, focused commits
- Each commit should represent one logical change
- Commit working code (tests should pass)

## Pull Request Process

### Before Submitting

1. **Update your branch** with the latest main:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   make lint
   make test
   ```

3. **Update documentation** if needed

4. **Add tests** for new features

### Submitting the PR

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create a Pull Request on GitHub

3. Fill out the PR template completely

4. Link related issues

### PR Title Format

Use the same format as commit messages:

```
feat(preprocessing): add robust scaler option
```

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Updated existing tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

### Review Process

- At least one approval required
- All CI checks must pass
- Address reviewer feedback
- Squash commits if requested

## ML-Specific Guidelines

### Data Management

- **Never commit large data files** to git
- Use DVC for data versioning:
  ```bash
  dvc add data/new_dataset.csv
  git add data/new_dataset.csv.dvc
  ```
- Document data sources and preprocessing steps
- Include data validation checks

### Model Development

1. **Experiment Tracking**
   ```python
   from experiment_tracker import ExperimentTracker

   tracker = ExperimentTracker()
   tracker.start_run("experiment_name")
   tracker.log_params({"learning_rate": 0.01})
   tracker.log_metrics({"accuracy": 0.89})
   tracker.end_run()
   ```

2. **Model Versioning**
   ```python
   from model_utils import save_model

   save_model(
       model=trained_model,
       preprocessors=preprocessors,
       metadata={"accuracy": 0.89}
   )
   ```

3. **Reproducibility**
   - Set random seeds
   - Document environment
   - Version control code and config
   - Track hyperparameters

### Configuration Changes

- Update `config.yaml` for new parameters
- Maintain backward compatibility
- Document new configuration options
- Update schema validation if needed

### Adding New Features

1. **Preprocessing Method**
   - Add to `constants.py`
   - Update `preprocess_data()` function
   - Add tests
   - Update documentation

2. **Model**
   - Add to `model_config` in constants
   - Update hyperparameter grid
   - Add tests
   - Document usage

3. **Metric**
   - Add to monitoring system
   - Update experiment tracking
   - Add visualization if needed

### Performance Optimization

- Profile before optimizing
- Document performance improvements
- Add benchmarks for critical paths
- Consider memory usage

### Code Review Focus Areas

Reviewers will check for:

- **Correctness**: Does it work as intended?
- **Tests**: Are there adequate tests?
- **Documentation**: Is it well documented?
- **Reproducibility**: Can results be reproduced?
- **Performance**: Any performance implications?
- **Data Validation**: Proper data checks?
- **Error Handling**: Appropriate error handling?

## Additional Guidelines

### Documentation

- Update README.md for user-facing changes
- Update ML_BEST_PRACTICES.md for methodology changes
- Add docstrings to all public functions
- Include examples in docstrings

### Dependencies

- Minimize new dependencies
- Pin versions in requirements.txt
- Update environment.yml
- Document why new dependency is needed

### Backwards Compatibility

- Avoid breaking changes when possible
- Deprecate before removing
- Document breaking changes clearly
- Update migration guide

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open an Issue
- **Features**: Open an Issue for discussion first
- **Security**: Email maintainers directly

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

## Recognition

Contributors will be:
- Listed in project documentation
- Credited in release notes
- Acknowledged in published papers (if applicable)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Gene Classifier ML! 🎉
