# Contributing to FeatureIQ

Thank you for your interest in contributing to FeatureIQ! This document provides guidelines and instructions for contributing.

## Getting Started

### Development Setup

1. **Fork and clone** the repository:
   ```bash
   git clone https://github.com/<your-username>/featureiq.git
   cd featureiq
   ```

2. **Install Poetry** (if not already installed):
   ```bash
   pip install poetry
   ```

3. **Install dependencies** (including dev tools):
   ```bash
   poetry install --with dev
   ```

4. **Install pre-commit hooks**:
   ```bash
   poetry run pre-commit install
   ```

5. **Run the test suite** to verify your setup:
   ```bash
   poetry run pytest
   ```

## How to Contribute

### Reporting Bugs

- Open a [GitHub Issue](https://github.com/khedekarpratik0337/featureiq/issues) with:
  - A clear title and description
  - Steps to reproduce
  - Expected vs actual behavior
  - Python version, OS, and `featureiq` version

### Suggesting Features

- Open a [GitHub Issue](https://github.com/khedekarpratik0337/featureiq/issues) with the `enhancement` label
- Describe the use case and proposed behavior

### Submitting Code

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines below

3. **Write tests** for any new functionality

4. **Run the full quality suite**:
   ```bash
   poetry run black featureiq/ tests/
   poetry run isort featureiq/ tests/
   poetry run flake8 featureiq/ tests/
   poetry run mypy featureiq/
   poetry run pytest
   ```

5. **Open a Pull Request** against `main` with:
   - A clear description of the change
   - Reference to any related issues
   - Confirmation that tests pass

### Contributing Ontology Rules

FeatureIQ's knowledge graph is open and extensible. See [docs/contributing_rules.md](docs/contributing_rules.md) for the full guide on adding new YAML-based feature engineering rules.

## Code Style

- **Formatter:** [Black](https://github.com/psf/black) (line length 88)
- **Import sorting:** [isort](https://pycqa.github.io/isort/) with `profile = "black"`
- **Linting:** [flake8](https://flake8.pycqa.org/) with `E203` ignored
- **Type checking:** [mypy](https://mypy-lang.org/) in strict mode
- **Docstrings:** Google style, required for all public functions and classes
- **Comments:** Only for non-obvious logic. Do not narrate what code does.

## Testing

- Tests live in `tests/` and mirror the `featureiq/` package structure
- Use `pytest` fixtures from `tests/conftest.py` where possible
- Aim for coverage on new code; run `poetry run pytest --cov=featureiq` to check

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
