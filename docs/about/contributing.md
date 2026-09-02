# Contributing to PowerGenome

Thank you for your interest in contributing to PowerGenome! This document provides guidelines for contributing code, documentation, and reporting issues.

## Ways to Contribute

- **Report bugs**: Open an issue on GitHub
- **Suggest features**: Start a discussion on groups.io or GitHub Discussions
- **Improve documentation**: Fix typos, clarify explanations, add examples
- **Submit code**: Fix bugs, add features, improve performance
- **Answer questions**: Help other users on groups.io

## Getting Started

### 1. Set Up Development Environment

```bash
# Clone the repository
git clone https://github.com/PowerGenome/PowerGenome.git
cd PowerGenome

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 3. Make Your Changes

- Write clear, readable code
- Add tests for new functionality
- Update documentation as needed
- Follow existing code style (enforced by pre-commit hooks)

### 4. Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/generators_test.py -v

# Run with coverage
pytest --cov=powergenome tests/
```

### 5. Submit a Pull Request

1. Push your branch to GitHub
2. Open a pull request against the `master` branch
3. Fill out the PR template with:
   - Description of changes
   - Related issues
   - Testing performed
4. Wait for review and CI checks

## Code Style

PowerGenome uses automated code formatting and linting:

- **black**: Code formatting (runs automatically via pre-commit)
- **isort**: Import sorting
- **flake8**: Linting (in some configurations)

Pre-commit hooks will automatically format your code when you commit. If formatting changes are made, you'll need to stage and commit again.

## Testing Guidelines

- All new features should include tests
- Bug fixes should include a regression test
- Tests should be fast and isolated
- Use fixtures for test data
- Mock external dependencies

Example test structure:

```python
def test_feature_name():
    # Arrange
    input_data = create_test_data()

    # Act
    result = function_to_test(input_data)

    # Assert
    assert result == expected_output
```

## Documentation Guidelines

When documenting:

- **Be clear and concise**: Use simple language
- **Provide examples**: Show, don't just tell
- **Update related docs**: Keep everything consistent
- **Follow Diataxis**: Put content in the right section
  - Tutorials: Learning-oriented, step-by-step
  - How-To: Task-oriented, specific goals
  - Reference: Information-oriented, technical details
  - Explanation: Understanding-oriented, background

### Building Documentation Locally

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Serve docs locally
mkdocs serve

# Open browser to http://127.0.0.1:8000
```

## Reporting Issues

When reporting bugs, include:

- PowerGenome version
- Python version
- Operating system
- Minimal reproducible example
- Full error message and traceback
- Expected vs. actual behavior

## Code Review Process

- All submissions require review before merging
- Reviewers may request changes
- CI must pass before merging
- Maintainers may ask for additional tests or documentation

## Questions?

- **General questions**: [groups.io/g/powergenome](https://groups.io/g/powergenome)
- **Bug reports**: [GitHub Issues](https://github.com/PowerGenome/PowerGenome/issues)
- **Feature discussions**: [GitHub Discussions](https://github.com/PowerGenome/PowerGenome/discussions)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
