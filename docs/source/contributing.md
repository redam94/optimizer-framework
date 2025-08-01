# Contributing to Atlas

Thank you for your interest in contributing to Atlas! This guide will help you get started with contributing code, documentation, and ideas to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Contributions](#making-contributions)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Community](#community)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please read and follow our Code of Conduct:

- Be respectful and considerate in your communication
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences
- Show empathy towards other community members

## Getting Started

### Types of Contributions

We welcome many types of contributions:

- **Code**: New features, bug fixes, performance improvements
- **Documentation**: Tutorials, examples, API docs, translations
- **Tests**: Unit tests, integration tests, performance benchmarks
- **Ideas**: Feature requests, design proposals, architecture discussions
- **Community**: Answering questions, reviewing PRs, mentoring

### First-Time Contributors

If you're new to open source or this project:

1. Look for issues labeled `good first issue` or `beginner-friendly`
2. Read through the documentation to understand the project
3. Set up your development environment
4. Start with small contributions to get familiar with the process
5. Don't hesitate to ask questions!

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, poetry)
- Docker (optional, for integration tests)

### Setting Up Your Environment

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/optimizer-framework.git
   cd optimizer-framework
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Install in development mode with all extras
   pip install -e ".[dev,test,docs]"
   
   # Install pre-commit hooks
   pre-commit install
   ```

4. **Set up remote tracking**
   ```bash
   git remote add upstream https://github.com/redam94/optimizer-framework.git
   git fetch upstream
   ```

5. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Development Tools

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing
- **coverage**: Code coverage
- **pre-commit**: Git hooks

Run all checks:
```bash
# Format code
black src tests
isort src tests

# Run linting
flake8 src tests

# Type checking
mypy src

# Run tests
pytest

# Check coverage
pytest --cov=optimizer_framework --cov-report=html
```

## Making Contributions

### Finding Something to Work On

1. **Check existing issues**: Look for open issues that interest you
2. **Create an issue**: If you have a new idea, create an issue to discuss it
3. **Ask for guidance**: Comment on issues if you need clarification

### Workflow

1. **Sync with upstream**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/descriptive-name
   ```

3. **Make your changes**
   - Write code following our standards
   - Add tests for new functionality
   - Update documentation as needed

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new optimization algorithm
   
   - Implemented XYZ algorithm
   - Added comprehensive tests
   - Updated documentation
   
   Closes #123"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/descriptive-name
   ```

6. **Create a Pull Request**
   - Go to GitHub and click "New Pull Request"
   - Fill out the PR template
   - Link related issues

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or changes
- `chore`: Build process or auxiliary tool changes

Examples:
```bash
feat(optimizer): add CVXPY backend support
fix(models): handle missing data in predictions
docs(api): update ModelWrapper documentation
test(integration): add Docker model tests
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 88 characters (Black default)
- Use double quotes for strings
- Use trailing commas in multi-line collections
- Sort imports with isort

### Code Structure

```python
"""Module docstring explaining purpose and usage."""

from __future__ import annotations

# Standard library imports
import os
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from optimizer_framework.core import BaseClass


class MyClass:
    """Class docstring with description.
    
    Longer description if needed, explaining the purpose
    and usage of the class.
    
    Attributes:
        attribute1: Description of attribute1
        attribute2: Description of attribute2
        
    Example:
        >>> obj = MyClass()
        >>> obj.method()
        'result'
    """
    
    def __init__(self, param1: str, param2: Optional[int] = None) -> None:
        """Initialize the class.
        
        Args:
            param1: Description of param1
            param2: Description of param2. Defaults to None.
            
        Raises:
            ValueError: If param1 is empty
        """
        if not param1:
            raise ValueError("param1 cannot be empty")
            
        self.param1 = param1
        self.param2 = param2 or 10
    
    def method(self) -> str:
        """Brief description of method.
        
        Longer description if needed.
        
        Returns:
            Description of return value
            
        Raises:
            RuntimeError: If something goes wrong
        """
        try:
            result = self._internal_method()
            return result
        except Exception as e:
            raise RuntimeError(f"Method failed: {e}") from e
    
    def _internal_method(self) -> str:
        """Private methods also get docstrings."""
        return "result"
```

### Type Hints

Always use type hints:

```python
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import xarray as xr

def optimize(
    budget: Dict[str, float],
    constraints: Optional[Dict[str, Any]] = None,
    *,  # Force keyword-only arguments after this
    max_iter: int = 1000,
    tolerance: float = 1e-6,
) -> Tuple[Dict[str, float], float]:
    """Optimize budget allocation."""
    ...

# Use numpy type hints
def process_array(data: np.ndarray) -> np.ndarray:
    """Process numpy array."""
    ...

# Use xarray type hints
def predict(self, x: xr.Dataset) -> xr.DataArray:
    """Generate predictions."""
    ...
```

### Error Handling

```python
# Good: Specific error handling with context
try:
    result = risky_operation()
except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
    raise ModelLoadError(f"Could not load model from {path}") from e
except ValueError as e:
    logger.warning(f"Invalid input, using default: {e}")
    result = default_value

# Bad: Bare except or no error context
try:
    result = risky_operation()
except:
    print("Error occurred")
    result = None
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/
│   ├── test_models.py
│   ├── test_optimizers.py
│   └── test_strategies.py
├── integration/
│   ├── test_docker_models.py
│   └── test_end_to_end.py
├── performance/
│   └── test_benchmarks.py
└── conftest.py
```

### Writing Tests

```python
import pytest
import numpy as np
from optimizer_framework import Model, OptimizationError


class TestModel:
    """Test cases for Model class."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing."""
        return Model(config={"channels": ["tv", "digital"]})
    
    @pytest.fixture
    def sample_data(self):
        """Create sample input data."""
        return {
            "tv": np.array([100_000]),
            "digital": np.array([200_000])
        }
    
    def test_model_initialization(self):
        """Test model can be initialized with config."""
        model = Model(config={"channels": ["tv"]})
        assert model.channels == ["tv"]
    
    def test_model_predict(self, sample_model, sample_data):
        """Test model prediction."""
        result = sample_model.predict(sample_data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] > 0
    
    def test_model_invalid_input(self, sample_model):
        """Test model handles invalid input gracefully."""
        with pytest.raises(ValueError, match="Missing required channel"):
            sample_model.predict({"invalid": np.array([100])})
    
    @pytest.mark.parametrize("budget,expected", [
        ({"tv": 100_000, "digital": 200_000}, 500_000),
        ({"tv": 0, "digital": 0}, 0),
        ({"tv": 50_000, "digital": 150_000}, 400_000),
    ])
    def test_model_various_budgets(self, sample_model, budget, expected):
        """Test model with various budget allocations."""
        result = sample_model.predict(budget)
        assert abs(result - expected) < 1000  # Allow small tolerance
    
    @pytest.mark.slow
    def test_model_performance(self, sample_model, sample_data):
        """Test model performance with large dataset."""
        import time
        
        start = time.time()
        for _ in range(1000):
            sample_model.predict(sample_data)
        duration = time.time() - start
        
        assert duration < 1.0  # Should complete in under 1 second
```

### Test Coverage

Aim for at least 80% code coverage:

```bash
# Run tests with coverage
pytest --cov=optimizer_framework --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Integration Tests

```python
@pytest.mark.integration
class TestDockerIntegration:
    """Integration tests for Docker models."""
    
    @pytest.fixture(scope="class")
    def docker_model(self):
        """Start Docker model service."""
        # Start container
        container = start_test_container()
        yield DockerModel(port=8000)
        # Cleanup
        container.stop()
    
    def test_docker_model_health(self, docker_model):
        """Test Docker model health check."""
        assert docker_model.health_check()
    
    def test_docker_model_predict(self, docker_model):
        """Test Docker model prediction."""
        result = docker_model.predict({"tv": 100_000})
        assert result > 0
```

## Documentation

### Docstring Format

We use Google-style docstrings:

```python
def complex_function(
    param1: str,
    param2: List[int],
    param3: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, float]:
    """Brief one-line description of function.
    
    Longer description providing more details about what the
    function does, its purpose, and any important notes about
    its behavior or usage.
    
    Args:
        param1: Description of param1. Should explain what it is
            and any constraints or expectations.
        param2: Description of param2. Can span multiple lines
            if needed for clarity.
        param3: Description of param3. Note that it's optional.
            Defaults to None.
    
    Returns:
        A tuple containing:
            - np.ndarray: Description of first return value
            - float: Description of second return value
    
    Raises:
        ValueError: If param1 is empty or invalid
        TypeError: If param2 contains non-integer values
        RuntimeError: If computation fails
    
    Example:
        >>> result, score = complex_function(
        ...     "example",
        ...     [1, 2, 3],
        ...     {"option": "value"}
        ... )
        >>> print(score)
        0.95
        
    Note:
        Additional notes about edge cases, performance
        considerations, or related functions.
        
    See Also:
        related_function: Does something similar
        another_function: Complementary functionality
    """
```

### Documentation Updates

When adding new features:

1. Update docstrings
2. Add usage examples
3. Update relevant guides
4. Add to API reference
5. Update changelog

### Building Documentation

```bash
# Build documentation
cd docs
make html

# View locally
open build/html/index.html

# Check for broken links
make linkcheck
```

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass**
   ```bash
   pytest
   ```

2. **Check code quality**
   ```bash
   black --check src tests
   isort --check-only src tests
   flake8 src tests
   mypy src
   ```

3. **Update documentation**
   - Add/update docstrings
   - Update user guides if needed
   - Add to changelog

4. **Commit your changes**
   - Use meaningful commit messages
   - Reference issues in commits

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Added new tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changelog updated

## Related Issues
Closes #123
```

### Review Process

1. **Automated checks**: CI/CD runs tests and quality checks
2. **Code review**: At least one maintainer reviews the code
3. **Discussion**: Address feedback and make changes
4. **Approval**: Maintainer approves the PR
5. **Merge**: PR is merged to main branch

### After Merge

- Delete your feature branch
- Sync your fork with upstream
- Celebrate your contribution! 🎉

## Community

### Getting Help

- **Discussions**: Use GitHub Discussions for questions
- **Chat**: Join our Slack/Discord community
- **Issues**: Report bugs or request features

### Becoming a Maintainer

Active contributors may be invited to become maintainers. Maintainers:

- Review and merge PRs
- Triage issues
- Guide project direction
- Mentor new contributors

### Recognition

We recognize contributors in several ways:

- Contributors list in README
- Credits in release notes
- Special badges for regular contributors
- Shoutouts in community calls

## Thank You!

Your contributions make Atlas better for everyone. We appreciate your time and effort in improving the project!

If you have questions or need help, don't hesitate to reach out through:
- GitHub Issues
- Community Forums

Happy contributing! 🚀