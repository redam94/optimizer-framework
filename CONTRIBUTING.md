# Contributing to Atlas

We welcome contributions to Atlas! This document provides guidelines for contributing to the project. All contributions are subject to the contributor licensing agreement see [CONTRIBUTOR-LICENSE](CONTRIBUTOR-LICENSE.md)

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/optimizer-framework.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Set up development environment: `make install-dev`

## Development Process

### Code Style

- We use Black for code formatting (line length: 100)
- isort for import sorting
- Type hints are required for all new code
- Docstrings should follow Google style

### Testing

- Write tests for all new functionality
- Maintain or improve code coverage
- Run tests locally before submitting PR: `make test`

### Commit Messages

Follow conventional commits format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or changes
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

### Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Submit PR with clear description

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

## Questions?

Feel free to open an issue for any questions about contributing.
