# Contributing to Video Penibility Assessment

We welcome contributions to the Video Penibility Assessment framework! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/video-penibility-assessment.git
   cd video-penibility-assessment
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## 🔧 Development Setup

### Code Style

We follow PEP 8 guidelines and use the following tools:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Format your code before submitting:
```bash
black src/ tests/ scripts/
flake8 src/ tests/ scripts/
mypy src/
```

### Testing

Run the test suite:
```bash
pytest tests/
```

For coverage reports:
```bash
pytest --cov=video_penibility tests/
```

## 📝 Contribution Guidelines

### Code Contributions

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write clean, documented code**:
   - Follow PEP 8 style guidelines
   - Add type hints to all functions
   - Write comprehensive docstrings (Google style)
   - Include unit tests for new functionality

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** on GitHub

### Commit Message Format

We use conventional commits:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code refactoring
- `test:` for adding tests
- `chore:` for maintenance tasks

### Documentation

- Update README.md if you add new features
- Add docstrings to all public functions and classes
- Update configuration documentation for new parameters
- Add examples for new functionality

### Adding New Models

To add a new model architecture:

1. Create a new file in `src/video_penibility/models/`
2. Inherit from `BaseModel`
3. Implement the `forward` method
4. Register the model in `ModelFactory`
5. Add configuration options to `ModelConfig`
6. Write unit tests
7. Update documentation

Example:
```python
from .base import BaseModel

class YourModel(BaseModel):
    def __init__(self, input_dim: int, **kwargs):
        super().__init__(input_dim=input_dim, **kwargs)
        # Your model implementation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your forward pass
        pass

# Register in factory
ModelFactory.register_model("your_model", YourModel)
```

### Adding New Features

For new feature types:

1. Update `DataConfig.features_type` in `config/schema.py`
2. Add feature loading logic in the dataset classes
3. Update input dimension mapping
4. Add tests and documentation

## 🐛 Bug Reports

When reporting bugs, please include:

- Python version
- PyTorch version
- Operating system
- Complete error traceback
- Minimal code example to reproduce the issue
- Expected vs. actual behavior

## 💡 Feature Requests

For feature requests:

- Describe the use case
- Explain why the feature would be beneficial
- Provide examples of how it would be used
- Consider implementation complexity

## 📋 Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows PEP 8 style guidelines
- [ ] All tests pass
- [ ] New functionality includes tests
- [ ] Documentation is updated
- [ ] Type hints are added
- [ ] Commit messages follow conventional format
- [ ] No merge conflicts with main branch

## 🤝 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Focus on the technical aspects

## 📞 Getting Help

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Check existing issues before creating new ones

Thank you for contributing to Video Penibility Assessment! 🎉 