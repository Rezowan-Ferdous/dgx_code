# Contributing to Universal ML/AI Platform

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🎯 Ways to Contribute

1. **Code Contributions**
   - Implement new models or algorithms
   - Add support for new datasets
   - Improve existing implementations
   - Fix bugs
   - Optimize performance

2. **Documentation**
   - Write tutorials
   - Improve API documentation
   - Add examples
   - Translate documentation

3. **Learning Resources**
   - Create learning modules
   - Add practice exercises
   - Write blog posts
   - Record video tutorials

4. **Testing**
   - Write unit tests
   - Add integration tests
   - Report bugs
   - Suggest improvements

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/dgx_code.git
cd dgx_code

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/dgx_code.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install development dependencies
pip install pytest black flake8 mypy pre-commit
```

### 3. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or bug fix branch
git checkout -b fix/bug-description
```

## 📝 Contribution Guidelines

### Code Style

We follow PEP 8 with some modifications:

```python
# Use black for formatting
black your_file.py

# Use flake8 for linting
flake8 your_file.py

# Use mypy for type checking
mypy your_file.py
```

### Code Standards

1. **Naming Conventions**
   - Classes: `PascalCase`
   - Functions/variables: `snake_case`
   - Constants: `UPPER_CASE`
   - Private methods: `_leading_underscore`

2. **Documentation**
   ```python
   def function_name(param1: int, param2: str) -> bool:
       """
       Brief description of function.

       Args:
           param1: Description of param1
           param2: Description of param2

       Returns:
           Description of return value

       Example:
           >>> function_name(1, "test")
           True
       """
       pass
   ```

3. **Type Hints**
   - Use type hints for all function signatures
   - Use `typing` module for complex types

4. **Error Handling**
   - Use specific exceptions
   - Add meaningful error messages
   - Document exceptions in docstrings

### Testing

1. **Write Tests**
   ```python
   # tests/test_your_feature.py
   import pytest
   from your_module import your_function

   def test_your_function():
       result = your_function(input_data)
       assert result == expected_output
   ```

2. **Run Tests**
   ```bash
   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=. --cov-report=html
   ```

3. **Test Requirements**
   - All new code must have tests
   - Maintain >80% code coverage
   - Tests must pass before merging

### Commit Messages

Follow conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Examples**:
```bash
git commit -m "feat(cv): add YOLOv8 object detection model"
git commit -m "fix(nlp): resolve tokenization bug in BERT"
git commit -m "docs(foundations): add linear algebra tutorial"
```

## 🎨 Adding New Features

### Adding a New Model

1. **Create Model File**
   ```python
   # domains/computer_vision/models/my_model.py
   import torch.nn as nn
   from framework.utils import MODEL_REGISTRY

   @MODEL_REGISTRY.register("MyModel")
   class MyModel(nn.Module):
       def __init__(self, num_classes, **kwargs):
           super().__init__()
           # Your implementation

       def forward(self, x):
           # Forward pass
           return x
   ```

2. **Add Configuration**
   ```yaml
   # configs/experiments/my_model_experiment.yaml
   model:
     name: "MyModel"
     num_classes: 10
     # ... other parameters
   ```

3. **Add Tests**
   ```python
   # tests/test_my_model.py
   def test_my_model():
       model = MyModel(num_classes=10)
       # Test logic
   ```

4. **Add Documentation**
   - Add docstrings
   - Create README in model directory
   - Add example usage

### Adding a New Dataset

1. **Create Dataset Class**
   ```python
   # datasets/my_dataset.py
   from torch.utils.data import Dataset
   from framework.utils import DATASET_REGISTRY

   @DATASET_REGISTRY.register("MyDataset")
   class MyDataset(Dataset):
       def __init__(self, root, split="train", **kwargs):
           # Implementation

       def __getitem__(self, idx):
           return {
               'features': features,
               'labels': labels,
               'mask': mask
           }
   ```

2. **Add Data Loader**
   - Document data format
   - Add preprocessing
   - Include data augmentation

### Adding a New Domain

1. **Create Directory Structure**
   ```bash
   mkdir -p domains/new_domain/{models,datasets,examples}
   ```

2. **Add README**
   - Domain description
   - Quick start guide
   - Examples
   - Benchmarks

3. **Integrate with Framework**
   - Update main README
   - Add to registry
   - Create configuration templates

## 🐛 Reporting Bugs

### Before Reporting

1. Check existing issues
2. Verify it's reproducible
3. Test on latest version

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Run command '...'
3. See error

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9]
- PyTorch version: [e.g., 2.0.0]
- GPU: [e.g., NVIDIA RTX 3090]

**Additional Context**
Error messages, logs, screenshots
```

## 💡 Suggesting Features

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature

**Motivation**
Why is this feature needed?

**Proposed Solution**
How should it work?

**Alternatives**
Other approaches considered

**Additional Context**
Examples, references, mockups
```

## 📚 Documentation Contributions

### Writing Documentation

1. **API Documentation**
   - Use Google-style docstrings
   - Include examples
   - Document parameters and returns

2. **Tutorials**
   - Step-by-step instructions
   - Code examples
   - Expected outputs
   - Common pitfalls

3. **README Files**
   - Clear structure
   - Quick start guide
   - Usage examples
   - Links to related docs

### Building Documentation

```bash
cd docs
make html
```

## 🔄 Pull Request Process

### 1. Update Your Fork

```bash
git fetch upstream
git rebase upstream/main
```

### 2. Make Changes

- Follow code style guidelines
- Write tests
- Update documentation
- Add comments

### 3. Test

```bash
# Run tests
pytest

# Check style
black .
flake8 .
mypy .
```

### 4. Commit

```bash
git add .
git commit -m "feat: add feature description"
```

### 5. Push

```bash
git push origin feature/your-feature-name
```

### 6. Create Pull Request

- Descriptive title
- Clear description
- Link related issues
- Add screenshots if applicable

### 7. Code Review

- Address review comments
- Update PR as needed
- Be responsive

### 8. Merge

Once approved, your PR will be merged!

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

## 📜 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Give constructive feedback
- Focus on what's best for the community
- Show empathy

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Personal attacks
- Publishing private information
- Other unprofessional conduct

## 📧 Contact

Questions? Reach out:
- Open an issue
- Email: contributors@example.com
- Discord: [Link]

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to the Universal ML/AI Platform!** 🎉
