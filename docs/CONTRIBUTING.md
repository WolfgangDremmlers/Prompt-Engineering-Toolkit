# Contributing to Prompt Engineering Toolkit

Thank you for your interest in contributing to the Prompt Engineering Toolkit (PET)! This document provides guidelines and information for contributors.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Review Process](#review-process)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background, experience level, or identity. We expect all community members to:

- Be respectful and considerate
- Focus on constructive feedback
- Accept responsibility for mistakes
- Prioritize the community's best interests

### Unacceptable Behavior

- Harassment, discrimination, or offensive language
- Personal attacks or trolling
- Sharing inappropriate content
- Violating others' privacy

### Enforcement

Report any violations to the project maintainers. All reports will be handled confidentially and fairly.

## Getting Started

### Areas for Contribution

We welcome contributions in several areas:

1. **Red Team Prompts**: Adding new, ethical red team prompts for testing
2. **Evaluation Logic**: Improving safety evaluation algorithms
3. **Multi-language Support**: Adding support for new languages
4. **API Integrations**: Supporting additional LLM providers
5. **Documentation**: Improving guides, examples, and API documentation
6. **Testing**: Adding test cases and improving test coverage
7. **Performance**: Optimizing processing speed and memory usage
8. **Bug Fixes**: Resolving reported issues

### Before You Start

1. **Check existing issues**: Look for related work or discussions
2. **Read the documentation**: Understand the project structure and goals
3. **Join discussions**: Participate in issue discussions before starting work
4. **Start small**: Begin with minor fixes or improvements

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- A GitHub account
- OpenAI API key (for testing)

### Setup Instructions

1. **Fork the repository**:
   ```bash
   # Visit https://github.com/WolfgangDremmler/prompt-engineering-toolkit
   # Click "Fork" to create your own copy
   ```

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/prompt-engineering-toolkit.git
   cd prompt-engineering-toolkit
   ```

3. **Set up development environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   
   # Install development dependencies
   pip install -e ".[dev]"
   
   # Install pre-commit hooks
   pre-commit install
   ```

4. **Configure environment**:
   ```bash
   # Copy example configuration
   cp config/example.yaml ~/.pet/config.yaml
   
   # Set API key for testing
   export OPENAI_API_KEY="your-test-api-key"
   ```

5. **Verify setup**:
   ```bash
   # Run tests
   pytest
   
   # Check code style
   black --check src/ tests/
   flake8 src/ tests/
   
   # Type checking
   mypy src/
   ```

### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**:
   ```bash
   # Run full test suite
   pytest
   
   # Run specific tests
   pytest tests/test_your_feature.py
   
   # Check coverage
   pytest --cov=pet --cov-report=html
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

## Contributing Guidelines

### Code Style

We follow Python community standards:

#### Formatting
- **Black**: Automatic code formatting
- **Line length**: 88 characters (Black default)
- **Import sorting**: Use `isort`

#### Naming Conventions
- **Classes**: PascalCase (`RedTeamPrompt`)
- **Functions/variables**: snake_case (`get_prompts`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_MODEL`)
- **Private members**: Leading underscore (`_internal_method`)

#### Documentation
- **Docstrings**: Use Google style docstrings
- **Type hints**: Required for all public APIs
- **Comments**: Explain why, not what

Example:
```python
from typing import List, Optional
from pet.models import RedTeamPrompt, PromptCategory

class PromptManager:
    """Manages red team prompts with filtering and search capabilities.
    
    This class provides methods for loading, filtering, and managing
    a database of red team prompts used for LLM safety testing.
    
    Attributes:
        database: The underlying prompt database
    """
    
    def get_prompts(
        self,
        category: Optional[PromptCategory] = None,
        limit: Optional[int] = None
    ) -> List[RedTeamPrompt]:
        """Get prompts with optional filtering.
        
        Args:
            category: Filter by prompt category
            limit: Maximum number of prompts to return
            
        Returns:
            List of matching prompts
            
        Raises:
            ValueError: If limit is negative
        """
        if limit is not None and limit < 0:
            raise ValueError("Limit must be non-negative")
            
        # Implementation here...
```

### Adding Red Team Prompts

When contributing new prompts, ensure they are:

#### Ethical and Responsible
- **Purpose**: Only for legitimate safety testing
- **Harmful content**: Avoid gratuitously offensive material
- **Legal compliance**: Don't include illegal content
- **Research focused**: Contribute to AI safety research

#### High Quality
- **Realistic**: Represent actual attack vectors
- **Diverse**: Cover different categories and techniques
- **Well-categorized**: Proper category and difficulty assignment
- **Multilingual**: Support for non-English languages appreciated

#### Proper Format

Use the standard JSON format:

```json
{
  "version": "2.0.0",
  "description": "Description of prompt set",
  "prompts": [
    {
      "id": "unique_identifier_001",
      "text": "The actual prompt text",
      "language": "en",
      "category": "jailbreak",
      "difficulty": "medium",
      "expected_behavior": "Should refuse the request",
      "tags": ["tag1", "tag2"],
      "source": "research_paper_2024",
      "verified": true,
      "metadata": {
        "technique": "roleplaying",
        "target": "content_filter"
      }
    }
  ]
}
```

#### Verification Process
1. **Test prompts**: Verify they work as expected
2. **Peer review**: Have others review for appropriateness
3. **Documentation**: Include source and rationale
4. **Multiple examples**: Provide diverse test cases

### API Changes

For changes to public APIs:

1. **Backward compatibility**: Maintain compatibility when possible
2. **Deprecation warnings**: Use for removed features
3. **Version bumping**: Follow semantic versioning
4. **Migration guides**: Document breaking changes

### New Features

When adding new features:

1. **Issue discussion**: Discuss design in GitHub issues first
2. **Modular design**: Keep features independent when possible
3. **Configuration**: Make features configurable
4. **Error handling**: Robust error handling and user feedback
5. **Documentation**: Comprehensive documentation and examples

## Testing

### Test Structure

We use pytest with the following test organization:

```
tests/
├── conftest.py           # Test fixtures and configuration
├── unit/                 # Unit tests
│   ├── test_models.py
│   ├── test_prompt_manager.py
│   └── test_evaluator.py
├── integration/          # Integration tests
│   ├── test_api_calls.py
│   └── test_full_workflow.py
└── fixtures/             # Test data
    ├── sample_prompts.json
    └── expected_results.json
```

### Test Categories

Use pytest markers to categorize tests:

```python
import pytest

@pytest.mark.unit
def test_prompt_creation():
    """Unit test for prompt creation"""
    pass

@pytest.mark.integration
def test_full_workflow():
    """Integration test for complete workflow"""
    pass

@pytest.mark.slow
def test_large_dataset():
    """Slow test for large datasets"""
    pass

@pytest.mark.api
def test_openai_integration():
    """Test requiring API access"""
    pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific categories
pytest -m unit
pytest -m integration
pytest -m "not slow"

# Run with coverage
pytest --cov=pet --cov-report=html

# Run tests requiring API access
pytest -m api --api-key="your-key"
```

### Writing Tests

#### Test Naming
- Use descriptive names: `test_get_prompts_filters_by_category`
- Follow pattern: `test_[method]_[scenario]_[expected_result]`

#### Test Structure
Use the Arrange-Act-Assert pattern:

```python
def test_prompt_manager_filters_by_category():
    # Arrange
    manager = PromptManager(data_dir=temp_dir, auto_load=False)
    jailbreak_prompt = create_test_prompt(category=PromptCategory.JAILBREAK)
    harmful_prompt = create_test_prompt(category=PromptCategory.HARMFUL_CONTENT)
    manager.add_prompt(jailbreak_prompt)
    manager.add_prompt(harmful_prompt)
    
    # Act
    filtered_prompts = manager.get_prompts(category=PromptCategory.JAILBREAK)
    
    # Assert
    assert len(filtered_prompts) == 1
    assert filtered_prompts[0].category == PromptCategory.JAILBREAK
```

#### Fixtures
Use fixtures for common test data:

```python
@pytest.fixture
def sample_prompt():
    return RedTeamPrompt(
        text="Test prompt",
        category=PromptCategory.JAILBREAK,
        difficulty=PromptDifficulty.MEDIUM
    )

def test_something(sample_prompt):
    # Use the fixture
    assert sample_prompt.category == PromptCategory.JAILBREAK
```

### Mocking

Mock external dependencies in tests:

```python
from unittest.mock import Mock, patch

@patch('pet.core.test_runner.openai.OpenAI')
def test_api_call_handling(mock_openai):
    # Setup mock
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices[0].message.content = "Test response"
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    # Test code that uses OpenAI API
    runner = TestRunner(config, prompt_manager)
    result = runner.run_single(test_prompt)
    
    # Verify mock was called correctly
    mock_client.chat.completions.create.assert_called_once()
```

## Documentation

### Types of Documentation

1. **API Documentation**: Docstrings for all public methods
2. **User Guides**: Step-by-step tutorials and guides
3. **Examples**: Working code examples
4. **README**: Project overview and quick start
5. **Changelog**: Record of changes between versions

### Documentation Standards

#### Docstrings
Use Google-style docstrings:

```python
def process_prompts(
    self,
    prompts: List[RedTeamPrompt],
    batch_size: int = 10
) -> List[TestResult]:
    """Process a list of prompts in batches.
    
    Processes the given prompts by sending them to the configured
    LLM API and evaluating the responses for safety concerns.
    
    Args:
        prompts: List of prompts to process
        batch_size: Number of prompts to process concurrently
        
    Returns:
        List of test results containing prompt, response, and evaluation
        
    Raises:
        APIError: If API calls fail
        ConfigurationError: If configuration is invalid
        
    Example:
        >>> manager = PromptManager()
        >>> runner = TestRunner(config, manager)
        >>> prompts = manager.get_prompts(limit=5)
        >>> results = runner.process_prompts(prompts)
        >>> print(f"Processed {len(results)} prompts")
    """
```

#### Markdown Documentation
- Use clear headings and structure
- Include code examples
- Link between related documents
- Keep examples up-to-date

### Building Documentation

We use Sphinx for API documentation:

```bash
# Install doc dependencies
pip install -e ".[docs]"

# Build HTML documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## Submitting Changes

### Pull Request Process

1. **Create Pull Request**:
   - Use descriptive title and description
   - Reference related issues
   - Include testing information
   - Add screenshots for UI changes

2. **Pull Request Template**:
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
   - [ ] Manual testing completed
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] Tests added/updated
   ```

3. **Commit Message Format**:
   Follow Conventional Commits:
   ```
   type(scope): description
   
   [optional body]
   
   [optional footer]
   ```
   
   Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
   
   Examples:
   ```
   feat(prompts): add multilingual jailbreak prompts
   
   fix(evaluator): handle edge case in safety rating calculation
   
   docs(api): update prompt manager documentation
   ```

### Branch Naming

Use descriptive branch names:
- `feature/multilingual-support`
- `fix/evaluation-edge-case`
- `docs/update-installation-guide`
- `refactor/prompt-manager-cleanup`

## Review Process

### What We Look For

1. **Functionality**: Does it work as intended?
2. **Code Quality**: Is it well-written and maintainable?
3. **Testing**: Are there adequate tests?
4. **Documentation**: Is it properly documented?
5. **Performance**: Does it introduce performance issues?
6. **Security**: Are there security implications?
7. **Compatibility**: Does it maintain backward compatibility?

### Review Timeline

- **Initial review**: Within 2-3 business days
- **Follow-up reviews**: Within 1-2 business days
- **Merge**: After approval and CI passes

### Addressing Feedback

1. **Respond promptly**: Address feedback within a few days
2. **Ask questions**: Clarify unclear feedback
3. **Test changes**: Verify fixes work correctly
4. **Update documentation**: Keep docs current with changes

## Recognition

### Contributors

All contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- GitHub contributors page

### Types of Contributions

We value all types of contributions:
- Code contributions
- Documentation improvements
- Bug reports and feature requests
- Testing and quality assurance
- Community support and discussions

## Getting Help

### Communication Channels

- **GitHub Issues**: For bugs, features, and discussions
- **GitHub Discussions**: For questions and community chat
- **Email**: For sensitive issues or private communications

### Mentorship

New contributors can:
- Look for "good first issue" labels
- Ask questions in issues or discussions
- Request mentorship from maintainers
- Participate in code reviews to learn

## License

By contributing to PET, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing to the Prompt Engineering Toolkit! Your efforts help make AI systems safer for everyone.