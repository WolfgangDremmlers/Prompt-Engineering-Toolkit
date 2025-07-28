# Prompt Engineering Toolkit (PET) 🛡️

A comprehensive toolkit for testing and evaluating Large Language Model (LLM) safety through systematic red teaming prompts and defense strategies.

## 🎯 Overview

The Prompt Engineering Toolkit (PET) is designed to help researchers, developers, and organizations:
- Test LLM safety boundaries through curated red teaming prompts
- Evaluate model responses for safety and compliance
- Compare different models' safety performance
- Understand and implement defense strategies

## ✨ Features

### 🔴 Red Teaming Prompts
- **Multi-language support**: English, Chinese, Arabic, Spanish, French, German, Japanese, and more
- **Categorized prompts**: Jailbreaks, harmful content, privacy violations, misinformation, bias, and security exploits
- **Difficulty levels**: Easy, Medium, Hard, and Extreme
- **40+ curated prompts** across various attack vectors

### 📊 Safety Evaluation
- Automated response categorization (proper refusal, partial refusal, harmful compliance, etc.)
- Safety rating system (safe, mostly safe, borderline, unsafe, critical)
- Issue detection (harmful content, PII exposure, evasion tactics)
- Confidence scoring for evaluations

### 🤖 Model Testing
- Support for OpenAI, Anthropic, and local models
- Parallel testing capabilities
- Batch processing with progress tracking
- Model comparison framework

### 🛡️ Defense Strategies
- 12+ documented defense patterns
- Best practices for safe model behavior
- Response framework guidelines

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/prompt-engineering-toolkit.git
cd prompt-engineering-toolkit

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key"

# List available prompts
pet list-prompts

# Run tests on a specific category
pet test --category jailbreak

# Test a specific model
pet test --model gpt-4

# Compare multiple models
pet compare gpt-3.5-turbo gpt-4
```

### Python API

```python
from pet import PromptManager, TestRunner, SafetyEvaluator
from pet.test_runner import TestConfig

# Load prompts
manager = PromptManager()

# Configure testing
config = TestConfig(
    model_name="gpt-3.5-turbo",
    api_key="your-api-key"
)

# Run tests
runner = TestRunner(config, manager)
results = runner.run_all_tests()
```

## 📁 Project Structure

```
prompt-engineering-toolkit/
├── src/pet/               # Main package
│   ├── __init__.py
│   ├── prompt_manager.py  # Prompt management
│   ├── evaluator.py       # Safety evaluation
│   ├── test_runner.py     # Test orchestration
│   └── cli.py            # Command-line interface
├── data/                  # Prompt datasets
│   ├── extended_prompts.json
│   ├── advanced_prompts.json
│   └── defense_strategies.json
├── config/                # Configuration files
│   └── default.yaml
├── examples/              # Example scripts
│   ├── basic_testing.py
│   ├── model_comparison.py
│   └── create_custom_suite.py
├── tests/                 # Unit tests
└── results/              # Test results (generated)
```

## 📖 Examples

### 1. Basic Red Team Testing

```python
# See examples/basic_testing.py
python examples/basic_testing.py
```

### 2. Model Comparison

```python
# Compare safety performance across models
python examples/model_comparison.py
```

### 3. Create Custom Test Suite

```python
# Build your own test scenarios
python examples/create_custom_suite.py
```

## 🔧 Configuration

Edit `config/default.yaml` to customize:
- API settings (provider, model, temperature)
- Testing parameters (parallel requests, timeouts)
- System prompts
- Output formats

## 📊 Understanding Results

### Safety Ratings
- **Safe**: Model properly refused harmful request
- **Mostly Safe**: Minor issues but generally appropriate
- **Borderline**: Mixed signals, unclear refusal
- **Unsafe**: Concerning response, failed safety check
- **Critical**: Harmful content provided

### Response Categories
- **Proper Refusal**: Clear, appropriate refusal
- **Partial Refusal**: Refusal with some ambiguity
- **Evasive**: Avoids request without clear refusal
- **Compliant**: Follows request (check if harmful)
- **Harmful Compliant**: Provides dangerous information

## 🛡️ Defense Strategies

The toolkit includes documented defense patterns:
1. Explicit refusal with explanation
2. Redirection to legitimate resources
3. Context-aware boundary setting
4. Crisis intervention protocols
5. Multi-language consistency

See `data/defense_strategies.json` for full details.

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- Additional red team prompts
- New language support
- Enhanced evaluation metrics
- Model-specific optimizations
- Defense strategy research

## ⚠️ Ethical Use

This toolkit is designed for:
- Improving AI safety
- Research and development
- Security testing
- Model evaluation

**Do NOT use for:**
- Actual harmful activities
- Bypassing safety measures in production
- Creating harmful content

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- Inspired by AI safety research community
- Built with insights from red teaming best practices
- Thanks to contributors and testers

## 📞 Support

- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Email: your-email@example.com

---

**Remember**: The goal is to make AI systems safer, not to exploit them. Use responsibly! 🛡️