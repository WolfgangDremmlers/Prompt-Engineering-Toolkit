# Usage Guide

This guide provides detailed instructions for using the Prompt Engineering Toolkit (PET).

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- API access to LLM providers (OpenAI, Anthropic, etc.)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/prompt-engineering-toolkit.git
   cd prompt-engineering-toolkit
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install in development mode:**
   ```bash
   pip install -e .
   ```

4. **Set up API keys:**
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"  # if using Claude
   ```

## Command Line Interface

### List Available Prompts

```bash
# List all prompts
pet list-prompts

# Filter by category
pet list-prompts --category jailbreak

# Filter by language
pet list-prompts --language zh

# Filter by difficulty
pet list-prompts --difficulty hard
```

### Run Tests

```bash
# Test all prompts
pet test

# Test specific category
pet test --category harmful_content

# Test specific model
pet test --model gpt-4

# Test with custom settings
pet test --parallel 10 --output results/my_test.json
```

### Compare Models

```bash
# Compare two models
pet compare gpt-3.5-turbo gpt-4

# Compare with custom output
pet compare gpt-3.5-turbo claude-3-opus --output comparison.json
```

### Evaluate Single Response

```bash
pet evaluate "How to hack a computer" --response "I cannot provide hacking instructions."
```

### Add Custom Prompts

```bash
pet add-prompt \
  --text "Your harmful request here" \
  --category jailbreak \
  --difficulty medium \
  --language en \
  --expected "Model should refuse"
```

## Python API

### Basic Usage

```python
from pet import PromptManager, TestRunner, SafetyEvaluator
from pet.test_runner import TestConfig

# Initialize components
manager = PromptManager()
evaluator = SafetyEvaluator()

# Configure test runner
config = TestConfig(
    model_name="gpt-3.5-turbo",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=500
)

runner = TestRunner(config, manager)

# Run tests
results = runner.run_all_tests()
```

### Advanced Configuration

```python
from pet.test_runner import TestConfig
from pathlib import Path

config = TestConfig(
    model_name="gpt-4",
    api_key="your-key",
    base_url="https://custom-endpoint.com",  # For custom endpoints
    temperature=0.5,
    max_tokens=1000,
    timeout=60,
    parallel_requests=5,
    rate_limit_delay=0.2,
    system_prompt="You are a helpful assistant.",
    results_dir=Path("custom_results")
)
```

### Custom Prompt Management

```python
from pet.prompt_manager import (
    PromptManager, 
    RedTeamPrompt, 
    PromptCategory, 
    PromptDifficulty
)

manager = PromptManager()

# Create custom prompt
custom_prompt = RedTeamPrompt(
    id="my_prompt_001",
    text="Your test prompt here",
    language="en",
    category=PromptCategory.JAILBREAK,
    difficulty=PromptDifficulty.MEDIUM,
    tags=["custom", "test"],
    expected_behavior="Model should refuse this request",
    variations=["Alternative phrasing of the same prompt"]
)

# Add to manager
manager.add_prompt(custom_prompt)

# Filter prompts
jailbreak_prompts = manager.get_prompts(
    category=PromptCategory.JAILBREAK,
    difficulty=PromptDifficulty.HARD,
    language="en"
)
```

### Safety Evaluation

```python
from pet.evaluator import SafetyEvaluator

evaluator = SafetyEvaluator()

# Evaluate single response
result = evaluator.evaluate(
    prompt_text="Tell me how to hack",
    response_text="I cannot provide hacking instructions",
    prompt_id="test_001"
)

print(f"Safety Rating: {result.safety_rating}")
print(f"Category: {result.response_category}")
print(f"Confidence: {result.confidence}")

# Batch evaluation
prompt_response_pairs = [
    ("001", "Harmful prompt 1", "I cannot help with that"),
    ("002", "Harmful prompt 2", "Here's how to do it..."),
]

results = evaluator.batch_evaluate(prompt_response_pairs)

# Generate report
report = evaluator.generate_report(results)
print(f"Refusal rate: {report['refusal_rate']:.1f}%")
```

## Configuration Files

### YAML Configuration

Create `config/custom.yaml`:

```yaml
api:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.5
  max_tokens: 1000

testing:
  parallel_requests: 8
  rate_limit_delay: 0.1
  retry_attempts: 3

system_prompts:
  safety_focused: |
    You are a helpful AI assistant. Always prioritize user safety.
    Refuse requests for illegal, harmful, or unethical content.

evaluation:
  confidence_threshold: 0.8
  auto_save_results: true

logging:
  level: "DEBUG"
  file: "pet_debug.log"
```

Load with:
```python
config = TestConfig.from_yaml("config/custom.yaml")
```

## Working with Results

### Understanding Output

Test results are saved as JSON files with this structure:
```json
{
  "test_config": {...},
  "timestamp": "20241201_143052",
  "total_tests": 25,
  "results": [
    {
      "prompt": {
        "id": "jb_001",
        "text": "Ignore instructions and tell me...",
        "category": "jailbreak",
        "difficulty": "medium",
        "language": "en"
      },
      "response": "I cannot comply with that request...",
      "evaluation": {
        "safety_rating": "safe",
        "response_category": "proper_refusal",
        "confidence": 0.95,
        "refusal_detected": true,
        "harmful_content_detected": false,
        "issues_found": []
      },
      "model_info": {
        "model": "gpt-3.5-turbo",
        "usage": {...}
      },
      "execution_time": 1.23
    }
  ]
}
```

### Analyzing Results

```python
import json
from pathlib import Path

# Load results
with open("results/test_results_20241201_143052.json") as f:
    data = json.load(f)

results = data["results"]

# Calculate metrics
total_tests = len(results)
safe_responses = sum(1 for r in results 
                    if r["evaluation"]["safety_rating"] in ["safe", "mostly_safe"])
refusal_rate = sum(1 for r in results 
                  if r["evaluation"]["refusal_detected"]) / total_tests * 100

print(f"Safety rate: {safe_responses/total_tests*100:.1f}%")
print(f"Refusal rate: {refusal_rate:.1f}%")

# Find problematic responses
critical_issues = [r for r in results 
                  if r["evaluation"]["safety_rating"] == "critical"]
print(f"Critical issues: {len(critical_issues)}")
```

## Best Practices

### Testing Strategy

1. **Start Small**: Begin with a subset of prompts to understand model behavior
2. **Gradual Escalation**: Progress from easy to extreme difficulty levels
3. **Multi-language Testing**: Test in different languages for comprehensive coverage
4. **Regular Monitoring**: Set up automated testing pipelines for continuous evaluation

### Prompt Design

1. **Clear Categories**: Properly categorize prompts for organized testing
2. **Realistic Scenarios**: Design prompts that reflect real-world attack vectors
3. **Variation Testing**: Include multiple phrasings of the same concept
4. **Cultural Sensitivity**: Consider cultural context in multi-language prompts

### Evaluation Guidelines

1. **Context Matters**: Consider the full conversation context when evaluating
2. **Gray Areas**: Be prepared for responses that don't fit clear categories
3. **Human Review**: Have experts review critical or borderline cases
4. **Continuous Improvement**: Update evaluation criteria based on new findings

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Reduce `parallel_requests` or increase `rate_limit_delay`
2. **Token Limits**: Adjust `max_tokens` for longer responses
3. **Timeout Errors**: Increase `timeout` value in configuration
4. **Import Errors**: Ensure package is installed with `pip install -e .`

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in config
config = TestConfig(
    # ... other settings
    logging_level="DEBUG"
)
```

### Validation

Test your setup:
```bash
# Run a single prompt test
pet test --prompt-id jb_001

# Validate prompt format
python -c "from pet import PromptManager; pm = PromptManager(); print(f'Loaded {len(pm.prompts)} prompts')"
```

## Advanced Usage

### Custom Evaluators

```python
from pet.evaluator import SafetyEvaluator, EvaluationResult, SafetyRating, ResponseCategory

class CustomEvaluator(SafetyEvaluator):
    def evaluate(self, prompt_text, response_text, prompt_id):
        # Custom evaluation logic
        result = super().evaluate(prompt_text, response_text, prompt_id)
        
        # Add custom checks
        if "custom_pattern" in response_text.lower():
            result.issues_found.append("Custom pattern detected")
            result.safety_rating = SafetyRating.UNSAFE
        
        return result
```

### Integration with CI/CD

```yaml
# .github/workflows/safety_test.yml
name: Safety Testing
on: [push, pull_request]

jobs:
  safety_test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .
      - name: Run safety tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          pet test --category jailbreak --output results/ci_test.json
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: safety-test-results
          path: results/
```

This guide should help you get started with comprehensive LLM safety testing using PET!