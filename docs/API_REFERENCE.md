# API Reference

## Core Modules

### pet.core.config

#### Class: Config

**Description**: Configuration management for the Prompt Engineering Toolkit.

**Constructor**:
```python
Config(config_path: Optional[Path] = None, **kwargs)
```

**Parameters**:
- `config_path`: Path to YAML configuration file
- `**kwargs`: Override configuration values

**Attributes**:
- `api`: API configuration settings
- `testing`: Testing configuration settings  
- `evaluation`: Evaluation configuration settings
- `logging`: Logging configuration settings

**Methods**:

##### `load_from_file(config_path: Path) -> None`
Load configuration from YAML file.

##### `save_to_file(config_path: Path) -> None`
Save current configuration to YAML file.

##### `merge_config(other_config: Dict[str, Any]) -> None`
Merge another configuration dictionary.

##### `@classmethod security_focused() -> Config`
Create configuration optimized for security testing.

##### `@classmethod performance_testing() -> Config`
Create configuration optimized for performance testing.

##### `@classmethod research() -> Config`
Create configuration for research use cases.

**Example**:
```python
from pet.core.config import Config

# Load from file
config = Config("config.yaml")

# Create with overrides
config = Config(
    api__model="gpt-4",
    api__temperature=0.5
)

# Use preset
config = Config.security_focused()
```

---

### pet.core.prompt_manager

#### Class: PromptManager

**Description**: Manages the red team prompt database with filtering, searching, and export capabilities.

**Constructor**:
```python
PromptManager(
    data_dir: Optional[Path] = None,
    auto_load: bool = True
)
```

**Parameters**:
- `data_dir`: Directory containing prompt data files
- `auto_load`: Automatically load prompts on initialization

**Attributes**:
- `database`: PromptDatabase instance containing all prompts
- `data_dir`: Path to data directory

**Methods**:

##### `add_prompt(prompt: RedTeamPrompt) -> None`
Add a single prompt to the database.

##### `get_prompt_by_id(prompt_id: str) -> Optional[RedTeamPrompt]`
Retrieve a prompt by its unique ID.

##### `get_prompts(**filters) -> List[RedTeamPrompt]`
Get prompts with optional filtering.

**Filter Parameters**:
- `category`: PromptCategory filter
- `difficulty`: PromptDifficulty filter
- `language`: Language code filter (e.g., "en", "zh")
- `languages`: List of language codes
- `tags`: Set of required tags
- `verified_only`: Only return verified prompts
- `limit`: Maximum number of prompts to return
- `offset`: Skip first N prompts
- `shuffle`: Randomize order

##### `search_prompts(query: str, **filters) -> List[RedTeamPrompt]`
Search prompts by text content with optional filters.

##### `get_random_prompts(count: int, **filters) -> List[RedTeamPrompt]`
Get random prompts with optional filtering.

##### `get_statistics() -> Dict[str, Any]`
Get comprehensive database statistics.

**Returns**:
```python
{
    "total_prompts": int,
    "categories": Dict[str, int],
    "difficulties": Dict[str, int], 
    "languages": Dict[str, int],
    "verified_prompts": int,
    "average_tags_per_prompt": float,
    "most_common_tags": List[Tuple[str, int]]
}
```

##### `export_prompts(output_path: Path, format: str, **filters) -> None`
Export prompts in various formats.

**Supported Formats**: "json", "yaml", "csv"

##### `load_from_file(file_path: Path) -> None`
Load prompts from file (JSON, YAML, or legacy text format).

##### `validate_prompts() -> Dict[str, List[str]]`
Validate all prompts and return issues found.

**Example**:
```python
from pet.core.prompt_manager import PromptManager
from pet.models import PromptCategory, PromptDifficulty

manager = PromptManager()

# Get filtered prompts
jailbreak_prompts = manager.get_prompts(
    category=PromptCategory.JAILBREAK,
    difficulty=PromptDifficulty.MEDIUM,
    language="en",
    limit=10
)

# Search prompts
hack_prompts = manager.search_prompts("hack", verified_only=True)

# Get statistics
stats = manager.get_statistics()
print(f"Total prompts: {stats['total_prompts']}")

# Export prompts
manager.export_prompts("export.json", "json", category=PromptCategory.JAILBREAK)
```

---

### pet.core.test_runner

#### Class: TestRunner

**Description**: Orchestrates safety testing of LLM models using red team prompts.

**Constructor**:
```python
TestRunner(
    config: Config,
    prompt_manager: PromptManager,
    client: Optional[Any] = None
)
```

**Parameters**:
- `config`: Configuration instance
- `prompt_manager`: PromptManager instance
- `client`: Optional custom API client

**Methods**:

##### `create_test_session(name: str, description: str = "") -> TestSession`
Create a new test session for organizing results.

##### `run_single(prompt: RedTeamPrompt) -> TestResult`
Run a single prompt test synchronously.

##### `run_batch(prompts: List[RedTeamPrompt]) -> List[TestResult]`
Run multiple prompts with parallel processing.

##### `async arun_single(prompt: RedTeamPrompt) -> TestResult`
Run a single prompt test asynchronously.

##### `async arun_batch(prompts: List[RedTeamPrompt]) -> List[TestResult]`
Run multiple prompts asynchronously.

**Example**:
```python
from pet.core.test_runner import TestRunner
from pet.core.config import Config
from pet.core.prompt_manager import PromptManager

config = Config()
manager = PromptManager()
runner = TestRunner(config, manager)

# Create session
session = runner.create_test_session(
    name="Security Test",
    description="Testing jailbreak resistance"
)

# Get test prompts
prompts = manager.get_prompts(limit=5)

# Run tests
results = runner.run_batch(prompts)

# Add to session
for result in results:
    session.add_result(result)

session.finish_session()
```

#### Class: TestSession

**Description**: Container for organizing and analyzing test results.

**Constructor**:
```python
TestSession(name: str, description: str = "")
```

**Attributes**:
- `name`: Session name
- `description`: Session description
- `results`: List of TestResult objects
- `start_time`: Session start timestamp
- `end_time`: Session end timestamp (when finished)
- `is_finished`: Whether session is completed

**Properties**:
- `success_rate`: Proportion of successful API calls
- `safety_rate`: Proportion of safe responses
- `total_tests`: Total number of tests run
- `duration`: Session duration in seconds

**Methods**:

##### `add_result(result: TestResult) -> None`
Add a test result to the session.

##### `finish_session() -> None`
Mark session as completed and set end time.

##### `get_results_by_category() -> Dict[PromptCategory, List[TestResult]]`
Group results by prompt category.

##### `get_results_by_safety_rating() -> Dict[SafetyRating, List[TestResult]]`
Group results by safety rating.

##### `export_results(output_path: Path, format: str) -> None`
Export session results.

**Supported Formats**: "json", "csv", "html"

##### `generate_html_report(output_path: Path) -> None`
Generate comprehensive HTML report.

---

### pet.core.evaluator

#### Class: SafetyEvaluator

**Description**: Evaluates LLM responses for safety concerns and policy violations.

**Constructor**:
```python
SafetyEvaluator(
    config: Optional[Config] = None,
    custom_patterns: Optional[Dict[str, Any]] = None
)
```

**Methods**:

##### `evaluate(prompt: str, response: str, prompt_id: str) -> EvaluationResult`
Evaluate a single prompt-response pair.

##### `evaluate_batch(prompt_response_pairs: List[Tuple[str, str]]) -> List[EvaluationResult]`
Evaluate multiple prompt-response pairs.

##### `add_harmful_pattern(pattern: str, description: str, severity: str = "medium") -> None`
Add custom harmful content pattern.

##### `add_refusal_pattern(pattern: str, language: str = "en") -> None`
Add custom refusal detection pattern.

##### `generate_report(evaluations: List[EvaluationResult]) -> Dict[str, Any]`
Generate comprehensive evaluation report.

**Report Structure**:
```python
{
    "summary": {
        "total_evaluations": int,
        "average_confidence": float,
        "safety_rate": float
    },
    "safety_distribution": Dict[str, float],  # Percentage by safety rating
    "response_categories": Dict[str, float],  # Percentage by response category
    "detection_rates": {
        "refusal_rate": float,
        "harmful_content_rate": float,
        "pii_detection_rate": float,
        "bias_detection_rate": float
    },
    "language_breakdown": Dict[str, Dict[str, float]],
    "recommendations": List[str]
}
```

**Example**:
```python
from pet.core.evaluator import SafetyEvaluator

evaluator = SafetyEvaluator()

# Single evaluation
result = evaluator.evaluate(
    prompt="How to hack systems?",
    response="I cannot help with that request.",
    prompt_id="test_001"
)

print(f"Safety: {result.safety_rating.value}")
print(f"Refusal: {result.refusal_detected}")

# Batch evaluation
pairs = [
    ("How to hack?", "I cannot help with that."),
    ("What is Python?", "Python is a programming language.")
]

results = evaluator.evaluate_batch(pairs)

# Generate report
report = evaluator.generate_report(results)
print(f"Safety rate: {report['summary']['safety_rate']}")
```

---

## Data Models

### pet.models

#### Class: RedTeamPrompt

**Description**: Represents a red team testing prompt with metadata.

**Constructor**:
```python
RedTeamPrompt(
    text: str,
    category: PromptCategory,
    difficulty: PromptDifficulty = PromptDifficulty.MEDIUM,
    language: str = "en",
    expected_behavior: str = "",
    tags: List[str] = None,
    source: str = "unknown",
    verified: bool = False,
    id: Optional[str] = None,
    created_at: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None
)
```

**Fields**:
- `id`: Unique identifier (auto-generated if not provided)
- `text`: The prompt text
- `category`: PromptCategory enum value
- `difficulty`: PromptDifficulty enum value
- `language`: ISO language code
- `expected_behavior`: Expected model behavior description
- `tags`: List of tags for categorization
- `source`: Source of the prompt
- `verified`: Whether prompt has been verified
- `created_at`: Creation timestamp
- `metadata`: Additional metadata dictionary

**Methods**:

##### `to_dict() -> Dict[str, Any]`
Convert to dictionary representation.

##### `@classmethod from_dict(data: Dict[str, Any]) -> RedTeamPrompt`
Create instance from dictionary.

#### Class: EvaluationResult

**Description**: Contains safety evaluation results for a prompt-response pair.

**Constructor**:
```python
EvaluationResult(
    prompt_id: str,
    response_text: str,
    safety_rating: SafetyRating,
    response_category: ResponseCategory,
    confidence: float,
    refusal_detected: bool = False,
    harmful_content_detected: bool = False,
    pii_detected: bool = False,
    bias_detected: bool = False,
    issues_found: List[SafetyIssue] = None,
    evaluation_time: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None
)
```

**Properties**:
- `is_safe`: Boolean indicating if response is considered safe
- `is_compliant`: Boolean indicating if response appropriately complied
- `risk_score`: Numeric risk score (0.0 to 1.0)

#### Class: TestResult

**Description**: Complete result of testing a prompt against an LLM.

**Constructor**:
```python
TestResult(
    prompt: RedTeamPrompt,
    response: str,
    evaluation: EvaluationResult,
    model_info: ModelInfo,
    execution_time: float,
    success: bool = True,
    error: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None
)
```

**Properties**:
- `is_safe`: Delegate to evaluation.is_safe
- `safety_rating`: Delegate to evaluation.safety_rating
- `response_category`: Delegate to evaluation.response_category

#### Class: ModelInfo

**Description**: Information about the tested model.

```python
ModelInfo(
    name: str,
    provider: str,
    version: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
)
```

#### Class: SafetyIssue

**Description**: Represents a specific safety issue found during evaluation.

```python
SafetyIssue(
    type: IssueType,
    description: str,
    severity: str = "medium",
    confidence: float = 1.0,
    location: Optional[str] = None
)
```

---

## Enumerations

### PromptCategory

```python
class PromptCategory(str, Enum):
    JAILBREAK = "jailbreak"
    HARMFUL_CONTENT = "harmful_content"
    PRIVACY_VIOLATION = "privacy_violation"
    SECURITY_EXPLOIT = "security_exploit"
    BIAS_DISCRIMINATION = "bias_discrimination"
    ROLE_PLAY = "role_play"
    INSTRUCTION_OVERRIDE = "instruction_override"
    MISINFORMATION = "misinformation"
```

### PromptDifficulty

```python
class PromptDifficulty(str, Enum):
    EASY = "easy"        # Basic attempts, easily detected
    MEDIUM = "medium"    # Moderate sophistication
    HARD = "hard"        # Advanced techniques
    EXTREME = "extreme"  # Highly sophisticated attacks
```

### SafetyRating

```python
class SafetyRating(str, Enum):
    SAFE = "safe"                    # Response is safe and appropriate
    MOSTLY_SAFE = "mostly_safe"      # Minor concerns but generally safe
    BORDERLINE = "borderline"        # Concerning but not clearly harmful
    UNSAFE = "unsafe"                # Contains harmful content
    CRITICAL = "critical"            # Severely harmful or dangerous
```

### ResponseCategory

```python
class ResponseCategory(str, Enum):
    PROPER_REFUSAL = "proper_refusal"        # Correctly refused harmful request
    COMPLIANT = "compliant"                  # Appropriately answered benign query
    HARMFUL_COMPLIANT = "harmful_compliant"  # Incorrectly provided harmful content
    EVASIVE = "evasive"                      # Avoided direct answer without clear refusal
    PARTIAL_COMPLIANCE = "partial_compliance" # Partially complied with harmful request
```

### IssueType

```python
class IssueType(str, Enum):
    HARMFUL_CONTENT = "harmful_content"
    PRIVACY_VIOLATION = "privacy_violation"
    BIAS = "bias"
    MISINFORMATION = "misinformation"
    INAPPROPRIATE_TONE = "inappropriate_tone"
    INSTRUCTION_FOLLOWING = "instruction_following"
    OTHER = "other"
```

---

## Exceptions

### PETError

**Description**: Base exception class for all PET-specific errors.

```python
class PETError(Exception):
    pass
```

### ConfigurationError

**Description**: Raised when configuration is invalid or missing.

```python
class ConfigurationError(PETError):
    pass
```

### PromptError

**Description**: Raised when prompt operations fail.

```python
class PromptError(PETError):
    pass
```

### EvaluationError

**Description**: Raised when evaluation operations fail.

```python
class EvaluationError(PETError):
    pass
```

### APIError

**Description**: Raised when API calls fail.

```python
class APIError(PETError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code
```

---

## Utilities

### pet.utils.text_processing

#### `detect_language(text: str) -> str`
Detect the language of text using heuristics.

#### `normalize_text(text: str) -> str`
Normalize text for consistent processing.

#### `extract_patterns(text: str, patterns: List[str]) -> List[str]`
Extract text matching regex patterns.

### pet.utils.data_validation

#### `validate_prompt(prompt: RedTeamPrompt) -> List[str]`
Validate a prompt and return list of issues.

#### `validate_config(config: Dict[str, Any]) -> List[str]`
Validate configuration dictionary.

### pet.utils.file_operations

#### `load_json(file_path: Path) -> Dict[str, Any]`
Load JSON file safely.

#### `save_json(data: Dict[str, Any], file_path: Path) -> None`
Save data to JSON file.

#### `load_yaml(file_path: Path) -> Dict[str, Any]`
Load YAML file safely.

#### `save_yaml(data: Dict[str, Any], file_path: Path) -> None`
Save data to YAML file.

---

## CLI Reference

### Main Commands

#### `pet test`
Run safety tests on prompts.

```bash
pet test [OPTIONS]
```

**Options**:
- `--category`: Filter by prompt category
- `--difficulty`: Filter by difficulty level
- `--language`: Filter by language
- `--limit`: Maximum number of prompts to test
- `--model`: Model to test (default from config)
- `--config`: Path to configuration file
- `--output`: Output directory for results
- `--parallel`: Number of parallel requests
- `--save-responses`: Save raw model responses

#### `pet compare`
Compare multiple models on the same prompts.

```bash
pet compare MODEL1 MODEL2 [MODEL3...] [OPTIONS]
```

**Options**:
- `--category`: Filter prompts by category
- `--limit`: Number of prompts to test
- `--output`: Output file for comparison results

#### `pet list`
List available prompts with filtering.

```bash
pet list [OPTIONS]
```

**Options**:
- `--category`: Filter by category
- `--language`: Filter by language
- `--stats`: Show database statistics
- `--search`: Search prompts by text

#### `pet evaluate`
Evaluate existing responses or prompt-response pairs.

```bash
pet evaluate INPUT_FILE [OPTIONS]
```

**Options**:
- `--format`: Input format (json, csv)
- `--output`: Output file for evaluation results

#### `pet export`
Export prompts in various formats.

```bash
pet export OUTPUT_FILE [OPTIONS]
```

**Options**:
- `--format`: Output format (json, yaml, csv)
- `--category`: Filter by category
- `--language`: Filter by language

---

This API reference provides comprehensive documentation of all public classes, methods, and interfaces in the Prompt Engineering Toolkit. For more examples and tutorials, see the main documentation.