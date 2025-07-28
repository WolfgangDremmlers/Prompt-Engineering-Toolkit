"""
Data models for PET
"""

from typing import List, Dict, Optional, Set, Any, Union
from datetime import datetime
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, validator
import uuid


class PromptCategory(str, Enum):
    """Categories for different types of red teaming prompts"""
    JAILBREAK = "jailbreak"
    HARMFUL_CONTENT = "harmful_content"
    PRIVACY_VIOLATION = "privacy_violation"
    MISINFORMATION = "misinformation"
    BIAS_DISCRIMINATION = "bias_discrimination"
    SECURITY_EXPLOIT = "security_exploit"
    ROLE_PLAY = "role_play"
    INSTRUCTION_OVERRIDE = "instruction_override"
    PROMPT_INJECTION = "prompt_injection"
    SOCIAL_ENGINEERING = "social_engineering"


class PromptDifficulty(str, Enum):
    """Difficulty levels for prompts"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


class SafetyRating(str, Enum):
    """Safety rating levels"""
    SAFE = "safe"
    MOSTLY_SAFE = "mostly_safe"
    BORDERLINE = "borderline"
    UNSAFE = "unsafe"
    CRITICAL = "critical"
    ERROR = "error"


class ResponseCategory(str, Enum):
    """Categories of model responses"""
    PROPER_REFUSAL = "proper_refusal"
    PARTIAL_REFUSAL = "partial_refusal"
    EVASIVE = "evasive"
    COMPLIANT = "compliant"
    HARMFUL_COMPLIANT = "harmful_compliant"
    ERROR = "error"
    AMBIGUOUS = "ambiguous"


class IssueType(str, Enum):
    """Types of issues that can be detected"""
    HARMFUL_CONTENT = "harmful_content"
    PII_EXPOSURE = "pii_exposure"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    FACTUAL_ERROR = "factual_error"
    BIAS_DETECTED = "bias_detected"
    PRIVACY_BREACH = "privacy_breach"
    SECURITY_RISK = "security_risk"


class RedTeamPrompt(BaseModel):
    """Model for a red teaming prompt"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(min_length=1)
    language: str = Field(default="en", min_length=2, max_length=5)
    category: PromptCategory
    difficulty: PromptDifficulty
    tags: List[str] = Field(default_factory=list)
    expected_behavior: str = Field(description="Expected safe behavior from the model")
    variations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    source: Optional[str] = None
    author: Optional[str] = None
    verified: bool = False
    
    @validator('tags')
    def validate_tags(cls, v):
        return [tag.lower().strip() for tag in v if tag.strip()]
    
    @validator('language')
    def validate_language(cls, v):
        return v.lower()
    
    def add_variation(self, variation: str):
        """Add a variation of this prompt"""
        if variation and variation not in self.variations:
            self.variations.append(variation)
            self.updated_at = datetime.now()
    
    def add_tag(self, tag: str):
        """Add a tag to this prompt"""
        tag = tag.lower().strip()
        if tag and tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()


class DetectedIssue(BaseModel):
    """Represents a detected issue in a response"""
    type: IssueType
    description: str
    severity: int = Field(ge=1, le=10)  # 1 = low, 10 = critical
    location: Optional[str] = None  # Location in text where issue was found
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Result of evaluating a model response"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt_id: str
    response_text: str
    safety_rating: SafetyRating
    response_category: ResponseCategory
    confidence: float = Field(ge=0.0, le=1.0)
    issues_found: List[DetectedIssue] = Field(default_factory=list)
    refusal_detected: bool = False
    harmful_content_detected: bool = False
    bias_detected: bool = False
    pii_detected: bool = False
    evaluation_notes: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    evaluator_version: str = "1.0.0"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_issue(self, issue: DetectedIssue):
        """Add an issue to this evaluation"""
        self.issues_found.append(issue)
        
        # Update detection flags based on issue type
        if issue.type == IssueType.HARMFUL_CONTENT:
            self.harmful_content_detected = True
        elif issue.type == IssueType.PII_EXPOSURE:
            self.pii_detected = True
        elif issue.type == IssueType.BIAS_DETECTED:
            self.bias_detected = True
    
    def get_max_severity(self) -> int:
        """Get the maximum severity of all issues"""
        return max((issue.severity for issue in self.issues_found), default=0)
    
    def get_critical_issues(self) -> List[DetectedIssue]:
        """Get issues with severity >= 8"""
        return [issue for issue in self.issues_found if issue.severity >= 8]


class ModelInfo(BaseModel):
    """Information about the tested model"""
    name: str
    provider: str
    version: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    capabilities: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class UsageStats(BaseModel):
    """Token usage statistics"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_estimate: Optional[float] = None
    currency: str = "USD"


class TestResult(BaseModel):
    """Result from a single test"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: RedTeamPrompt
    response: str
    evaluation: EvaluationResult
    model_info: ModelInfo
    usage_stats: Optional[UsageStats] = None
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None
    retry_count: int = 0
    context: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def is_successful(self) -> bool:
        """Check if the test was successful (no errors)"""
        return self.error is None
    
    @property
    def is_safe(self) -> bool:
        """Check if the response is considered safe"""
        return self.evaluation.safety_rating in [SafetyRating.SAFE, SafetyRating.MOSTLY_SAFE]


class TestSession(BaseModel):
    """A testing session containing multiple test results"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    description: Optional[str] = None
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    model_info: ModelInfo
    test_results: List[TestResult] = Field(default_factory=list)
    configuration: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_result(self, result: TestResult):
        """Add a test result to this session"""
        self.test_results.append(result)
    
    def finish_session(self):
        """Mark the session as finished"""
        self.end_time = datetime.now()
    
    @property
    def duration(self) -> Optional[float]:
        """Get session duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def success_rate(self) -> float:
        """Get the success rate (tests without errors)"""
        if not self.test_results:
            return 0.0
        successful = sum(1 for result in self.test_results if result.is_successful)
        return successful / len(self.test_results)
    
    @property
    def safety_rate(self) -> float:
        """Get the safety rate (safe responses)"""
        if not self.test_results:
            return 0.0
        safe = sum(1 for result in self.test_results if result.is_safe)
        return safe / len(self.test_results)


class Benchmark(BaseModel):
    """A benchmark containing multiple test sessions"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.now)
    sessions: List[TestSession] = Field(default_factory=list)
    baseline_metrics: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_session(self, session: TestSession):
        """Add a test session to this benchmark"""
        self.sessions.append(session)
    
    def get_session_by_model(self, model_name: str) -> Optional[TestSession]:
        """Get session for a specific model"""
        for session in self.sessions:
            if session.model_info.name == model_name:
                return session
        return None
    
    def compare_models(self) -> Dict[str, Dict[str, float]]:
        """Compare performance across models"""
        comparison = {}
        for session in self.sessions:
            model_name = session.model_info.name
            comparison[model_name] = {
                "success_rate": session.success_rate,
                "safety_rate": session.safety_rate,
                "total_tests": len(session.test_results),
                "avg_confidence": sum(r.evaluation.confidence for r in session.test_results) / len(session.test_results) if session.test_results else 0,
                "critical_issues": sum(1 for r in session.test_results for issue in r.evaluation.issues_found if issue.severity >= 8)
            }
        return comparison


class PromptDatabase(BaseModel):
    """Database of red teaming prompts"""
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    prompts: List[RedTeamPrompt] = Field(default_factory=list)
    categories: Set[PromptCategory] = Field(default_factory=set)
    languages: Set[str] = Field(default_factory=set)
    tags: Set[str] = Field(default_factory=set)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_prompt(self, prompt: RedTeamPrompt):
        """Add a prompt to the database"""
        self.prompts.append(prompt)
        self.categories.add(prompt.category)
        self.languages.add(prompt.language)
        self.tags.update(prompt.tags)
        self.updated_at = datetime.now()
    
    def get_prompts_by_category(self, category: PromptCategory) -> List[RedTeamPrompt]:
        """Get prompts by category"""
        return [p for p in self.prompts if p.category == category]
    
    def get_prompts_by_language(self, language: str) -> List[RedTeamPrompt]:
        """Get prompts by language"""
        return [p for p in self.prompts if p.language == language]
    
    def get_prompts_by_difficulty(self, difficulty: PromptDifficulty) -> List[RedTeamPrompt]:
        """Get prompts by difficulty"""
        return [p for p in self.prompts if p.difficulty == difficulty]
    
    def search_prompts(self, query: str) -> List[RedTeamPrompt]:
        """Search prompts by text content"""
        query_lower = query.lower()
        return [
            p for p in self.prompts 
            if query_lower in p.text.lower() or 
            any(query_lower in tag for tag in p.tags)
        ]