"""
Configuration management for PET
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
import os
from pydantic import BaseModel, Field
from enum import Enum


class ModelProvider(str, Enum):
    """Supported model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class APIConfig(BaseModel):
    """API configuration"""
    provider: ModelProvider = ModelProvider.OPENAI
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, gt=0)
    timeout: int = Field(default=30, gt=0)
    max_retries: int = Field(default=3, ge=0)


class TestingConfig(BaseModel):
    """Testing configuration"""
    parallel_requests: int = Field(default=5, gt=0)
    rate_limit_delay: float = Field(default=0.1, ge=0.0)
    batch_size: int = Field(default=10, gt=0)
    retry_delay: float = Field(default=1.0, ge=0.0)
    save_responses: bool = True
    save_raw_responses: bool = False


class EvaluationConfig(BaseModel):
    """Evaluation configuration"""
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    strict_mode: bool = False
    custom_patterns: List[str] = Field(default_factory=list)
    language_specific_rules: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class OutputConfig(BaseModel):
    """Output configuration"""
    results_dir: Path = Field(default=Path("results"))
    reports_dir: Path = Field(default=Path("reports"))
    logs_dir: Path = Field(default=Path("logs"))
    create_timestamp_folders: bool = True
    export_formats: List[str] = Field(default=["json", "csv"])
    compress_results: bool = False


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: LogLevel = LogLevel.INFO
    file: Optional[str] = "pet.log"
    console: bool = True
    max_size: str = "10MB"
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class Config(BaseModel):
    """Main configuration class"""
    api: APIConfig = Field(default_factory=APIConfig)
    testing: TestingConfig = Field(default_factory=TestingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # System prompts
    system_prompts: Dict[str, str] = Field(default_factory=dict)
    
    # Model-specific configurations
    model_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Custom settings
    custom_settings: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, file_path: Path) -> "Config":
        """Load configuration from YAML file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        config = cls()
        
        # API configuration from environment
        if os.getenv("PET_API_KEY"):
            config.api.api_key = os.getenv("PET_API_KEY")
        if os.getenv("OPENAI_API_KEY"):
            config.api.api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("ANTHROPIC_API_KEY"):
            config.api.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if os.getenv("PET_MODEL"):
            config.api.model = os.getenv("PET_MODEL")
        if os.getenv("PET_TEMPERATURE"):
            config.api.temperature = float(os.getenv("PET_TEMPERATURE"))
        if os.getenv("PET_MAX_TOKENS"):
            config.api.max_tokens = int(os.getenv("PET_MAX_TOKENS"))
        
        # Testing configuration
        if os.getenv("PET_PARALLEL_REQUESTS"):
            config.testing.parallel_requests = int(os.getenv("PET_PARALLEL_REQUESTS"))
        if os.getenv("PET_RATE_LIMIT_DELAY"):
            config.testing.rate_limit_delay = float(os.getenv("PET_RATE_LIMIT_DELAY"))
        
        # Logging configuration
        if os.getenv("PET_LOG_LEVEL"):
            config.logging.level = LogLevel(os.getenv("PET_LOG_LEVEL"))
        
        return config
    
    def to_yaml(self, file_path: Path):
        """Save configuration to YAML file"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                self.model_dump(exclude_none=True), 
                f, 
                default_flow_style=False,
                allow_unicode=True,
                indent=2
            )
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return self.model_configs.get(model_name, {})
    
    def set_model_config(self, model_name: str, config: Dict[str, Any]):
        """Set model-specific configuration"""
        self.model_configs[model_name] = config
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check API key
        if not self.api.api_key:
            issues.append("API key is required")
        
        # Check directories exist or can be created
        for dir_path in [self.output.results_dir, self.output.reports_dir, self.output.logs_dir]:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create directory {dir_path}: {e}")
        
        # Validate temperature range
        if not 0.0 <= self.api.temperature <= 2.0:
            issues.append("Temperature must be between 0.0 and 2.0")
        
        # Validate parallel requests
        if self.testing.parallel_requests > 20:
            issues.append("Warning: High parallel requests may hit rate limits")
        
        return issues


# Default configurations for different use cases
class PresetConfigs:
    """Preset configurations for common use cases"""
    
    @staticmethod
    def development() -> Config:
        """Development configuration"""
        config = Config()
        config.api.temperature = 0.0  # Deterministic for testing
        config.testing.parallel_requests = 2
        config.testing.rate_limit_delay = 0.5
        config.logging.level = LogLevel.DEBUG
        config.output.save_raw_responses = True
        return config
    
    @staticmethod
    def production() -> Config:
        """Production configuration"""
        config = Config()
        config.api.temperature = 0.7
        config.testing.parallel_requests = 10
        config.testing.rate_limit_delay = 0.1
        config.logging.level = LogLevel.INFO
        config.output.compress_results = True
        return config
    
    @staticmethod
    def research() -> Config:
        """Research configuration"""
        config = Config()
        config.api.temperature = 0.5
        config.evaluation.strict_mode = True
        config.output.export_formats = ["json", "csv", "xlsx"]
        config.output.save_raw_responses = True
        config.logging.level = LogLevel.DEBUG
        return config
    
    @staticmethod
    def high_throughput() -> Config:
        """High throughput configuration"""
        config = Config()
        config.testing.parallel_requests = 20
        config.testing.batch_size = 50
        config.testing.rate_limit_delay = 0.05
        config.output.compress_results = True
        config.logging.level = LogLevel.WARNING
        return config


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration with fallback logic"""
    
    # Try to load from specified path
    if config_path and config_path.exists():
        return Config.from_yaml(config_path)
    
    # Try to load from default locations
    default_paths = [
        Path("config/config.yaml"),
        Path("config/default.yaml"),
        Path("pet_config.yaml"),
        Path.home() / ".pet" / "config.yaml"
    ]
    
    for path in default_paths:
        if path.exists():
            return Config.from_yaml(path)
    
    # Fall back to environment variables
    config = Config.from_env()
    
    # Validate configuration
    issues = config.validate_config()
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    return config