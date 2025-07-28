"""
Prompt Manager Module

Handles loading, categorizing, and managing red teaming prompts.
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Set
from enum import Enum
from pydantic import BaseModel, Field


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


class PromptDifficulty(str, Enum):
    """Difficulty levels for prompts"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


class RedTeamPrompt(BaseModel):
    """Model for a red teaming prompt"""
    id: str
    text: str
    language: str = Field(default="en")
    category: PromptCategory
    difficulty: PromptDifficulty
    tags: List[str] = Field(default_factory=list)
    expected_behavior: str = Field(description="Expected safe behavior from the model")
    variations: List[str] = Field(default_factory=list)
    metadata: Dict[str, any] = Field(default_factory=dict)


class PromptManager:
    """Manages red teaming prompts with categorization and filtering"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data")
        self.prompts: List[RedTeamPrompt] = []
        self._load_prompts()
    
    def _load_prompts(self):
        """Load prompts from various sources"""
        # Load from legacy text file
        legacy_file = Path("red_teaming_prompts.txt")
        if legacy_file.exists():
            self._load_legacy_prompts(legacy_file)
        
        # Load from JSON files in data directory
        if self.data_dir.exists():
            for json_file in self.data_dir.glob("*.json"):
                self._load_json_prompts(json_file)
            
            for yaml_file in self.data_dir.glob("*.yaml"):
                self._load_yaml_prompts(yaml_file)
    
    def _load_legacy_prompts(self, file_path: Path):
        """Load prompts from the legacy text format"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Parse language and prompt
            if ': ' in line:
                lang, text = line.split(': ', 1)
                lang_code = lang.lower()
            else:
                lang_code = 'en'
                text = line
            
            # Categorize based on content
            category = self._infer_category(text)
            
            prompt = RedTeamPrompt(
                id=f"legacy_{idx}",
                text=text,
                language=lang_code,
                category=category,
                difficulty=PromptDifficulty.MEDIUM,
                expected_behavior="The model should refuse this request and explain why it cannot comply.",
                tags=["legacy", "imported"]
            )
            self.prompts.append(prompt)
    
    def _load_json_prompts(self, file_path: Path):
        """Load prompts from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                prompt = RedTeamPrompt(**item)
                self.prompts.append(prompt)
        elif isinstance(data, dict) and 'prompts' in data:
            for item in data['prompts']:
                prompt = RedTeamPrompt(**item)
                self.prompts.append(prompt)
    
    def _load_yaml_prompts(self, file_path: Path):
        """Load prompts from YAML file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if isinstance(data, list):
            for item in data:
                prompt = RedTeamPrompt(**item)
                self.prompts.append(prompt)
        elif isinstance(data, dict) and 'prompts' in data:
            for item in data['prompts']:
                prompt = RedTeamPrompt(**item)
                self.prompts.append(prompt)
    
    def _infer_category(self, text: str) -> PromptCategory:
        """Infer category from prompt text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['illegal', 'crime', 'hack', 'exploit']):
            return PromptCategory.HARMFUL_CONTENT
        elif any(word in text_lower for word in ['bomb', 'explosive', 'weapon']):
            return PromptCategory.HARMFUL_CONTENT
        elif any(word in text_lower for word in ['pretend', 'act as', 'roleplay']):
            return PromptCategory.ROLE_PLAY
        else:
            return PromptCategory.JAILBREAK
    
    def add_prompt(self, prompt: RedTeamPrompt):
        """Add a new prompt to the collection"""
        self.prompts.append(prompt)
    
    def get_prompts(
        self,
        category: Optional[PromptCategory] = None,
        difficulty: Optional[PromptDifficulty] = None,
        language: Optional[str] = None,
        tags: Optional[Set[str]] = None
    ) -> List[RedTeamPrompt]:
        """Get filtered prompts based on criteria"""
        filtered = self.prompts
        
        if category:
            filtered = [p for p in filtered if p.category == category]
        
        if difficulty:
            filtered = [p for p in filtered if p.difficulty == difficulty]
        
        if language:
            filtered = [p for p in filtered if p.language == language]
        
        if tags:
            filtered = [p for p in filtered if any(tag in p.tags for tag in tags)]
        
        return filtered
    
    def get_languages(self) -> Set[str]:
        """Get all available languages"""
        return {p.language for p in self.prompts}
    
    def get_categories(self) -> Set[PromptCategory]:
        """Get all used categories"""
        return {p.category for p in self.prompts}
    
    def save_to_json(self, file_path: Path):
        """Save all prompts to JSON file"""
        data = {
            "version": "1.0",
            "prompts": [p.model_dump() for p in self.prompts]
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def save_to_yaml(self, file_path: Path):
        """Save all prompts to YAML file"""
        data = {
            "version": "1.0",
            "prompts": [p.model_dump() for p in self.prompts]
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)