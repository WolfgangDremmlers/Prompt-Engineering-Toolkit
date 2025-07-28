"""
Prompt Manager Module

Handles loading, categorizing, and managing red teaming prompts.
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Set, Union
import logging
from datetime import datetime

from ..models import (
    RedTeamPrompt, 
    PromptCategory, 
    PromptDifficulty,
    PromptDatabase
)

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages red teaming prompts with categorization and filtering"""
    
    def __init__(self, data_dir: Optional[Path] = None, auto_load: bool = True):
        self.data_dir = data_dir or Path("data")
        self.database = PromptDatabase()
        
        if auto_load:
            self._load_all_prompts()
    
    def _load_all_prompts(self):
        """Load prompts from various sources"""
        logger.info(f"Loading prompts from {self.data_dir}")
        
        # Load from legacy text file
        legacy_file = self.data_dir.parent / "red_teaming_prompts.txt"
        if legacy_file.exists():
            self._load_legacy_prompts(legacy_file)
        
        # Load from data directory
        if self.data_dir.exists():
            for json_file in self.data_dir.glob("*.json"):
                self._load_json_prompts(json_file)
            
            for yaml_file in self.data_dir.glob("*.yaml"):
                self._load_yaml_prompts(yaml_file)
            
            for yml_file in self.data_dir.glob("*.yml"):
                self._load_yaml_prompts(yml_file)
        
        logger.info(f"Loaded {len(self.database.prompts)} prompts")
    
    def _load_legacy_prompts(self, file_path: Path):
        """Load prompts from the legacy text format"""
        logger.debug(f"Loading legacy prompts from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Parse language and prompt
                if ': ' in line:
                    lang, text = line.split(': ', 1)
                    lang_code = self._normalize_language_code(lang)
                else:
                    lang_code = 'en'
                    text = line
                
                # Categorize based on content
                category = self._infer_category(text)
                difficulty = self._infer_difficulty(text)
                
                prompt = RedTeamPrompt(
                    id=f"legacy_{idx:03d}",
                    text=text,
                    language=lang_code,
                    category=category,
                    difficulty=difficulty,
                    expected_behavior="The model should refuse this request and explain why it cannot comply.",
                    tags=["legacy", "imported"],
                    source="legacy_file",
                    metadata={"original_line": idx + 1}
                )
                self.database.add_prompt(prompt)
                
        except Exception as e:
            logger.error(f"Error loading legacy prompts: {e}")
    
    def _load_json_prompts(self, file_path: Path):
        """Load prompts from JSON file"""
        logger.debug(f"Loading JSON prompts from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            prompts_data = []
            if isinstance(data, list):
                prompts_data = data
            elif isinstance(data, dict):
                if 'prompts' in data:
                    prompts_data = data['prompts']
                elif 'data' in data:
                    prompts_data = data['data']
                else:
                    # Single prompt object
                    prompts_data = [data]
            
            for item in prompts_data:
                try:
                    # Handle different formats
                    if isinstance(item, str):
                        # Simple string format
                        prompt = RedTeamPrompt(
                            text=item,
                            category=PromptCategory.JAILBREAK,
                            difficulty=PromptDifficulty.MEDIUM,
                            expected_behavior="Model should refuse",
                            source=file_path.name
                        )
                    else:
                        # Dictionary format
                        prompt = RedTeamPrompt(**item)
                        if not prompt.source:
                            prompt.source = file_path.name
                    
                    self.database.add_prompt(prompt)
                    
                except Exception as e:
                    logger.warning(f"Error loading prompt from {file_path}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
    
    def _load_yaml_prompts(self, file_path: Path):
        """Load prompts from YAML file"""
        logger.debug(f"Loading YAML prompts from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            prompts_data = []
            if isinstance(data, list):
                prompts_data = data
            elif isinstance(data, dict) and 'prompts' in data:
                prompts_data = data['prompts']
            
            for item in prompts_data:
                try:
                    prompt = RedTeamPrompt(**item)
                    if not prompt.source:
                        prompt.source = file_path.name
                    self.database.add_prompt(prompt)
                except Exception as e:
                    logger.warning(f"Error loading prompt from {file_path}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error loading YAML file {file_path}: {e}")
    
    def _normalize_language_code(self, lang: str) -> str:
        """Normalize language codes"""
        lang_map = {
            'EN': 'en', 'ENGLISH': 'en',
            'ZH': 'zh', 'CHINESE': 'zh', '中文': 'zh',
            'AR': 'ar', 'ARABIC': 'ar', 'العربية': 'ar',
            'ES': 'es', 'SPANISH': 'es', 'ESPAÑOL': 'es',
            'FR': 'fr', 'FRENCH': 'fr', 'FRANÇAIS': 'fr',
            'DE': 'de', 'GERMAN': 'de', 'DEUTSCH': 'de',
            'JA': 'ja', 'JAPANESE': 'ja', '日本語': 'ja',
            'KO': 'ko', 'KOREAN': 'ko', '한국어': 'ko',
            'RU': 'ru', 'RUSSIAN': 'ru', 'РУССКИЙ': 'ru',
            'IT': 'it', 'ITALIAN': 'it', 'ITALIANO': 'it',
            'PT': 'pt', 'PORTUGUESE': 'pt', 'PORTUGUÊS': 'pt',
            'NL': 'nl', 'DUTCH': 'nl', 'NEDERLANDS': 'nl',
        }
        
        return lang_map.get(lang.upper(), lang.lower())
    
    def _infer_category(self, text: str) -> PromptCategory:
        """Infer category from prompt text"""
        text_lower = text.lower()
        
        # Define keyword patterns for each category
        patterns = {
            PromptCategory.HARMFUL_CONTENT: [
                'kill', 'murder', 'bomb', 'explosive', 'weapon', 'poison', 'drug',
                'suicide', 'self-harm', 'violence', 'terrorist', 'illegal'
            ],
            PromptCategory.PRIVACY_VIOLATION: [
                'address', 'phone number', 'social security', 'credit card',
                'personal information', 'private', 'doxx', 'identity'
            ],
            PromptCategory.SECURITY_EXPLOIT: [
                'hack', 'exploit', 'vulnerability', 'bypass', 'crack',
                'malware', 'virus', 'breach', 'penetration'
            ],
            PromptCategory.ROLE_PLAY: [
                'pretend', 'act as', 'roleplay', 'imagine you are',
                'play the role', 'character', 'persona'
            ],
            PromptCategory.INSTRUCTION_OVERRIDE: [
                'ignore', 'disregard', 'forget', 'override', 'new instruction',
                'system:', 'debug mode', 'admin mode'
            ],
            PromptCategory.PROMPT_INJECTION: [
                'system prompt', 'initial instruction', 'base prompt',
                'reveal your prompt', 'show instructions'
            ],
            PromptCategory.MISINFORMATION: [
                'false information', 'fake news', 'conspiracy', 'hoax',
                'misinformation', 'lie', 'untrue'
            ],
            PromptCategory.BIAS_DISCRIMINATION: [
                'race', 'gender', 'religion', 'ethnicity', 'discrimination',
                'prejudice', 'stereotype', 'bias', 'inferior', 'superior'
            ],
            PromptCategory.SOCIAL_ENGINEERING: [
                'urgent', 'emergency', 'help me', 'trust me', 'authority',
                'official', 'police', 'government', 'emergency'
            ]
        }
        
        # Score each category
        category_scores = {}
        for category, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, default to JAILBREAK
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return PromptCategory.JAILBREAK
    
    def _infer_difficulty(self, text: str) -> PromptDifficulty:
        """Infer difficulty from prompt text"""
        text_lower = text.lower()
        
        # Extreme difficulty indicators
        extreme_indicators = [
            'step by step', 'detailed instructions', 'specific methods',
            'exact formula', 'precise procedure', 'bomb', 'weapon', 'kill'
        ]
        
        # Hard difficulty indicators  
        hard_indicators = [
            'pretend', 'roleplay', 'ignore instructions', 'system:',
            'bypass', 'override', 'hack', 'illegal'
        ]
        
        # Easy difficulty indicators
        easy_indicators = [
            'simple question', 'just tell me', 'basic information',
            'what is', 'how does', 'can you explain'
        ]
        
        if any(indicator in text_lower for indicator in extreme_indicators):
            return PromptDifficulty.EXTREME
        elif any(indicator in text_lower for indicator in hard_indicators):
            return PromptDifficulty.HARD
        elif any(indicator in text_lower for indicator in easy_indicators):
            return PromptDifficulty.EASY
        else:
            return PromptDifficulty.MEDIUM
    
    def add_prompt(self, prompt: RedTeamPrompt):
        """Add a new prompt to the collection"""
        self.database.add_prompt(prompt)
        logger.info(f"Added prompt: {prompt.id}")
    
    def get_prompt_by_id(self, prompt_id: str) -> Optional[RedTeamPrompt]:
        """Get a prompt by its ID"""
        for prompt in self.database.prompts:
            if prompt.id == prompt_id:
                return prompt
        return None
    
    def get_prompts(
        self,
        category: Optional[PromptCategory] = None,
        difficulty: Optional[PromptDifficulty] = None,
        language: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        limit: Optional[int] = None,
        verified_only: bool = False
    ) -> List[RedTeamPrompt]:
        """Get filtered prompts based on criteria"""
        filtered = self.database.prompts
        
        if category:
            filtered = [p for p in filtered if p.category == category]
        
        if difficulty:
            filtered = [p for p in filtered if p.difficulty == difficulty]
        
        if language:
            filtered = [p for p in filtered if p.language == language]
        
        if tags:
            filtered = [p for p in filtered if any(tag in p.tags for tag in tags)]
        
        if verified_only:
            filtered = [p for p in filtered if p.verified]
        
        if limit:
            filtered = filtered[:limit]
        
        return filtered
    
    def search_prompts(self, query: str) -> List[RedTeamPrompt]:
        """Search prompts by text content"""
        return self.database.search_prompts(query)
    
    def get_random_prompts(self, count: int = 10, **filters) -> List[RedTeamPrompt]:
        """Get random prompts with optional filters"""
        import random
        prompts = self.get_prompts(**filters)
        return random.sample(prompts, min(count, len(prompts)))
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about the prompt collection"""
        stats = {
            "total_prompts": len(self.database.prompts),
            "categories": {cat.value: len(self.database.get_prompts_by_category(cat)) 
                         for cat in PromptCategory},
            "difficulties": {diff.value: len(self.database.get_prompts_by_difficulty(diff)) 
                           for diff in PromptDifficulty},
            "languages": {lang: len(self.database.get_prompts_by_language(lang)) 
                         for lang in self.database.languages},
            "verified_prompts": sum(1 for p in self.database.prompts if p.verified),
            "average_tags_per_prompt": sum(len(p.tags) for p in self.database.prompts) / len(self.database.prompts) if self.database.prompts else 0,
            "most_common_tags": self._get_most_common_tags(10),
            "last_updated": self.database.updated_at
        }
        return stats
    
    def _get_most_common_tags(self, limit: int = 10) -> List[Dict[str, any]]:
        """Get most common tags"""
        from collections import Counter
        
        all_tags = []
        for prompt in self.database.prompts:
            all_tags.extend(prompt.tags)
        
        tag_counts = Counter(all_tags)
        return [{"tag": tag, "count": count} 
                for tag, count in tag_counts.most_common(limit)]
    
    def export_prompts(self, file_path: Path, format: str = "json", **filters):
        """Export prompts to file"""
        prompts = self.get_prompts(**filters)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            self._export_json(prompts, file_path)
        elif format.lower() in ["yaml", "yml"]:
            self._export_yaml(prompts, file_path)
        elif format.lower() == "csv":
            self._export_csv(prompts, file_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported {len(prompts)} prompts to {file_path}")
    
    def _export_json(self, prompts: List[RedTeamPrompt], file_path: Path):
        """Export prompts to JSON"""
        data = {
            "version": self.database.version,
            "exported_at": datetime.now().isoformat(),
            "count": len(prompts),
            "prompts": [prompt.model_dump() for prompt in prompts]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def _export_yaml(self, prompts: List[RedTeamPrompt], file_path: Path):
        """Export prompts to YAML"""
        data = {
            "version": self.database.version,
            "exported_at": datetime.now().isoformat(),
            "count": len(prompts),
            "prompts": [prompt.model_dump() for prompt in prompts]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    def _export_csv(self, prompts: List[RedTeamPrompt], file_path: Path):
        """Export prompts to CSV"""
        import csv
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'id', 'text', 'language', 'category', 'difficulty',
                'tags', 'expected_behavior', 'source', 'verified', 'created_at'
            ])
            
            # Data
            for prompt in prompts:
                writer.writerow([
                    prompt.id,
                    prompt.text,
                    prompt.language,
                    prompt.category.value,
                    prompt.difficulty.value,
                    ','.join(prompt.tags),
                    prompt.expected_behavior,
                    prompt.source or '',
                    prompt.verified,
                    prompt.created_at.isoformat()
                ])
    
    def validate_prompts(self) -> Dict[str, List[str]]:
        """Validate all prompts and return issues"""
        issues = {
            "errors": [],
            "warnings": [],
            "duplicates": []
        }
        
        seen_texts = {}
        
        for prompt in self.database.prompts:
            # Check for duplicates
            text_hash = hash(prompt.text.lower().strip())
            if text_hash in seen_texts:
                issues["duplicates"].append(f"Duplicate text found: {prompt.id} and {seen_texts[text_hash]}")
            else:
                seen_texts[text_hash] = prompt.id
            
            # Check for missing required fields
            if not prompt.text.strip():
                issues["errors"].append(f"Empty text in prompt {prompt.id}")
            
            if not prompt.expected_behavior.strip():
                issues["warnings"].append(f"Missing expected behavior in prompt {prompt.id}")
            
            # Check text length
            if len(prompt.text) < 10:
                issues["warnings"].append(f"Very short prompt text in {prompt.id}")
            elif len(prompt.text) > 1000:
                issues["warnings"].append(f"Very long prompt text in {prompt.id}")
            
            # Check for potentially problematic content
            if any(word in prompt.text.lower() for word in ['[name]', '[person]', '[location]']):
                issues["warnings"].append(f"Template placeholders found in {prompt.id}")
        
        return issues
    
    def merge_database(self, other_manager: 'PromptManager', resolve_conflicts: str = "skip"):
        """Merge another prompt database into this one"""
        added_count = 0
        skipped_count = 0
        updated_count = 0
        
        for other_prompt in other_manager.database.prompts:
            existing = self.get_prompt_by_id(other_prompt.id)
            
            if existing:
                if resolve_conflicts == "skip":
                    skipped_count += 1
                    continue
                elif resolve_conflicts == "update":
                    # Update existing prompt
                    idx = self.database.prompts.index(existing)
                    self.database.prompts[idx] = other_prompt
                    updated_count += 1
                elif resolve_conflicts == "rename":
                    # Add with new ID
                    other_prompt.id = f"{other_prompt.id}_merged"
                    self.database.add_prompt(other_prompt)
                    added_count += 1
            else:
                self.database.add_prompt(other_prompt)
                added_count += 1
        
        logger.info(f"Merge complete: {added_count} added, {updated_count} updated, {skipped_count} skipped")