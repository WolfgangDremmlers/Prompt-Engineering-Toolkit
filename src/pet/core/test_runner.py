"""
Test Runner Module

Orchestrates red teaming tests against LLM APIs.
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor
import openai
from openai import OpenAI, AsyncOpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from ..models import (
    RedTeamPrompt, 
    TestResult, 
    TestSession,
    ModelInfo,
    UsageStats
)
from ..core.config import Config
from ..core.prompt_manager import PromptManager
from ..core.evaluator import SafetyEvaluator

logger = logging.getLogger(__name__)


class TestRunner:
    """Runs red teaming tests against LLM APIs"""
    
    def __init__(self, config: Config, prompt_manager: PromptManager):
        self.config = config
        self.prompt_manager = prompt_manager
        self.evaluator = SafetyEvaluator(config.evaluation.model_dump())
        self.console = Console()
        
        # Initialize API clients
        self._init_clients()
        
        # Create output directories
        self._create_directories()
        
    def _init_clients(self):
        """Initialize API clients based on configuration"""
        if self.config.api.provider == "openai":
            self.client = OpenAI(
                api_key=self.config.api.api_key,
                base_url=self.config.api.base_url
            )
            self.async_client = AsyncOpenAI(
                api_key=self.config.api.api_key,
                base_url=self.config.api.base_url
            )
        elif self.config.api.provider == "anthropic":
            # Import and initialize Anthropic client if available
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.config.api.api_key)
                self.async_client = anthropic.AsyncAnthropic(api_key=self.config.api.api_key)
            except ImportError:
                logger.error("Anthropic package not installed. Run: pip install anthropic")
                raise
        else:
            # For local or other providers, use OpenAI-compatible interface
            self.client = OpenAI(
                api_key=self.config.api.api_key or "dummy",
                base_url=self.config.api.base_url or "http://localhost:8080"
            )
            self.async_client = AsyncOpenAI(
                api_key=self.config.api.api_key or "dummy",
                base_url=self.config.api.base_url or "http://localhost:8080"
            )
    
    def _create_directories(self):
        """Create necessary output directories"""
        for dir_path in [self.config.output.results_dir, 
                        self.config.output.reports_dir,
                        self.config.output.logs_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def run_single_test(self, prompt: RedTeamPrompt, 
                       system_prompt: Optional[str] = None) -> TestResult:
        """Run a single test with a prompt"""
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = []
            if system_prompt or self.config.system_prompts.get("default"):
                system_content = system_prompt or self.config.system_prompts["default"]
                messages.append({
                    "role": "system",
                    "content": system_content
                })
            
            messages.append({
                "role": "user",
                "content": prompt.text
            })
            
            # Get model-specific configuration
            model_config = self.config.get_model_config(self.config.api.model)
            
            # Call API
            response = self.client.chat.completions.create(
                model=self.config.api.model,
                messages=messages,
                temperature=model_config.get("temperature", self.config.api.temperature),
                max_tokens=model_config.get("max_tokens", self.config.api.max_tokens),
                timeout=self.config.api.timeout
            )
            
            response_text = response.choices[0].message.content or ""
            
            # Create model info
            model_info = ModelInfo(
                name=response.model,
                provider=self.config.api.provider,
                parameters=model_config
            )
            
            # Create usage stats
            usage_stats = None
            if response.usage:
                usage_stats = UsageStats(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
            
            # Evaluate response
            evaluation = self.evaluator.evaluate(
                prompt.text,
                response_text,
                prompt.id,
                prompt.language
            )
            
            execution_time = time.time() - start_time
            
            return TestResult(
                prompt=prompt,
                response=response_text,
                evaluation=evaluation,
                model_info=model_info,
                usage_stats=usage_stats,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Error testing prompt {prompt.id}: {error_msg}")
            
            # Create error evaluation
            from ..models import EvaluationResult, SafetyRating, ResponseCategory
            evaluation = EvaluationResult(
                prompt_id=prompt.id,
                response_text="",
                safety_rating=SafetyRating.ERROR,
                response_category=ResponseCategory.ERROR,
                confidence=0.0,
                evaluation_notes=f"Error during API call: {error_msg}"
            )
            
            return TestResult(
                prompt=prompt,
                response="",
                evaluation=evaluation,
                model_info=ModelInfo(
                    name=self.config.api.model,
                    provider=self.config.api.provider
                ),
                execution_time=execution_time,
                error=error_msg
            )
    
    async def _run_single_test_async(self, prompt: RedTeamPrompt, 
                                   system_prompt: Optional[str] = None) -> TestResult:
        """Async version of run_single_test"""
        start_time = time.time()
        
        try:
            messages = []
            if system_prompt or self.config.system_prompts.get("default"):
                system_content = system_prompt or self.config.system_prompts["default"]
                messages.append({
                    "role": "system",
                    "content": system_content
                })
            
            messages.append({
                "role": "user",
                "content": prompt.text
            })
            
            # Add rate limiting delay
            await asyncio.sleep(self.config.testing.rate_limit_delay)
            
            # Get model-specific configuration
            model_config = self.config.get_model_config(self.config.api.model)
            
            response = await self.async_client.chat.completions.create(
                model=self.config.api.model,
                messages=messages,
                temperature=model_config.get("temperature", self.config.api.temperature),
                max_tokens=model_config.get("max_tokens", self.config.api.max_tokens),
                timeout=self.config.api.timeout
            )
            
            response_text = response.choices[0].message.content or ""
            
            # Create model info
            model_info = ModelInfo(
                name=response.model,
                provider=self.config.api.provider,
                parameters=model_config
            )
            
            # Create usage stats
            usage_stats = None
            if response.usage:
                usage_stats = UsageStats(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
            
            evaluation = self.evaluator.evaluate(
                prompt.text,
                response_text,
                prompt.id,
                prompt.language
            )
            
            execution_time = time.time() - start_time
            
            return TestResult(
                prompt=prompt,
                response=response_text,
                evaluation=evaluation,
                model_info=model_info,
                usage_stats=usage_stats,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            from ..models import EvaluationResult, SafetyRating, ResponseCategory
            evaluation = EvaluationResult(
                prompt_id=prompt.id,
                response_text="",
                safety_rating=SafetyRating.ERROR,
                response_category=ResponseCategory.ERROR,
                confidence=0.0,
                evaluation_notes=f"Error during API call: {error_msg}"
            )
            
            return TestResult(
                prompt=prompt,
                response="",
                evaluation=evaluation,
                model_info=ModelInfo(
                    name=self.config.api.model,
                    provider=self.config.api.provider
                ),
                execution_time=execution_time,
                error=error_msg
            )
    
    async def _run_batch_async(self, prompts: List[RedTeamPrompt],
                              system_prompt: Optional[str] = None) -> List[TestResult]:
        """Run a batch of prompts asynchronously"""
        semaphore = asyncio.Semaphore(self.config.testing.parallel_requests)
        
        async def limited_test(prompt):
            async with semaphore:
                return await self._run_single_test_async(prompt, system_prompt)
        
        tasks = [limited_test(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def run_batch(self, prompts: List[RedTeamPrompt],
                  system_prompt: Optional[str] = None,
                  show_progress: bool = True) -> List[TestResult]:
        """Run a batch of prompts"""
        if not prompts:
            return []
        
        results = []
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console
            ) as progress:
                task = progress.add_task(
                    f"Testing {len(prompts)} prompts...",
                    total=len(prompts)
                )
                
                # Use async for larger batches
                if len(prompts) > 10:
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        results = loop.run_until_complete(
                            self._run_batch_async(prompts, system_prompt)
                        )
                        loop.close()
                        progress.update(task, completed=len(prompts))
                    except Exception as e:
                        logger.error(f"Async batch failed, falling back to sequential: {e}")
                        # Fallback to sequential
                        for prompt in prompts:
                            result = self.run_single_test(prompt, system_prompt)
                            results.append(result)
                            progress.update(task, advance=1)
                else:
                    # Use sequential for small batches
                    for prompt in prompts:
                        result = self.run_single_test(prompt, system_prompt)
                        results.append(result)
                        progress.update(task, advance=1)
        else:
            # No progress display
            if len(prompts) > 10:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(
                        self._run_batch_async(prompts, system_prompt)
                    )
                    loop.close()
                except Exception as e:
                    logger.error(f"Async batch failed, falling back to sequential: {e}")
                    results = [self.run_single_test(prompt, system_prompt) for prompt in prompts]
            else:
                results = [self.run_single_test(prompt, system_prompt) for prompt in prompts]
        
        return results
    
    def create_test_session(self, name: Optional[str] = None, 
                          description: Optional[str] = None) -> TestSession:
        """Create a new test session"""
        model_info = ModelInfo(
            name=self.config.api.model,
            provider=self.config.api.provider,
            parameters=self.config.get_model_config(self.config.api.model)
        )
        
        return TestSession(
            name=name,
            description=description,
            model_info=model_info,
            configuration=self.config.model_dump()
        )
    
    def run_comprehensive_test(self, 
                             test_name: Optional[str] = None,
                             categories: Optional[List[str]] = None,
                             languages: Optional[List[str]] = None,
                             difficulties: Optional[List[str]] = None,
                             limit: Optional[int] = None) -> TestSession:
        """Run comprehensive tests with filtering options"""
        
        # Get filtered prompts
        prompts = self.prompt_manager.get_prompts(
            category=categories[0] if categories and len(categories) == 1 else None,
            language=languages[0] if languages and len(languages) == 1 else None,
            difficulty=difficulties[0] if difficulties and len(difficulties) == 1 else None,
            limit=limit
        )
        
        # Apply additional filters if multiple values provided
        if categories and len(categories) > 1:
            prompts = [p for p in prompts if p.category.value in categories]
        if languages and len(languages) > 1:
            prompts = [p for p in prompts if p.language in languages]
        if difficulties and len(difficulties) > 1:
            prompts = [p for p in prompts if p.difficulty.value in difficulties]
        
        self.console.print(f"\n[bold]Running comprehensive test with {len(prompts)} prompts...[/bold]\n")
        
        # Create test session
        session = self.create_test_session(
            name=test_name or f"Comprehensive Test {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=f"Testing {len(prompts)} prompts across categories: {categories or 'all'}"
        )
        
        # Run tests
        results = self.run_batch(prompts)
        
        # Add results to session
        for result in results:
            session.add_result(result)
        
        session.finish_session()
        
        # Save session if configured
        if self.config.testing.save_responses:
            self._save_session(session)
        
        # Display summary
        self._display_session_summary(session)
        
        return session
    
    def compare_models(self, model_names: List[str], 
                      prompts: Optional[List[RedTeamPrompt]] = None) -> Dict[str, TestSession]:
        """Compare multiple models on the same test set"""
        
        if prompts is None:
            # Use a default set of diverse prompts
            prompts = self.prompt_manager.get_random_prompts(count=20)
        
        comparison_results = {}
        
        # Save original configuration
        original_model = self.config.api.model
        
        try:
            for model_name in model_names:
                self.console.print(f"\n[bold]Testing model: {model_name}[/bold]")
                
                # Update configuration for this model
                self.config.api.model = model_name
                self._init_clients()  # Reinitialize clients
                
                # Create session for this model
                session = self.create_test_session(
                    name=f"Model Comparison - {model_name}",
                    description=f"Comparison test with {len(prompts)} prompts"
                )
                
                # Run tests
                results = self.run_batch(prompts)
                
                # Add results to session
                for result in results:
                    session.add_result(result)
                
                session.finish_session()
                comparison_results[model_name] = session
                
                # Quick summary
                safe_count = sum(1 for r in results if r.is_safe)
                self.console.print(f"✓ Safety rate: {safe_count/len(results)*100:.1f}%")
                
                errors = sum(1 for r in results if r.error)
                if errors:
                    self.console.print(f"⚠️  Errors: {errors}")
        
        finally:
            # Restore original configuration
            self.config.api.model = original_model
            self._init_clients()
        
        # Display comparison table
        self._display_model_comparison(comparison_results)
        
        return comparison_results
    
    def _save_session(self, session: TestSession):
        """Save test session to file"""
        timestamp = session.start_time.strftime("%Y%m%d_%H%M%S")
        
        if self.config.output.create_timestamp_folders:
            output_dir = Path(self.config.output.results_dir) / timestamp
        else:
            output_dir = Path(self.config.output.results_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"test_session_{session.model_info.name}_{timestamp}.json"
        file_path = output_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(session.model_dump(), f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Session saved to: {file_path}")
    
    def _display_session_summary(self, session: TestSession):
        """Display a summary of test session results"""
        evaluations = [r.evaluation for r in session.test_results]
        report = self.evaluator.generate_report(evaluations)
        
        # Create summary table
        table = Table(title="Test Session Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Session Name", session.name or "Unnamed")
        table.add_row("Model", session.model_info.name)
        table.add_row("Total Tests", str(len(session.test_results)))
        table.add_row("Success Rate", f"{session.success_rate*100:.1f}%")
        table.add_row("Safety Rate", f"{session.safety_rate*100:.1f}%")
        table.add_row("Duration", f"{session.duration:.1f}s" if session.duration else "N/A")
        
        self.console.print("\n")
        self.console.print(table)
        
        # Safety distribution table
        safety_table = Table(title="Safety Rating Distribution", show_header=True)
        safety_table.add_column("Rating", style="cyan")
        safety_table.add_column("Count", style="magenta")
        safety_table.add_column("Percentage", style="green")
        
        for rating, percentage in report["safety_distribution"].items():
            count = int(percentage * len(session.test_results) / 100)
            safety_table.add_row(rating, str(count), f"{percentage:.1f}%")
        
        self.console.print("\n")
        self.console.print(safety_table)
        
        # Show critical issues if any
        if report["critical_issues"]:
            self.console.print("\n[red bold]Critical Issues Found:[/red bold]")
            for prompt_id in report["critical_issues"][:5]:  # Show first 5
                failed_result = next((r for r in session.test_results if r.prompt.id == prompt_id), None)
                if failed_result:
                    self.console.print(f"  - {prompt_id}: {failed_result.prompt.text[:60]}...")
    
    def _display_model_comparison(self, comparison_results: Dict[str, TestSession]):
        """Display model comparison results"""
        comparison_table = Table(title="Model Comparison", show_header=True)
        comparison_table.add_column("Model", style="cyan")
        comparison_table.add_column("Success Rate", style="green")
        comparison_table.add_column("Safety Rate", style="magenta")
        comparison_table.add_column("Avg Response Time", style="yellow")
        comparison_table.add_column("Errors", style="red")
        
        for model_name, session in comparison_results.items():
            avg_time = sum(r.execution_time for r in session.test_results) / len(session.test_results)
            error_count = sum(1 for r in session.test_results if r.error)
            
            comparison_table.add_row(
                model_name,
                f"{session.success_rate*100:.1f}%",
                f"{session.safety_rate*100:.1f}%",
                f"{avg_time:.2f}s",
                str(error_count)
            )
        
        self.console.print("\n")
        self.console.print(comparison_table)
        
        # Determine best model
        best_model = max(comparison_results.items(), 
                        key=lambda x: x[1].safety_rate)
        self.console.print(f"\n🏆 [bold green]Best performing model (highest safety rate): {best_model[0]}[/bold green]")