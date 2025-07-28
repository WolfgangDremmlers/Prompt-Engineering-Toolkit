"""
Test Runner Module

Orchestrates red teaming tests against LLM APIs.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor
import openai
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from .prompt_manager import PromptManager, RedTeamPrompt
from .evaluator import SafetyEvaluator, EvaluationResult


class TestConfig(BaseModel):
    """Configuration for test runs"""
    model_name: str = Field(default="gpt-3.5-turbo")
    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500)
    timeout: int = Field(default=30)
    parallel_requests: int = Field(default=5)
    rate_limit_delay: float = Field(default=0.1)
    system_prompt: Optional[str] = Field(default=None)
    save_results: bool = Field(default=True)
    results_dir: Path = Field(default=Path("results"))


class TestResult(BaseModel):
    """Result from a single test"""
    prompt: RedTeamPrompt
    response: str
    evaluation: EvaluationResult
    model_info: Dict[str, Any]
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = Field(default=None)


class TestRunner:
    """Runs red teaming tests against LLM APIs"""
    
    def __init__(self, config: TestConfig, prompt_manager: PromptManager):
        self.config = config
        self.prompt_manager = prompt_manager
        self.evaluator = SafetyEvaluator()
        self.console = Console()
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.async_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
    
    def run_single_test(self, prompt: RedTeamPrompt) -> TestResult:
        """Run a single test with a prompt"""
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = []
            if self.config.system_prompt:
                messages.append({
                    "role": "system",
                    "content": self.config.system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": prompt.text
            })
            
            # Call API
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            
            response_text = response.choices[0].message.content
            
            # Evaluate response
            evaluation = self.evaluator.evaluate(
                prompt.text,
                response_text,
                prompt.id
            )
            
            execution_time = time.time() - start_time
            
            return TestResult(
                prompt=prompt,
                response=response_text,
                evaluation=evaluation,
                model_info={
                    "model": response.model,
                    "usage": response.usage.model_dump() if response.usage else {}
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            # Create a dummy evaluation for error cases
            evaluation = EvaluationResult(
                prompt_id=prompt.id,
                response_text="",
                safety_rating="error",
                response_category="error",
                confidence=0.0,
                evaluation_notes=f"Error during API call: {error_msg}"
            )
            
            return TestResult(
                prompt=prompt,
                response="",
                evaluation=evaluation,
                model_info={"model": self.config.model_name},
                execution_time=execution_time,
                error=error_msg
            )
    
    async def _run_single_test_async(self, prompt: RedTeamPrompt) -> TestResult:
        """Async version of run_single_test"""
        start_time = time.time()
        
        try:
            messages = []
            if self.config.system_prompt:
                messages.append({
                    "role": "system",
                    "content": self.config.system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": prompt.text
            })
            
            # Add rate limiting delay
            await asyncio.sleep(self.config.rate_limit_delay)
            
            response = await self.async_client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            
            response_text = response.choices[0].message.content
            
            evaluation = self.evaluator.evaluate(
                prompt.text,
                response_text,
                prompt.id
            )
            
            execution_time = time.time() - start_time
            
            return TestResult(
                prompt=prompt,
                response=response_text,
                evaluation=evaluation,
                model_info={
                    "model": response.model,
                    "usage": response.usage.model_dump() if response.usage else {}
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            evaluation = EvaluationResult(
                prompt_id=prompt.id,
                response_text="",
                safety_rating="error",
                response_category="error",
                confidence=0.0,
                evaluation_notes=f"Error during API call: {error_msg}"
            )
            
            return TestResult(
                prompt=prompt,
                response="",
                evaluation=evaluation,
                model_info={"model": self.config.model_name},
                execution_time=execution_time,
                error=error_msg
            )
    
    async def _run_batch_async(self, prompts: List[RedTeamPrompt]) -> List[TestResult]:
        """Run a batch of prompts asynchronously"""
        semaphore = asyncio.Semaphore(self.config.parallel_requests)
        
        async def limited_test(prompt):
            async with semaphore:
                return await self._run_single_test_async(prompt)
        
        tasks = [limited_test(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def run_batch(self, prompts: List[RedTeamPrompt]) -> List[TestResult]:
        """Run a batch of prompts"""
        results = []
        
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
            
            # Use async if we have many prompts
            if len(prompts) > 10:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(self._run_batch_async(prompts))
                loop.close()
            else:
                # Use sequential for small batches
                for prompt in prompts:
                    result = self.run_single_test(prompt)
                    results.append(result)
                    progress.update(task, advance=1)
        
        return results
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all available prompts"""
        prompts = self.prompt_manager.prompts
        self.console.print(f"\n[bold]Running {len(prompts)} red teaming tests...[/bold]\n")
        
        results = self.run_batch(prompts)
        
        if self.config.save_results:
            self._save_results(results)
        
        self._display_summary(results)
        
        return results
    
    def run_category_tests(self, category: str) -> List[TestResult]:
        """Run tests for a specific category"""
        prompts = self.prompt_manager.get_prompts(category=category)
        self.console.print(f"\n[bold]Running {len(prompts)} tests for category: {category}[/bold]\n")
        
        results = self.run_batch(prompts)
        
        if self.config.save_results:
            self._save_results(results, suffix=f"_{category}")
        
        self._display_summary(results)
        
        return results
    
    def _save_results(self, results: List[TestResult], suffix: str = ""):
        """Save test results to file"""
        self.config.results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}{suffix}.json"
        file_path = self.config.results_dir / filename
        
        data = {
            "test_config": self.config.model_dump(),
            "timestamp": timestamp,
            "total_tests": len(results),
            "results": [r.model_dump() for r in results]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        self.console.print(f"\n[green]Results saved to: {file_path}[/green]")
    
    def _display_summary(self, results: List[TestResult]):
        """Display a summary of test results"""
        # Generate report
        evaluations = [r.evaluation for r in results]
        report = self.evaluator.generate_report(evaluations)
        
        # Create summary table
        table = Table(title="Test Results Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Tests", str(report["total_evaluations"]))
        table.add_row("Refusal Rate", f"{report['refusal_rate']:.1f}%")
        table.add_row("Harmful Content Rate", f"{report['harmful_content_rate']:.1f}%")
        table.add_row("Average Confidence", f"{report['average_confidence']:.2f}")
        
        self.console.print("\n")
        self.console.print(table)
        
        # Safety distribution table
        safety_table = Table(title="Safety Rating Distribution", show_header=True)
        safety_table.add_column("Rating", style="cyan")
        safety_table.add_column("Percentage", style="magenta")
        
        for rating, percentage in report["safety_distribution"].items():
            safety_table.add_row(rating, f"{percentage:.1f}%")
        
        self.console.print("\n")
        self.console.print(safety_table)
        
        # Show critical issues if any
        if report["critical_issues"]:
            self.console.print("\n[red bold]Critical Issues Found:[/red bold]")
            for prompt_id in report["critical_issues"]:
                self.console.print(f"  - Prompt ID: {prompt_id}")
        
        # Show errors if any
        errors = [r for r in results if r.error]
        if errors:
            self.console.print(f"\n[yellow]Errors encountered: {len(errors)}[/yellow]")
            for r in errors[:5]:  # Show first 5 errors
                self.console.print(f"  - {r.prompt.id}: {r.error}")
    
    def analyze_model_comparison(self, model_configs: List[TestConfig]) -> Dict[str, Any]:
        """Compare multiple models on the same test set"""
        comparison_results = {}
        
        for config in model_configs:
            self.config = config
            self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
            self.async_client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
            
            self.console.print(f"\n[bold]Testing model: {config.model_name}[/bold]")
            results = self.run_all_tests()
            
            evaluations = [r.evaluation for r in results]
            report = self.evaluator.generate_report(evaluations)
            
            comparison_results[config.model_name] = {
                "config": config.model_dump(),
                "report": report,
                "results": results
            }
        
        # Create comparison table
        comparison_table = Table(title="Model Comparison", show_header=True)
        comparison_table.add_column("Model", style="cyan")
        comparison_table.add_column("Refusal Rate", style="magenta")
        comparison_table.add_column("Safety Score", style="green")
        comparison_table.add_column("Critical Issues", style="red")
        
        for model_name, data in comparison_results.items():
            report = data["report"]
            safety_score = (
                report["safety_distribution"].get("safe", 0) + 
                report["safety_distribution"].get("mostly_safe", 0)
            )
            comparison_table.add_row(
                model_name,
                f"{report['refusal_rate']:.1f}%",
                f"{safety_score:.1f}%",
                str(len(report["critical_issues"]))
            )
        
        self.console.print("\n")
        self.console.print(comparison_table)
        
        return comparison_results