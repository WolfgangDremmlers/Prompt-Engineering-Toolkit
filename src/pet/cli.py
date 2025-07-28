"""
CLI Interface for Prompt Engineering Toolkit
"""

import click
import json
import yaml
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import Optional, List

from ..core.config import Config, load_config
from ..core.prompt_manager import PromptManager
from ..core.evaluator import SafetyEvaluator
from ..core.test_runner import TestRunner
from ..models import RedTeamPrompt, PromptCategory, PromptDifficulty

console = Console()


@click.group()
@click.version_option(version="2.0.0", prog_name="pet")
@click.option("--config", type=Path, help="Configuration file path")
@click.pass_context
def cli(ctx, config):
    """Prompt Engineering Toolkit - Advanced red teaming for LLM safety"""
    # Load configuration
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)


@cli.command()
@click.option("--data-dir", type=Path, default="data", help="Directory containing prompt data")
@click.option("--category", type=str, help="Filter by category")
@click.option("--language", type=str, help="Filter by language")
@click.option("--difficulty", type=str, help="Filter by difficulty")
@click.option("--tags", multiple=True, help="Filter by tags")
@click.option("--limit", type=int, help="Limit number of results")
@click.option("--format", "output_format", type=click.Choice(['table', 'json', 'yaml']), 
              default='table', help="Output format")
@click.pass_context
def list_prompts(ctx, data_dir, category, language, difficulty, tags, limit, output_format):
    """List available red teaming prompts"""
    manager = PromptManager(data_dir)
    
    # Apply filters
    prompts = manager.get_prompts(
        category=PromptCategory(category) if category else None,
        language=language,
        difficulty=PromptDifficulty(difficulty) if difficulty else None,
        tags=set(tags) if tags else None,
        limit=limit
    )
    
    if output_format == 'json':
        output = [p.model_dump() for p in prompts]
        console.print_json(json.dumps(output, indent=2, default=str))
    elif output_format == 'yaml':
        output = [p.model_dump() for p in prompts]
        console.print(yaml.dump(output, default_flow_style=False))
    else:
        # Table format
        table = Table(title=f"Red Teaming Prompts ({len(prompts)} total)")
        table.add_column("ID", style="cyan", width=12)
        table.add_column("Language", style="magenta", width=8)
        table.add_column("Category", style="green", width=15)
        table.add_column("Difficulty", style="yellow", width=10)
        table.add_column("Prompt", style="white")
        
        for prompt in prompts[:50]:  # Show first 50
            prompt_text = prompt.text[:80] + "..." if len(prompt.text) > 80 else prompt.text
            table.add_row(
                prompt.id,
                prompt.language,
                prompt.category.value,
                prompt.difficulty.value,
                prompt_text
            )
        
        console.print(table)
        
        if len(prompts) > 50:
            console.print(f"\n[dim]Showing first 50 of {len(prompts)} prompts. Use --limit to see more.[/dim]")
    
    # Show statistics
    if output_format == 'table':
        stats = manager.get_statistics()
        console.print(f"\n[bold]Statistics:[/bold]")
        console.print(f"Total prompts: {stats['total_prompts']}")
        console.print(f"Languages: {', '.join(stats['languages'].keys())}")
        console.print(f"Categories: {', '.join(stats['categories'].keys())}")


@cli.command()
@click.option("--model", default=None, help="Model name to test")
@click.option("--category", type=str, help="Test specific category")
@click.option("--language", type=str, help="Test specific language")
@click.option("--difficulty", type=str, help="Test specific difficulty")
@click.option("--prompt-id", type=str, help="Test specific prompt by ID")
@click.option("--limit", type=int, help="Limit number of prompts to test")
@click.option("--system-prompt", type=str, help="Custom system prompt")
@click.option("--output", type=Path, help="Output file for results")
@click.option("--no-progress", is_flag=True, help="Disable progress display")
@click.pass_context
def test(ctx, model, category, language, difficulty, prompt_id, limit, system_prompt, output, no_progress):
    """Run red teaming tests against an LLM"""
    config = ctx.obj['config']
    
    # Override model if specified
    if model:
        config.api.model = model
    
    # Check API key
    if not config.api.api_key:
        console.print("[red]Error: API key required. Set in config or environment variable[/red]")
        return
    
    # Initialize components
    console.print("📂 Loading prompts...")
    manager = PromptManager()
    runner = TestRunner(config, manager)
    
    console.print(f"✓ Loaded {len(manager.database.prompts)} prompts")
    
    # Get prompts to test
    if prompt_id:
        prompt = manager.get_prompt_by_id(prompt_id)
        if not prompt:
            console.print(f"[red]Prompt ID '{prompt_id}' not found[/red]")
            return
        prompts = [prompt]
    else:
        prompts = manager.get_prompts(
            category=PromptCategory(category) if category else None,
            language=language,
            difficulty=PromptDifficulty(difficulty) if difficulty else None,
            limit=limit
        )
    
    if not prompts:
        console.print("[yellow]No prompts found matching criteria[/yellow]")
        return
    
    console.print(f"🧪 Testing {len(prompts)} prompts with model: {config.api.model}")
    
    # Run tests
    try:
        session = runner.create_test_session(
            name=f"CLI Test {len(prompts)} prompts",
            description=f"Testing with filters: category={category}, language={language}, difficulty={difficulty}"
        )
        
        results = runner.run_batch(prompts, system_prompt, show_progress=not no_progress)
        
        for result in results:
            session.add_result(result)
        
        session.finish_session()
        
        # Save results
        if output:
            runner._save_session(session)
            console.print(f"\n[green]Results saved to: {output}[/green]")
        
        # Display summary
        runner._display_session_summary(session)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Testing interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error during testing: {e}[/red]")


@cli.command()
@click.argument("models", nargs=-1, required=True)
@click.option("--prompts-file", type=Path, help="JSON file with specific prompts to test")
@click.option("--category", type=str, help="Test specific category")
@click.option("--limit", type=int, default=20, help="Number of prompts to test")
@click.option("--output", type=Path, help="Output file for comparison")
@click.pass_context
def compare(ctx, models, prompts_file, category, limit, output):
    """Compare multiple models on the same test set"""
    config = ctx.obj['config']
    
    if not config.api.api_key:
        console.print("[red]Error: API key required[/red]")
        return
    
    # Initialize components
    manager = PromptManager()
    runner = TestRunner(config, manager)
    
    # Get test prompts
    if prompts_file:
        with open(prompts_file, 'r') as f:
            prompt_data = json.load(f)
        prompts = [RedTeamPrompt(**p) for p in prompt_data]
    else:
        prompts = manager.get_prompts(
            category=PromptCategory(category) if category else None,
            limit=limit
        )
    
    console.print(f"🔬 Comparing {len(models)} models on {len(prompts)} prompts")
    
    # Run comparison
    try:
        comparison = runner.compare_models(list(models), prompts)
        
        # Save results if output specified
        if output:
            comparison_data = {}
            for model_name, session in comparison.items():
                comparison_data[model_name] = session.model_dump()
            
            with open(output, 'w') as f:
                json.dump(comparison_data, f, indent=2, default=str)
            
            console.print(f"\n[green]Comparison saved to: {output}[/green]")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Comparison interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error during comparison: {e}[/red]")


@cli.command()
@click.argument("prompt_text")
@click.argument("response_text")
@click.option("--prompt-id", default="manual", help="ID for the prompt")
@click.option("--language", default="en", help="Language of the prompt")
@click.pass_context
def evaluate(ctx, prompt_text, response_text, prompt_id, language):
    """Evaluate a single model response"""
    config = ctx.obj['config']
    evaluator = SafetyEvaluator(config.evaluation.model_dump())
    
    result = evaluator.evaluate(prompt_text, response_text, prompt_id, language)
    
    # Display results
    console.print("\n[bold]Evaluation Results:[/bold]")
    
    # Create results table
    table = Table(show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Safety Rating", f"[{'green' if result.safety_rating.value == 'safe' else 'red'}]{result.safety_rating.value}[/]")
    table.add_row("Response Category", result.response_category.value)
    table.add_row("Confidence", f"{result.confidence:.2f}")
    table.add_row("Refusal Detected", "✓" if result.refusal_detected else "✗")
    table.add_row("Harmful Content", "✓" if result.harmful_content_detected else "✗")
    table.add_row("PII Detected", "✓" if result.pii_detected else "✗")
    table.add_row("Bias Detected", "✓" if result.bias_detected else "✗")
    
    console.print(table)
    
    if result.issues_found:
        console.print("\n[yellow]Issues Found:[/yellow]")
        for issue in result.issues_found:
            severity_color = "red" if issue.severity >= 8 else "yellow" if issue.severity >= 5 else "green"
            console.print(f"  • [{severity_color}]{issue.type.value}[/]: {issue.description} (Severity: {issue.severity})")
    
    console.print(f"\n[dim]Notes: {result.evaluation_notes}[/dim]")


@cli.command()
@click.option("--text", required=True, help="Prompt text")
@click.option("--language", default="en", help="Language code")
@click.option("--category", required=True, type=click.Choice([c.value for c in PromptCategory]))
@click.option("--difficulty", required=True, type=click.Choice([d.value for d in PromptDifficulty]))
@click.option("--expected", default="Model should refuse", help="Expected behavior")
@click.option("--tags", multiple=True, help="Tags for the prompt")
@click.option("--output", type=Path, default="data/custom_prompts.json", help="Output file")
def add_prompt(text, language, category, difficulty, expected, tags, output):
    """Add a new red teaming prompt"""
    # Create new prompt
    new_prompt = RedTeamPrompt(
        text=text,
        language=language,
        category=PromptCategory(category),
        difficulty=PromptDifficulty(difficulty),
        expected_behavior=expected,
        tags=list(tags),
        source="cli_manual",
        verified=False
    )
    
    # Load existing prompts if file exists
    prompts = []
    if output.exists():
        try:
            with open(output, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'prompts' in data:
                    prompts = data['prompts']
                elif isinstance(data, list):
                    prompts = data
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load existing file: {e}[/yellow]")
    
    prompts.append(new_prompt.model_dump())
    
    # Save back
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump({
            "version": "2.0.0",
            "description": "Custom red teaming prompts",
            "prompts": prompts
        }, f, indent=2, default=str)
    
    console.print(f"[green]✓ Prompt added successfully to {output}[/green]")
    console.print(f"ID: {new_prompt.id}")


@cli.command()
@click.option("--source", type=Path, required=True, help="Source file to convert")
@click.option("--output", type=Path, required=True, help="Output file")
@click.option("--format", "output_format", type=click.Choice(["json", "yaml"]), help="Output format")
def convert(source, output, output_format):
    """Convert prompt files between formats"""
    # Detect output format from extension if not specified
    if not output_format:
        if output.suffix.lower() in ['.yaml', '.yml']:
            output_format = 'yaml'
        else:
            output_format = 'json'
    
    # Load prompts using manager
    if source.is_dir():
        manager = PromptManager(data_dir=source, auto_load=True)
    else:
        manager = PromptManager(auto_load=False)
        if source.suffix == '.txt':
            manager._load_legacy_prompts(source)
        elif source.suffix == '.json':
            manager._load_json_prompts(source)
        elif source.suffix in ['.yaml', '.yml']:
            manager._load_yaml_prompts(source)
        else:
            console.print(f"[red]Unsupported source format: {source.suffix}[/red]")
            return
    
    # Export in new format
    try:
        manager.export_prompts(output, output_format)
        console.print(f"[green]✓ Converted {len(manager.database.prompts)} prompts to {output}[/green]")
    except Exception as e:
        console.print(f"[red]Conversion failed: {e}[/red]")


@cli.command()
@click.option("--data-dir", type=Path, default="data", help="Directory containing prompt data")
def stats(data_dir):
    """Show statistics about the prompt database"""
    manager = PromptManager(data_dir)
    stats = manager.get_statistics()
    
    console.print("\n[bold]📊 Prompt Database Statistics[/bold]\n")
    
    # Overview table
    overview = Table(title="Overview")
    overview.add_column("Metric", style="cyan")
    overview.add_column("Value", style="magenta")
    
    overview.add_row("Total Prompts", str(stats['total_prompts']))
    overview.add_row("Verified Prompts", str(stats['verified_prompts']))
    overview.add_row("Average Tags per Prompt", f"{stats['average_tags_per_prompt']:.1f}")
    overview.add_row("Last Updated", str(stats['last_updated']))
    
    console.print(overview)
    
    # Categories table
    categories = Table(title="Categories")
    categories.add_column("Category", style="green")
    categories.add_column("Count", style="magenta")
    
    for category, count in stats['categories'].items():
        if count > 0:
            categories.add_row(category, str(count))
    
    console.print("\n")
    console.print(categories)
    
    # Languages table
    languages = Table(title="Languages")
    languages.add_column("Language", style="blue")
    languages.add_column("Count", style="magenta")
    
    for language, count in stats['languages'].items():
        languages.add_row(language, str(count))
    
    console.print("\n")
    console.print(languages)
    
    # Top tags
    if stats['most_common_tags']:
        console.print("\n[bold]🏷️  Most Common Tags:[/bold]")
        for tag_info in stats['most_common_tags'][:10]:
            console.print(f"  • {tag_info['tag']}: {tag_info['count']}")


@cli.command()
@click.option("--data-dir", type=Path, default="data", help="Directory containing prompt data")
@click.option("--fix", is_flag=True, help="Attempt to fix issues automatically")
def validate(data_dir, fix):
    """Validate prompt database for issues"""
    manager = PromptManager(data_dir)
    issues = manager.validate_prompts()
    
    console.print("\n[bold]🔍 Validation Results[/bold]\n")
    
    if issues['errors']:
        console.print("[red bold]Errors:[/red bold]")
        for error in issues['errors']:
            console.print(f"  ❌ {error}")
    
    if issues['warnings']:
        console.print("\n[yellow bold]Warnings:[/yellow bold]")
        for warning in issues['warnings']:
            console.print(f"  ⚠️  {warning}")
    
    if issues['duplicates']:
        console.print("\n[blue bold]Duplicates:[/blue bold]")
        for duplicate in issues['duplicates']:
            console.print(f"  🔄 {duplicate}")
    
    if not any(issues.values()):
        console.print("[green]✅ No issues found! Database is clean.[/green]")
    else:
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        console.print(f"\n[bold]Total issues found: {total_issues}[/bold]")
        
        if fix:
            console.print("\n[yellow]Auto-fix functionality not yet implemented[/yellow]")


def main():
    """Main entry point"""
    cli()


if __name__ == "__main__":
    main()