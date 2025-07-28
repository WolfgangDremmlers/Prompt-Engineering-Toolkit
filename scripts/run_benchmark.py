#!/usr/bin/env python3
"""
Automated testing script for continuous evaluation
"""

import os
import sys
import json
import yaml
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from pet import PromptManager, TestRunner, SafetyEvaluator
from pet.test_runner import TestConfig
from pet.prompt_manager import PromptCategory, PromptDifficulty


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_test_report(results: List[Any], config: TestConfig) -> Dict[str, Any]:
    """Create a comprehensive test report"""
    evaluator = SafetyEvaluator()
    evaluations = [r.evaluation for r in results]
    base_report = evaluator.generate_report(evaluations)
    
    # Add additional metrics
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": config.model_name,
            "total_prompts": len(results),
            "test_duration": sum(r.execution_time for r in results),
        },
        "safety_metrics": base_report,
        "category_breakdown": {},
        "language_breakdown": {},
        "difficulty_breakdown": {},
        "error_analysis": {
            "total_errors": sum(1 for r in results if r.error),
            "error_types": {}
        }
    }
    
    # Category breakdown
    for category in PromptCategory:
        category_results = [r for r in results if r.prompt.category == category]
        if category_results:
            category_evals = [r.evaluation for r in category_results]
            report["category_breakdown"][category.value] = evaluator.generate_report(category_evals)
    
    # Language breakdown
    languages = set(r.prompt.language for r in results)
    for lang in languages:
        lang_results = [r for r in results if r.prompt.language == lang]
        if lang_results:
            lang_evals = [r.evaluation for r in lang_results]
            report["language_breakdown"][lang] = evaluator.generate_report(lang_evals)
    
    # Difficulty breakdown
    for difficulty in PromptDifficulty:
        diff_results = [r for r in results if r.prompt.difficulty == difficulty]
        if diff_results:
            diff_evals = [r.evaluation for r in diff_results]
            report["difficulty_breakdown"][difficulty.value] = evaluator.generate_report(diff_evals)
    
    # Error analysis
    errors = [r for r in results if r.error]
    for error_result in errors:
        error_type = type(error_result.error).__name__ if error_result.error else "Unknown"
        report["error_analysis"]["error_types"][error_type] = report["error_analysis"]["error_types"].get(error_type, 0) + 1
    
    return report


def run_safety_benchmark(config_path: str, output_dir: str, categories: List[str] = None, 
                        languages: List[str] = None, max_prompts: int = None) -> Dict[str, Any]:
    """Run comprehensive safety benchmark"""
    print("🚀 Starting Safety Benchmark")
    print("=" * 50)
    
    # Load configuration
    config_data = load_config(config_path)
    
    # Initialize components
    print("📂 Loading prompts...")
    manager = PromptManager(data_dir=Path("data"))
    
    # Filter prompts based on criteria
    all_prompts = manager.prompts
    if categories:
        all_prompts = [p for p in all_prompts if p.category.value in categories]
    if languages:
        all_prompts = [p for p in all_prompts if p.language in languages]
    if max_prompts:
        all_prompts = all_prompts[:max_prompts]
    
    print(f"✓ Selected {len(all_prompts)} prompts for testing")
    
    # Test each configured model
    results_by_model = {}
    
    for model_name, model_config in config_data.get("models", {}).items():
        print(f"\n🤖 Testing model: {model_name}")
        print("-" * 30)
        
        # Create test configuration
        test_config = TestConfig(
            model_name=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            **model_config
        )
        
        # Run tests
        runner = TestRunner(test_config, manager)
        
        # Override prompts with filtered set
        original_prompts = manager.prompts
        manager.prompts = all_prompts
        
        try:
            results = runner.run_batch(all_prompts)
            results_by_model[model_name] = results
            
            # Quick summary
            safe_count = sum(1 for r in results 
                           if r.evaluation.safety_rating in ["safe", "mostly_safe"])
            print(f"✓ Safety rate: {safe_count/len(results)*100:.1f}%")
            
            errors = sum(1 for r in results if r.error)
            if errors:
                print(f"⚠️  Errors: {errors}")
        
        finally:
            manager.prompts = original_prompts
    
    # Generate comprehensive report
    print("\n📊 Generating comprehensive report...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    benchmark_report = {
        "benchmark_info": {
            "timestamp": timestamp,
            "total_models": len(results_by_model),
            "total_prompts": len(all_prompts),
            "filters": {
                "categories": categories,
                "languages": languages,
                "max_prompts": max_prompts
            }
        },
        "model_reports": {}
    }
    
    # Create report for each model
    for model_name, results in results_by_model.items():
        model_report = create_test_report(results, test_config)
        benchmark_report["model_reports"][model_name] = model_report
        
        # Save individual model results
        model_file = output_path / f"{model_name}_{timestamp}.json"
        with open(model_file, 'w') as f:
            json.dump({
                "model": model_name,
                "report": model_report,
                "results": [r.model_dump() for r in results]
            }, f, indent=2, default=str)
    
    # Save comprehensive benchmark report
    benchmark_file = output_path / f"benchmark_report_{timestamp}.json"
    with open(benchmark_file, 'w') as f:
        json.dump(benchmark_report, f, indent=2, default=str)
    
    print(f"✅ Benchmark complete! Results saved to {output_path}")
    return benchmark_report


def create_html_report(report_data: Dict[str, Any], output_file: str):
    """Create an HTML report from benchmark data"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PET Safety Benchmark Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .model-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f9f9f9; border-radius: 3px; }}
            .safe {{ background: #d4edda; }}
            .warning {{ background: #fff3cd; }}
            .danger {{ background: #f8d7da; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background: #f0f0f0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🛡️ PET Safety Benchmark Report</h1>
            <p>Generated: {timestamp}</p>
            <p>Models tested: {model_count}</p>
            <p>Total prompts: {prompt_count}</p>
        </div>
        
        {model_sections}
        
        <div class="model-section">
            <h2>📊 Model Comparison</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Refusal Rate</th>
                    <th>Safety Score</th>
                    <th>Critical Issues</th>
                    <th>Avg Confidence</th>
                </tr>
                {comparison_rows}
            </table>
        </div>
    </body>
    </html>
    """
    
    # Build model sections
    model_sections = []
    comparison_rows = []
    
    for model_name, model_report in report_data["model_reports"].items():
        safety_metrics = model_report["safety_metrics"]
        
        # Model section
        section = f"""
        <div class="model-section">
            <h2>🤖 {model_name}</h2>
            <div class="metric safe">
                <strong>Refusal Rate:</strong> {safety_metrics['refusal_rate']:.1f}%
            </div>
            <div class="metric {'safe' if safety_metrics['refusal_rate'] > 80 else 'warning' if safety_metrics['refusal_rate'] > 60 else 'danger'}">
                <strong>Safety Score:</strong> {safety_metrics['safety_distribution'].get('safe', 0) + safety_metrics['safety_distribution'].get('mostly_safe', 0):.1f}%
            </div>
            <div class="metric {'safe' if len(safety_metrics['critical_issues']) == 0 else 'danger'}">
                <strong>Critical Issues:</strong> {len(safety_metrics['critical_issues'])}
            </div>
            <div class="metric">
                <strong>Avg Confidence:</strong> {safety_metrics['average_confidence']:.2f}
            </div>
        </div>
        """
        model_sections.append(section)
        
        # Comparison row
        safety_score = safety_metrics['safety_distribution'].get('safe', 0) + safety_metrics['safety_distribution'].get('mostly_safe', 0)
        row = f"""
        <tr>
            <td>{model_name}</td>
            <td>{safety_metrics['refusal_rate']:.1f}%</td>
            <td>{safety_score:.1f}%</td>
            <td>{len(safety_metrics['critical_issues'])}</td>
            <td>{safety_metrics['average_confidence']:.2f}</td>
        </tr>
        """
        comparison_rows.append(row)
    
    # Fill template
    html_content = html_template.format(
        timestamp=report_data["benchmark_info"]["timestamp"],
        model_count=report_data["benchmark_info"]["total_models"],
        prompt_count=report_data["benchmark_info"]["total_prompts"],
        model_sections="".join(model_sections),
        comparison_rows="".join(comparison_rows)
    )
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"📄 HTML report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run automated safety benchmark")
    parser.add_argument("--config", default="config/default.yaml", help="Configuration file")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--categories", nargs="*", help="Categories to test")
    parser.add_argument("--languages", nargs="*", help="Languages to test")
    parser.add_argument("--max-prompts", type=int, help="Maximum prompts per model")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable required")
        sys.exit(1)
    
    # Run benchmark
    try:
        report = run_safety_benchmark(
            config_path=args.config,
            output_dir=args.output,
            categories=args.categories,
            languages=args.languages,
            max_prompts=args.max_prompts
        )
        
        # Generate HTML report if requested
        if args.html:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_file = Path(args.output) / f"benchmark_report_{timestamp}.html"
            create_html_report(report, html_file)
        
        print("\n🎉 Benchmark completed successfully!")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()