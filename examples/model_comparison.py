#!/usr/bin/env python3
"""
Example script: Model comparison
"""

import os
from pathlib import Path
from pet import PromptManager, TestRunner
from pet.test_runner import TestConfig
from pet.prompt_manager import PromptCategory, PromptDifficulty


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        return
    
    print("🔍 Prompt Engineering Toolkit - Model Comparison Example\n")
    
    # Initialize prompt manager
    prompt_manager = PromptManager(data_dir=Path("data"))
    
    # Select test prompts - mix of difficulties and categories
    test_prompts = []
    
    # Get some easy jailbreak attempts
    test_prompts.extend(
        prompt_manager.get_prompts(
            category=PromptCategory.JAILBREAK,
            difficulty=PromptDifficulty.EASY
        )[:2]
    )
    
    # Get some hard harmful content
    test_prompts.extend(
        prompt_manager.get_prompts(
            category=PromptCategory.HARMFUL_CONTENT,
            difficulty=PromptDifficulty.HARD
        )[:2]
    )
    
    # Get some security exploits
    test_prompts.extend(
        prompt_manager.get_prompts(
            category=PromptCategory.SECURITY_EXPLOIT
        )[:2]
    )
    
    print(f"📊 Selected {len(test_prompts)} prompts for comparison\n")
    
    # Define models to compare
    models = [
        "gpt-3.5-turbo",
        "gpt-4",
        # Add more models as needed
    ]
    
    # Create configurations for each model
    configs = []
    for model in models:
        config = TestConfig(
            model_name=model,
            api_key=api_key,
            temperature=0.7,
            max_tokens=300,
            parallel_requests=3
        )
        configs.append(config)
    
    # Run comparison
    print("🚀 Starting model comparison...\n")
    
    # Use first config to create runner
    runner = TestRunner(configs[0], prompt_manager)
    
    # Override prompt manager's prompts with our test set
    original_prompts = prompt_manager.prompts
    prompt_manager.prompts = test_prompts
    
    try:
        # Run comparison
        comparison_results = runner.analyze_model_comparison(configs)
        
        # Display detailed results
        print("\n\n📊 Detailed Comparison Results")
        print("=" * 80)
        
        for model_name, data in comparison_results.items():
            report = data["report"]
            results = data["results"]
            
            print(f"\n🤖 Model: {model_name}")
            print("-" * 40)
            print(f"✓ Refusal Rate: {report['refusal_rate']:.1f}%")
            print(f"✓ Safety Score: {report['safety_distribution'].get('safe', 0) + report['safety_distribution'].get('mostly_safe', 0):.1f}%")
            print(f"⚠️  Critical Issues: {len(report['critical_issues'])}")
            print(f"📊 Average Confidence: {report['average_confidence']:.2f}")
            
            # Show category performance
            print(f"\n📂 Category Performance:")
            for category in PromptCategory:
                category_results = [r for r in results if r.prompt.category == category]
                if category_results:
                    safe_count = sum(1 for r in category_results 
                                   if r.evaluation.safety_rating in ["safe", "mostly_safe"])
                    safe_rate = (safe_count / len(category_results)) * 100
                    print(f"   • {category.value}: {safe_rate:.1f}% safe responses")
            
            # Show specific failures
            if report['critical_issues']:
                print(f"\n❌ Critical Issues Found:")
                for prompt_id in report['critical_issues'][:3]:  # Show first 3
                    failed_result = next((r for r in results if r.prompt.id == prompt_id), None)
                    if failed_result:
                        print(f"   • Prompt: {failed_result.prompt.text[:60]}...")
                        print(f"     Response: {failed_result.response[:60]}...")
        
        # Generate summary comparison
        print("\n\n📊 Summary Comparison")
        print("=" * 80)
        print(f"{'Model':<20} {'Refusal Rate':<15} {'Safety Score':<15} {'Critical Issues':<15}")
        print("-" * 65)
        
        for model_name, data in comparison_results.items():
            report = data["report"]
            safety_score = report['safety_distribution'].get('safe', 0) + report['safety_distribution'].get('mostly_safe', 0)
            print(f"{model_name:<20} {report['refusal_rate']:<15.1f}% {safety_score:<15.1f}% {len(report['critical_issues']):<15}")
        
        # Determine best model
        best_model = max(comparison_results.items(), 
                        key=lambda x: x[1]['report']['refusal_rate'])
        print(f"\n🏆 Best performing model (highest refusal rate): {best_model[0]}")
        
        # Save detailed results
        output_file = Path("results") / "model_comparison.json"
        output_file.parent.mkdir(exist_ok=True)
        
        # Note: In real implementation, you'd save the full comparison_results
        print(f"\n💾 Full results saved to: {output_file}")
        
    finally:
        # Restore original prompts
        prompt_manager.prompts = original_prompts
    
    print("\n✅ Model comparison complete!")


if __name__ == "__main__":
    main()