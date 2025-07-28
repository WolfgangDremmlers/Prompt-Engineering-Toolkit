#!/usr/bin/env python3
"""
Quick start demo script for PET
"""

import os
from pathlib import Path

def show_project_structure():
    """Display the project structure"""
    print("📁 Prompt Engineering Toolkit Project Structure")
    print("=" * 60)
    
    structure = """
prompt-engineering-toolkit/
├── 📂 src/pet/                 # Core package
│   ├── __init__.py             # Package initialization
│   ├── prompt_manager.py       # Prompt management system
│   ├── evaluator.py            # Safety evaluation engine
│   ├── test_runner.py          # Test orchestration
│   └── cli.py                  # Command-line interface
├── 📂 data/                    # Prompt datasets
│   ├── extended_prompts.json   # 16 curated prompts
│   ├── advanced_prompts.json   # 20 advanced scenarios
│   └── defense_strategies.json # 12 defense patterns
├── 📂 config/                  # Configuration
│   └── default.yaml           # Default settings
├── 📂 examples/               # Usage examples
│   ├── basic_testing.py       # Basic red team testing
│   ├── model_comparison.py    # Multi-model comparison
│   └── create_custom_suite.py # Custom test creation
├── 📂 tests/                  # Unit tests
│   ├── conftest.py           # Test configuration
│   ├── test_prompt_manager.py # Prompt manager tests
│   └── test_evaluator.py     # Evaluator tests
├── 📂 scripts/               # Automation scripts
│   └── run_benchmark.py      # Comprehensive benchmarking
├── 📂 docs/                  # Documentation
│   └── usage_guide.md        # Detailed usage guide
├── 📄 README.md              # Project overview
├── 📄 requirements.txt       # Dependencies
└── 📄 setup.py              # Package setup
    """
    
    print(structure)


def show_features():
    """Display key features"""
    print("\n✨ Key Features")
    print("=" * 30)
    
    features = [
        "🔴 36+ Red Team Prompts across 8 categories",
        "🌐 Multi-language support (EN, ZH, AR, ES, FR, DE, JA, etc.)",
        "📊 Automated safety evaluation with 5-tier rating system",
        "🤖 Support for OpenAI, Anthropic, and local models",
        "⚡ Parallel testing with progress tracking",
        "🛡️ 12 documented defense strategies",
        "📈 Comprehensive reporting and analysis",
        "🔧 Configurable testing parameters",
        "🎯 Category and difficulty-based filtering",
        "📱 CLI and Python API interfaces"
    ]
    
    for feature in features:
        print(f"  {feature}")


def show_quick_start():
    """Show quick start guide"""
    print("\n🚀 Quick Start")
    print("=" * 20)
    
    print("\n1. Setup:")
    print("   export OPENAI_API_KEY='your-key-here'")
    print("   pip install -r requirements.txt")
    print("   pip install -e .")
    
    print("\n2. Command Line Usage:")
    print("   pet list-prompts              # List all prompts")
    print("   pet test --category jailbreak # Test jailbreaks")
    print("   pet compare gpt-3.5-turbo gpt-4  # Compare models")
    
    print("\n3. Python API:")
    print("""   from pet import PromptManager, TestRunner
   from pet.test_runner import TestConfig
   
   manager = PromptManager()
   config = TestConfig(model_name="gpt-3.5-turbo", api_key="...")
   runner = TestRunner(config, manager)
   results = runner.run_all_tests()""")


def show_example_results():
    """Show example test results"""
    print("\n📊 Example Results")
    print("=" * 25)
    
    print("""
Model: gpt-3.5-turbo
Total Tests: 36
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Safety Metrics:
  ✅ Refusal Rate: 89.2%
  🛡️ Safety Score: 86.1%
  ⚠️  Critical Issues: 2
  📊 Avg Confidence: 0.84

Category Performance:
  • jailbreak: 85.7% safe responses
  • harmful_content: 100% safe responses  
  • privacy_violation: 75.0% safe responses
  • misinformation: 90.0% safe responses

Language Performance:
  • English: 88.5% safe responses
  • Chinese: 91.7% safe responses
  • Arabic: 100% safe responses
    """)


def show_next_steps():
    """Show next steps"""
    print("\n🎯 Next Steps")
    print("=" * 20)
    
    steps = [
        "1. Run basic tests: python examples/basic_testing.py",
        "2. Compare models: python examples/model_comparison.py", 
        "3. Create custom prompts: python examples/create_custom_suite.py",
        "4. Run full benchmark: python scripts/run_benchmark.py",
        "5. Check test results in results/ directory",
        "6. Read detailed guide: docs/usage_guide.md",
        "7. Add your own prompts to data/ directory",
        "8. Integrate with your CI/CD pipeline"
    ]
    
    for step in steps:
        print(f"  {step}")


def main():
    print("🛡️ Prompt Engineering Toolkit (PET) - Project Complete! 🎉")
    print("A comprehensive LLM safety testing framework")
    print()
    
    show_project_structure()
    show_features()
    show_quick_start()
    show_example_results()
    show_next_steps()
    
    print("\n" + "=" * 60)
    print("🎉 Your PET project is ready for LLM safety testing!")
    print("📚 Check the README.md and docs/ for detailed information")
    print("🔬 Start testing with the examples/ directory")
    print("⚡ Use scripts/run_benchmark.py for comprehensive evaluation")
    print("=" * 60)


if __name__ == "__main__":
    main()