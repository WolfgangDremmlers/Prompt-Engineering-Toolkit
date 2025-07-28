from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="prompt-engineering-toolkit",
    version="2.0.0",
    author="Wolfgang Dremmler",
    author_email="wolfgang@example.com",
    description="Advanced red teaming toolkit for LLM safety evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WolfgangDremmler/prompt-engineering-toolkit",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "pandas>=1.5.0",
        "numpy>=1.20.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
        "click>=8.0.0",
        "pyyaml>=6.0",
        "asyncio>=3.4.3",
        "aiofiles>=23.0.0",
        "httpx>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
            "isort>=5.12.0",
        ],
        "anthropic": [
            "anthropic>=0.7.0",
        ],
        "azure": [
            "azure-openai>=1.0.0",
        ],
        "web": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "jinja2>=3.1.0",
            "python-multipart>=0.0.6",
        ],
        "analysis": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
            "jupyter>=1.0.0",
        ],
        "all": [
            "anthropic>=0.7.0",
            "azure-openai>=1.0.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "jinja2>=3.1.0",
            "python-multipart>=0.0.6",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
            "jupyter>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "pet=pet.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pet": ["data/*.json", "data/*.yaml", "config/*.yaml"],
    },
    keywords="llm safety red-team prompt-engineering ai-safety evaluation testing",
    project_urls={
        "Bug Reports": "https://github.com/WolfgangDremmler/prompt-engineering-toolkit/issues",
        "Source": "https://github.com/WolfgangDremmler/prompt-engineering-toolkit",
        "Documentation": "https://prompt-engineering-toolkit.readthedocs.io/",
    },
)