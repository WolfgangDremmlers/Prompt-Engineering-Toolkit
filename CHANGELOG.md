# Changelog

All notable changes to the Prompt Engineering Toolkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and architecture
- Core Python package with modular design
- Comprehensive red team prompt database (70+ prompts)
- Advanced safety evaluation engine
- Multi-language support (English, Chinese, Arabic, Spanish, French, etc.)
- CLI interface with rich functionality
- Web interface with FastAPI backend
- Docker containerization support
- CI/CD pipeline with GitHub Actions
- Comprehensive documentation suite
- Automated deployment scripts

### Security
- Basic security measures implemented
- Input validation and sanitization
- CORS configuration for web interface

## [2.0.0] - 2024-12-28

### Added
- **Core Features**
  - Red team prompt database with 70+ carefully curated prompts
  - Safety evaluation engine with pattern matching and confidence scoring
  - Multi-language prompt support across 15+ languages
  - Async/sync LLM testing framework with batch processing
  - Session management for organizing test results
  - Rich CLI interface with progress tracking and colored output

- **Web Interface**
  - FastAPI-based web application
  - Dashboard with statistics and visualizations
  - Interactive test runner interface
  - Response evaluation tool
  - Bootstrap-based responsive UI
  - Real-time progress updates

- **Development Infrastructure**
  - Comprehensive pytest test suite with 90%+ coverage
  - Code quality tools (Black, flake8, mypy, pre-commit hooks)
  - GitHub Actions CI/CD pipeline
  - Docker multi-stage builds with production optimization
  - Automated security scanning and dependency checks

- **Documentation**
  - Complete user guide with tutorials and examples
  - API reference documentation
  - Installation guide for multiple platforms
  - Contributing guidelines for developers
  - Deployment documentation

- **Configuration & Deployment**
  - YAML-based configuration with environment variable support
  - Preset configurations for different use cases
  - Automated deployment scripts for PyPI
  - Docker Compose setup for development and production
  - Kubernetes deployment manifests

### Technical Details
- **Architecture**: Modular design with clear separation of concerns
- **Database**: File-based storage with JSON/YAML formats
- **API**: RESTful API with OpenAPI/Swagger documentation
- **Testing**: Unit tests, integration tests, and API tests
- **Performance**: Async processing with configurable parallelization
- **Security**: Input validation, CORS protection, secure defaults

### Supported Platforms
- **Operating Systems**: Windows, macOS, Linux
- **Python Versions**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Container Platforms**: Docker, Kubernetes
- **Cloud Providers**: AWS, Azure, GCP compatible

### LLM Provider Support
- OpenAI GPT models (GPT-3.5, GPT-4)
- Extensible architecture for additional providers
- Configurable API parameters and endpoints

## [1.0.0] - 2024-12-27

### Added
- Basic project structure
- Initial prompt collection
- Simple CLI interface
- Basic documentation

---

## Upgrade Guide

### From 1.0.0 to 2.0.0

This is a major release with significant architectural changes:

1. **Installation**: Use `pip install prompt-engineering-toolkit` instead of manual installation
2. **Configuration**: Migrate from simple config files to YAML-based configuration
3. **CLI**: New command structure - see `pet --help` for details
4. **API**: RESTful API replaces simple function calls
5. **Data**: Migrate existing prompts to new JSON format

### Migration Steps

1. **Backup existing data**:
   ```bash
   cp -r old_pet_data/ backup/
   ```

2. **Install new version**:
   ```bash
   pip install prompt-engineering-toolkit
   ```

3. **Migrate configuration**:
   ```bash
   pet config migrate --from old_config.txt --to ~/.pet/config.yaml
   ```

4. **Import existing prompts**:
   ```bash
   pet prompts import --file old_prompts.txt --format legacy
   ```

---

## Support

- **Documentation**: [GitHub Wiki](https://github.com/WolfgangDremmler/prompt-engineering-toolkit/wiki)
- **Issues**: [GitHub Issues](https://github.com/WolfgangDremmler/prompt-engineering-toolkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/WolfgangDremmler/prompt-engineering-toolkit/discussions)
- **Security**: See [SECURITY.md](SECURITY.md) for security issue reporting