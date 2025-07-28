# Installation Guide

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Verification](#verification)
4. [Configuration](#configuration)
5. [Optional Dependencies](#optional-dependencies)
6. [Development Installation](#development-installation)
7. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Disk Space**: 500MB for installation, additional space for results and logs
- **Internet Connection**: Required for API calls and package installation

### Recommended Requirements

- **Python**: 3.10 or higher
- **Memory**: 16GB RAM for large-scale testing
- **Disk Space**: 2GB for extensive result storage
- **API Access**: OpenAI API key with appropriate usage limits

### Supported Python Versions

| Python Version | Support Status | Notes |
|---------------|----------------|-------|
| 3.8 | ✅ Supported | Minimum version |
| 3.9 | ✅ Supported | Recommended |
| 3.10 | ✅ Supported | Recommended |
| 3.11 | ✅ Supported | Latest features |
| 3.12 | ✅ Supported | Latest features |

## Installation Methods

### Method 1: PyPI Installation (Recommended)

Install the latest stable version from PyPI:

```bash
pip install prompt-engineering-toolkit
```

Verify installation:
```bash
pet --version
```

### Method 2: Git Installation (Latest Features)

Install directly from the GitHub repository:

```bash
pip install git+https://github.com/WolfgangDremmler/prompt-engineering-toolkit.git
```

### Method 3: Local Development Installation

For development or customization:

```bash
# Clone the repository
git clone https://github.com/WolfgangDremmler/prompt-engineering-toolkit.git
cd prompt-engineering-toolkit

# Install in editable mode
pip install -e .
```

### Method 4: Docker Installation

Use the pre-built Docker image:

```bash
# Pull the image
docker pull wolfgangdremmler/prompt-engineering-toolkit:latest

# Run with API key
docker run -e OPENAI_API_KEY="your-key" wolfgangdremmler/prompt-engineering-toolkit pet --help
```

### Method 5: Conda Installation

Install via conda-forge (if available):

```bash
conda install -c conda-forge prompt-engineering-toolkit
```

## Verification

### Basic Verification

After installation, verify that PET is working correctly:

```bash
# Check version
pet --version

# Verify CLI commands
pet --help

# Check Python import
python -c "import pet; print('PET imported successfully')"
```

### Database Verification

Verify that the prompt database is loaded correctly:

```bash
# Show database statistics
pet list --stats

# List some prompts
pet list --limit 5
```

Expected output should show:
- Total number of prompts (70+)
- Multiple categories
- Multiple languages

### API Connectivity Test

Test API connectivity (requires API key):

```bash
# Set API key
export OPENAI_API_KEY="your-openai-api-key"

# Run a simple test
pet test --limit 1
```

## Configuration

### API Key Setup

#### Method 1: Environment Variable (Recommended)

```bash
# Linux/macOS
export OPENAI_API_KEY="your-openai-api-key-here"

# Windows Command Prompt
set OPENAI_API_KEY=your-openai-api-key-here

# Windows PowerShell
$env:OPENAI_API_KEY="your-openai-api-key-here"
```

#### Method 2: Configuration File

Create `~/.pet/config.yaml`:

```yaml
api:
  api_key: "your-openai-api-key-here"
  model: "gpt-3.5-turbo"
  temperature: 0.7
```

#### Method 3: Interactive Setup

Run the setup wizard:

```bash
pet setup
```

This will guide you through:
1. API key configuration
2. Default model selection
3. Output directory setup
4. Logging preferences

### Directory Structure

PET creates the following directory structure:

```
~/.pet/
├── config.yaml          # Main configuration file
├── data/                 # Custom prompt data
│   ├── custom_prompts.json
│   └── imported_prompts.yaml
├── cache/                # API response cache
├── logs/                 # Log files
│   └── pet.log
└── results/              # Test results
    ├── sessions/
    └── exports/
```

### Configuration Validation

Validate your configuration:

```bash
# Check configuration
pet config --check

# Show current configuration
pet config --show

# Test API connectivity
pet config --test-api
```

## Optional Dependencies

PET supports several optional dependencies for extended functionality:

### Anthropic Claude Support

```bash
pip install "prompt-engineering-toolkit[anthropic]"
```

Enables testing with Claude models:
```python
config.api.provider = "anthropic"
config.api.model = "claude-3-sonnet-20240229"
```

### Azure OpenAI Support

```bash
pip install "prompt-engineering-toolkit[azure]"
```

Configure for Azure:
```yaml
api:
  provider: "azure"
  azure_endpoint: "https://your-resource.openai.azure.com"
  api_version: "2023-12-01-preview"
```

### Web Interface

```bash
pip install "prompt-engineering-toolkit[web]"
```

Launch web interface:
```bash
pet web --port 8080
```

### Analysis and Plotting

```bash
pip install "prompt-engineering-toolkit[analysis]"
```

Enables:
- Statistical analysis
- Data visualization
- Jupyter notebook integration

### All Optional Dependencies

Install everything:

```bash
pip install "prompt-engineering-toolkit[all]"
```

## Development Installation

### Setting Up Development Environment

1. **Clone and install in development mode**:
   ```bash
   git clone https://github.com/WolfgangDremmler/prompt-engineering-toolkit.git
   cd prompt-engineering-toolkit
   pip install -e ".[dev]"
   ```

2. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

3. **Verify development setup**:
   ```bash
   # Run tests
   pytest

   # Check code style
   black --check src/ tests/
   flake8 src/ tests/

   # Type checking
   mypy src/
   ```

### Development Dependencies

The development installation includes:

- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **pytest-asyncio**: Async test support
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Git hooks
- **isort**: Import sorting

### Building Documentation

Build documentation locally:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build HTML documentation
cd docs/
make html

# Serve documentation locally
python -m http.server 8000 --directory _build/html/
```

### Running Tests

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pet --cov-report=html

# Run specific test categories
pytest -m unit        # Unit tests only
pytest -m integration # Integration tests only
pytest -m slow        # Slow tests only

# Run tests with API access (requires key)
pytest -m api --api-key="your-key"
```

## Troubleshooting

### Common Installation Issues

#### Issue: `pip` command not found

**Solution**:
```bash
# Install pip if missing
python -m ensurepip --upgrade

# Or use python -m pip instead of pip
python -m pip install prompt-engineering-toolkit
```

#### Issue: Permission denied during installation

**Solutions**:
```bash
# Option 1: Install for current user only
pip install --user prompt-engineering-toolkit

# Option 2: Use virtual environment (recommended)
python -m venv pet_env
source pet_env/bin/activate  # Linux/macOS
# or
pet_env\Scripts\activate     # Windows
pip install prompt-engineering-toolkit

# Option 3: Use sudo (not recommended)
sudo pip install prompt-engineering-toolkit
```

#### Issue: Python version incompatibility

**Symptoms**: 
- `requires python_version >= "3.8"` error
- Import errors with typing features

**Solutions**:
```bash
# Check Python version
python --version

# Install/upgrade Python to 3.8+
# Use pyenv for version management
curl https://pyenv.run | bash
pyenv install 3.11.0
pyenv global 3.11.0
```

#### Issue: SSL/TLS certificate errors

**Symptoms**:
- Certificate verification failed
- SSL errors during installation

**Solutions**:
```bash
# Option 1: Upgrade certificates
pip install --upgrade certifi

# Option 2: Use trusted hosts (temporary)
pip install --trusted-host pypi.org --trusted-host pypi.python.org prompt-engineering-toolkit

# Option 3: Configure pip permanently
pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org"
```

### Platform-Specific Issues

#### Windows

**Issue**: Windows Defender blocking installation

**Solution**:
1. Add Python installation directory to Windows Defender exclusions
2. Run Command Prompt as Administrator
3. Use Windows Subsystem for Linux (WSL) as alternative

**Issue**: Long path names causing errors

**Solution**:
```bash
# Enable long path support in Windows
# Run as Administrator in PowerShell:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

#### macOS

**Issue**: Command line tools not installed

**Solution**:
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew (optional but recommended)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python via Homebrew
brew install python
```

#### Linux

**Issue**: Missing system dependencies

**Solution**:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv build-essential

# CentOS/RHEL
sudo yum install python3 python3-pip python3-venv gcc

# Arch Linux
sudo pacman -S python python-pip base-devel
```

### Post-Installation Issues

#### Issue: `pet` command not found

**Symptoms**:
- Command line tool not available after installation
- `pet: command not found` error

**Solutions**:
```bash
# Check if installed in user directory
echo $PATH | grep -E "(\.local/bin|Python.*Scripts)"

# Add to PATH if needed (Linux/macOS)
export PATH="$HOME/.local/bin:$PATH"

# Add to PATH if needed (Windows)
set PATH=%PATH%;%APPDATA%\Python\Python311\Scripts

# Or use python -m
python -m pet --help
```

#### Issue: Import errors

**Symptoms**:
- `ModuleNotFoundError: No module named 'pet'`
- Import errors in Python scripts

**Solutions**:
```bash
# Check installation
pip list | grep prompt-engineering-toolkit

# Reinstall if missing
pip install --force-reinstall prompt-engineering-toolkit

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Use virtual environment
python -m venv pet_env
source pet_env/bin/activate
pip install prompt-engineering-toolkit
```

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Look in `~/.pet/logs/pet.log` for detailed error information
2. **Enable debug mode**: Set `PET_LOG_LEVEL=DEBUG` environment variable
3. **Search GitHub Issues**: Check existing issues at https://github.com/WolfgangDremmler/prompt-engineering-toolkit/issues
4. **Create a new issue**: Include:
   - Operating system and version
   - Python version (`python --version`)
   - PET version (`pet --version`)
   - Full error message
   - Steps to reproduce

### Uninstallation

To completely remove PET:

```bash
# Uninstall package
pip uninstall prompt-engineering-toolkit

# Remove configuration and data (optional)
rm -rf ~/.pet/

# Remove from PATH if manually added
# Edit ~/.bashrc, ~/.zshrc, or equivalent
```

---

## Next Steps

After successful installation:

1. **Set up your API key**: Follow the [Configuration](#configuration) section
2. **Run your first test**: Try `pet test --limit 1`
3. **Read the User Guide**: Check out [USER_GUIDE.md](USER_GUIDE.md) for detailed usage instructions
4. **Explore examples**: Look at the examples in the `examples/` directory

For detailed usage instructions, see the [User Guide](USER_GUIDE.md) and [API Reference](API_REFERENCE.md).