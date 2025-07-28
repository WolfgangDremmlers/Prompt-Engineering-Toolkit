#!/bin/bash

# Deployment script for Prompt Engineering Toolkit
# This script automates the deployment process to PyPI

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PACKAGE_NAME="prompt-engineering-toolkit"
BUILD_DIR="dist"
VENV_DIR="venv_deploy"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking deployment requirements..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    # Check if git is available
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed or not in PATH"
        exit 1
    fi
    
    # Check if we're in a git repository
    if ! git rev-parse --is-inside-work-tree &> /dev/null; then
        log_error "Not in a git repository"
        exit 1
    fi
    
    # Check if working directory is clean
    if [[ -n $(git status --porcelain) ]]; then
        log_warning "Working directory is not clean. Uncommitted changes detected."
        read -p "Do you want to continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    log_success "Requirements check passed"
}

setup_environment() {
    log_info "Setting up deployment environment..."
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "$VENV_DIR" ]]; then
        python3 -m venv "$VENV_DIR"
        log_info "Created virtual environment: $VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip and install build tools
    pip install --upgrade pip
    pip install --upgrade build twine wheel setuptools
    
    # Install package in development mode for testing
    pip install -e ".[dev]"
    
    log_success "Environment setup complete"
}

run_tests() {
    log_info "Running test suite..."
    
    # Run tests
    if ! pytest tests/ --tb=short; then
        log_error "Tests failed. Deployment aborted."
        exit 1
    fi
    
    # Run code quality checks
    log_info "Running code quality checks..."
    
    # Check code formatting
    if ! black --check src/ tests/; then
        log_error "Code formatting check failed. Run 'black src/ tests/' to fix."
        exit 1
    fi
    
    # Check linting
    if ! flake8 src/ tests/; then
        log_error "Linting check failed. Fix the issues before deploying."
        exit 1
    fi
    
    # Check type hints
    if ! mypy src/; then
        log_warning "Type checking found issues. Consider fixing them."
    fi
    
    log_success "All tests and checks passed"
}

get_version() {
    # Extract version from setup.py or __init__.py
    VERSION=$(python3 setup.py --version 2>/dev/null || python3 -c "import pet; print(pet.__version__)" 2>/dev/null)
    
    if [[ -z "$VERSION" ]]; then
        log_error "Could not determine package version"
        exit 1
    fi
    
    echo "$VERSION"
}

check_version_tag() {
    local version=$1
    local tag="v$version"
    
    # Check if tag already exists
    if git rev-parse "$tag" >/dev/null 2>&1; then
        log_error "Tag $tag already exists. Update version number first."
        exit 1
    fi
    
    log_info "Version $version is ready for deployment"
}

build_package() {
    log_info "Building package..."
    
    # Clean previous builds
    rm -rf "$BUILD_DIR"
    rm -rf *.egg-info
    
    # Build package
    python3 -m build
    
    # Verify build artifacts
    if [[ ! -d "$BUILD_DIR" ]] || [[ -z "$(ls -A $BUILD_DIR)" ]]; then
        log_error "Build failed - no artifacts created"
        exit 1
    fi
    
    log_success "Package built successfully"
    ls -la "$BUILD_DIR"
}

test_package() {
    log_info "Testing built package..."
    
    # Create temporary environment for testing
    TEMP_VENV="venv_test"
    python3 -m venv "$TEMP_VENV"
    source "$TEMP_VENV/bin/activate"
    
    # Install from wheel
    local wheel_file=$(ls dist/*.whl | head -n1)
    pip install "$wheel_file"
    
    # Test basic import
    if ! python3 -c "import pet; print('Package import successful')"; then
        log_error "Package import test failed"
        rm -rf "$TEMP_VENV"
        exit 1
    fi
    
    # Test CLI
    if ! pet --version; then
        log_error "CLI test failed"
        rm -rf "$TEMP_VENV"
        exit 1
    fi
    
    # Cleanup
    deactivate
    rm -rf "$TEMP_VENV"
    source "$VENV_DIR/bin/activate"  # Return to deployment environment
    
    log_success "Package testing completed"
}

upload_to_pypi() {
    local target=$1
    
    if [[ "$target" == "test" ]]; then
        log_info "Uploading to Test PyPI..."
        repository_url="--repository testpypi"
        log_warning "This will upload to Test PyPI. Use 'deploy.sh prod' for production."
    elif [[ "$target" == "prod" ]]; then
        log_info "Uploading to Production PyPI..."
        repository_url=""
        log_warning "This will upload to Production PyPI. This action cannot be undone!"
        
        # Final confirmation for production
        read -p "Are you sure you want to deploy to PRODUCTION PyPI? (yes/NO): " -r
        if [[ ! $REPLY == "yes" ]]; then
            log_info "Production deployment cancelled by user"
            exit 0
        fi
    else
        log_error "Invalid target. Use 'test' or 'prod'"
        exit 1
    fi
    
    # Check for API tokens
    if [[ "$target" == "test" && -z "$TESTPYPI_API_TOKEN" ]]; then
        log_warning "TESTPYPI_API_TOKEN not set. You'll be prompted for credentials."
    elif [[ "$target" == "prod" && -z "$PYPI_API_TOKEN" ]]; then
        log_warning "PYPI_API_TOKEN not set. You'll be prompted for credentials."
    fi
    
    # Upload using twine
    if [[ "$target" == "test" ]]; then
        twine upload $repository_url dist/*
    else
        twine upload dist/*
    fi
    
    log_success "Upload completed successfully"
}

create_git_tag() {
    local version=$1
    local tag="v$version"
    
    log_info "Creating git tag: $tag"
    
    # Create annotated tag
    git tag -a "$tag" -m "Release version $version"
    
    # Push tag to origin
    git push origin "$tag"
    
    log_success "Git tag created and pushed: $tag"
}

create_github_release() {
    local version=$1
    local tag="v$version"
    
    log_info "Creating GitHub release..."
    
    # Check if gh CLI is available
    if ! command -v gh &> /dev/null; then
        log_warning "GitHub CLI (gh) not found. Skipping GitHub release creation."
        log_info "You can manually create a release at: https://github.com/WolfgangDremmler/prompt-engineering-toolkit/releases/new"
        return
    fi
    
    # Create release notes
    cat > release_notes.md << EOF
## Release $version

### What's New
- Automated deployment and release process
- Comprehensive documentation updates
- Enhanced testing coverage
- Performance improvements

### Installation
\`\`\`bash
pip install prompt-engineering-toolkit==$version
\`\`\`

### Full Changelog
See the [commit history](https://github.com/WolfgangDremmler/prompt-engineering-toolkit/compare/$(git describe --tags --abbrev=0 HEAD^)...$tag) for detailed changes.

### Verification
To verify the installation:
\`\`\`bash
pet --version
\`\`\`
EOF
    
    # Create GitHub release
    gh release create "$tag" \
        --title "Release $version" \
        --notes-file release_notes.md \
        dist/*
    
    # Cleanup
    rm release_notes.md
    
    log_success "GitHub release created: https://github.com/WolfgangDremmler/prompt-engineering-toolkit/releases/tag/$tag"
}

cleanup() {
    log_info "Cleaning up..."
    
    # Deactivate virtual environment if active
    if [[ -n "$VIRTUAL_ENV" ]]; then
        deactivate
    fi
    
    # Remove build artifacts (optional)
    read -p "Remove build artifacts? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$BUILD_DIR"
        rm -rf *.egg-info
        log_info "Build artifacts removed"
    fi
    
    log_success "Cleanup completed"
}

print_usage() {
    echo "Usage: $0 [test|prod] [options]"
    echo ""
    echo "Arguments:"
    echo "  test    Deploy to Test PyPI"
    echo "  prod    Deploy to Production PyPI"
    echo ""
    echo "Options:"
    echo "  --skip-tests     Skip running tests"
    echo "  --skip-git-tag   Skip creating git tag"
    echo "  --skip-github    Skip GitHub release creation"
    echo "  --help          Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  TESTPYPI_API_TOKEN    API token for Test PyPI"
    echo "  PYPI_API_TOKEN        API token for Production PyPI"
    echo ""
    echo "Examples:"
    echo "  $0 test                    # Deploy to Test PyPI"
    echo "  $0 prod                    # Deploy to Production PyPI"
    echo "  $0 test --skip-tests       # Deploy to Test PyPI without running tests"
}

main() {
    local target=$1
    local skip_tests=false
    local skip_git_tag=false
    local skip_github=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            test|prod)
                target=$1
                shift
                ;;
            --skip-tests)
                skip_tests=true
                shift
                ;;
            --skip-git-tag)
                skip_git_tag=true
                shift
                ;;
            --skip-github)
                skip_github=true
                shift
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    if [[ -z "$target" ]]; then
        log_error "Target not specified"
        print_usage
        exit 1
    fi
    
    if [[ "$target" != "test" && "$target" != "prod" ]]; then
        log_error "Invalid target: $target"
        print_usage
        exit 1
    fi
    
    # Deployment process
    log_info "Starting deployment process for $target environment..."
    
    # Trap cleanup on exit
    trap cleanup EXIT
    
    check_requirements
    setup_environment
    
    if [[ "$skip_tests" != true ]]; then
        run_tests
    else
        log_warning "Skipping tests as requested"
    fi
    
    VERSION=$(get_version)
    check_version_tag "$VERSION"
    
    build_package
    test_package
    
    upload_to_pypi "$target"
    
    if [[ "$target" == "prod" ]]; then
        if [[ "$skip_git_tag" != true ]]; then
            create_git_tag "$VERSION"
        else
            log_warning "Skipping git tag creation as requested"
        fi
        
        if [[ "$skip_github" != true ]]; then
            create_github_release "$VERSION"
        else
            log_warning "Skipping GitHub release creation as requested"
        fi
    fi
    
    log_success "Deployment completed successfully!"
    
    if [[ "$target" == "test" ]]; then
        log_info "Test deployment URL: https://test.pypi.org/project/$PACKAGE_NAME/$VERSION/"
        log_info "Install with: pip install --index-url https://test.pypi.org/simple/ $PACKAGE_NAME==$VERSION"
    else
        log_info "Production deployment URL: https://pypi.org/project/$PACKAGE_NAME/$VERSION/"
        log_info "Install with: pip install $PACKAGE_NAME==$VERSION"
    fi
}

# Run main function with all arguments
main "$@"