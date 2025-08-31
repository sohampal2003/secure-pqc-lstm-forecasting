#!/bin/bash

# Enhanced Time Series Forecasting + PQC Runner Script
# This script provides multiple options for running the system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  original     Run the original LSTM + PQC script"
    echo "  dashboard    Run the Streamlit dashboard"
    echo "  deploy       Deploy the full system with Docker"
    echo "  test         Run tests and quality checks"
    echo "  help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 original    # Run original script"
    echo "  $0 dashboard   # Start Streamlit dashboard"
    echo "  $0 deploy      # Deploy full system"
    echo ""
    echo "Default: Runs the original script if no option specified"
}

# Function to run original script
run_original() {
    print_status "Running original LSTM + PQC script..."
    
    # Check if Python dependencies are installed
    if ! python -c "import torch, yfinance, pqcrypto" 2>/dev/null; then
        print_warning "Installing Python dependencies..."
        pip install -r requirements.txt
    fi
    
    # Run the original script
    python main.py
    
    print_success "Original script completed"
}

# Function to run Streamlit dashboard
run_dashboard() {
    print_status "Starting Streamlit dashboard..."
    
    # Check if Streamlit is installed
    if ! python -c "import streamlit" 2>/dev/null; then
        print_warning "Installing Streamlit dependencies..."
        pip install streamlit plotly
    fi
    
    # Start Streamlit dashboard
    print_status "Dashboard will be available at: http://localhost:8501"
    print_status "Press Ctrl+C to stop the dashboard"
    
    streamlit run streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0
}

# Function to deploy full system
deploy_system() {
    print_status "Deploying full system with Docker..."
    
    # Check if deploy script exists
    if [ ! -f "deploy.sh" ]; then
        print_error "deploy.sh script not found. Please ensure you're in the correct directory."
        exit 1
    fi
    
    # Make deploy script executable and run it
    chmod +x deploy.sh
    ./deploy.sh deploy
}

# Function to run tests
run_tests() {
    print_status "Running tests and quality checks..."
    
    # Check if test dependencies are installed
    if ! python -c "import pytest, flake8, black" 2>/dev/null; then
        print_warning "Installing test dependencies..."
        pip install pytest pytest-cov flake8 black isort bandit safety
    fi
    
    # Run linting
    print_status "Running code quality checks..."
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || true
    black --check --diff . || true
    isort --check-only --diff . || true
    
    # Run security checks
    print_status "Running security checks..."
    bandit -r . -f json -o bandit-report.json || true
    safety check --json --output safety-report.json || true
    
    # Run tests if they exist
    if [ -f "test_*.py" ] || [ -d "tests" ]; then
        print_status "Running tests..."
        pytest --cov=./ --cov-report=xml --cov-report=html || true
    else
        print_warning "No test files found. Skipping test execution."
    fi
    
    print_success "Tests and quality checks completed"
}

# Function to check system status
check_status() {
    print_status "Checking system status..."
    
    # Check if Docker is running
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        print_status "Docker is running"
        
        # Check if containers are running
        if [ -f "docker-compose.yml" ]; then
            print_status "Docker Compose status:"
            docker-compose ps 2>/dev/null || print_warning "Docker Compose not configured"
        fi
    else
        print_warning "Docker is not running or not installed"
    fi
    
    # Check Python environment
    print_status "Python environment:"
    python --version
    pip list | grep -E "(torch|streamlit|plotly|pqcrypto)" || print_warning "Some dependencies not installed"
}

# Main script logic
main() {
    case "${1:-original}" in
        "original")
            run_original
            ;;
        "dashboard")
            run_dashboard
            ;;
        "deploy")
            deploy_system
            ;;
        "test")
            run_tests
            ;;
        "status")
            check_status
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_warning "Unknown option: $1"
            print_status "Running original script as default..."
            run_original
            ;;
    esac
}

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    print_error "main.py not found. Please run this script from the project directory."
    exit 1
fi

# Run main function with all arguments
main "$@"
