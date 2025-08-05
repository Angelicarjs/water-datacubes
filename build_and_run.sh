#!/bin/bash

# Water Analysis Framework - Build and Run Script
# This script builds and runs the Docker environment for water analysis

set -e  # Exit on any error

echo "🌊 Water Analysis Framework - Docker Setup"
echo "=========================================="

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

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are available"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data
    mkdir -p outputs
    mkdir -p logs
    mkdir -p notebooks
    
    print_success "Directories created successfully"
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    
    docker-compose build
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Run analysis
run_analysis() {
    print_status "Running water analysis..."
    
    docker-compose up water-analysis
    
    print_success "Analysis completed"
}

# Run Jupyter (interactive mode)
run_jupyter() {
    print_status "Starting Jupyter Lab..."
    print_warning "Jupyter will be available at: http://localhost:8888"
    
    docker-compose --profile jupyter up jupyter
}

# Show help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  build     Build the Docker image"
    echo "  run       Run the water analysis"
    echo "  jupyter   Start Jupyter Lab for interactive analysis"
    echo "  all       Build and run analysis (default)"
    echo "  clean     Clean up Docker containers and images"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build     # Only build the image"
    echo "  $0 run       # Run analysis (requires built image)"
    echo "  $0 jupyter   # Start Jupyter Lab"
    echo "  $0 all       # Build and run analysis"
}

# Clean up
clean_up() {
    print_status "Cleaning up Docker resources..."
    
    docker-compose down
    docker system prune -f
    
    print_success "Cleanup completed"
}

# Main script logic
main() {
    case "${1:-all}" in
        "build")
            check_docker
            create_directories
            build_image
            ;;
        "run")
            check_docker
            run_analysis
            ;;
        "jupyter")
            check_docker
            create_directories
            run_jupyter
            ;;
        "all")
            check_docker
            create_directories
            build_image
            run_analysis
            ;;
        "clean")
            clean_up
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 