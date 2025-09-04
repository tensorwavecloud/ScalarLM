#!/bin/bash
#
# ScalarLM Master Test Runner - Unified interface for all test types
# Many tests (integration) require running in the Docker instance.
# For example, To run a single test in the container:
#  % docker exec scalarlm-cray-1 python -m pytest /app/cray/test/integration/test_end_to_end_pipeline.py::test_quick_pipeline -v -s
#
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../../"

# Configuration
DOCKER_IMAGE="${SCALARLM_DOCKER_IMAGE:-scalarlm-cray:latest}"
TEST_ENV="local"  # local, docker, or auto
HEALTH_MODE=false
SERVER_CHECK=true
SHOW_HELP=false

# Test selection flags
RUN_DEPLOYMENT=false
RUN_UNIT=false
RUN_INTEGRATION=false
RUN_ALL=true

print_header() {
    echo ""
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}                 🚀 ScalarLM Test Suite 🚀                  ${NC}"
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${NC}"
}

usage() {
    echo -e "${BOLD}ScalarLM Unified Test Runner${NC}"
    echo ""
    echo "Usage: $0 [TEST_TYPE] [OPTIONS]"
    echo ""
    echo -e "${BOLD}Test Types:${NC}"
    echo "  deployment          🚀 Deployment tests (health, generate, train)"
    echo "  integration        🔗 Integration tests (Docker required)"
    echo ""
    echo "  all                🎯 All test suites (default)"
    echo ""
    echo -e "${BOLD}Environment Options:${NC}"
    echo "  --local            Run tests locally (default)"
    echo "  --docker           Force Docker execution for all tests"
    echo "  --auto             Auto-detect environment (Docker if available)"
    echo ""
    echo -e "${BOLD}Control Options:${NC}"
    echo "  -v, --verbose      Verbose output"
    echo "  -f, --fail-fast    Stop on first failure"
    echo "  -c, --coverage     Generate coverage report"
    echo "  --no-server-check  Skip server availability check"
    echo "  --image IMAGE      Use specific Docker image"
    echo "  -h, --help         Show this help"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo "  $0                          # Run all tests"
    echo "  $0 deployment               # Deployment tests"
    echo "  $0 integration --docker     # Integration tests in Docker"
    echo ""
    echo "  $0 all -v -f                # All tests, verbose, fail-fast"
    echo ""
    echo -e "${BOLD}Prerequisites:${NC}"
    echo "  • For integration tests: ScalarLM server running (docker-compose up -d cray)"
    echo "  • For integration tests: Docker image built (docker-compose build cray)"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        deployment)
            RUN_DEPLOYMENT=true
            RUN_ALL=false
            shift
            ;;
        integration)
            RUN_INTEGRATION=true
            RUN_ALL=false
            shift
            ;;
        all)
            RUN_ALL=true
            shift
            ;;
        --local)
            TEST_ENV="local"
            shift
            ;;
        --docker)
            TEST_ENV="docker"
            shift
            ;;
        --auto)
            TEST_ENV="auto"
            shift
            ;;
        --no-server-check)
            SERVER_CHECK=false
            shift
            ;;
        --image)
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        -v|--verbose)
            export VERBOSE=true
            shift
            ;;
        -f|--fail-fast)
            export FAIL_FAST=true
            shift
            ;;
        -c|--coverage)
            export COVERAGE=true
            shift
            ;;
        -h|--help)
            SHOW_HELP=true
            shift
            ;;
        *)
            # Check if it's a test file path
            if [ -f "$1" ] || [[ "$1" == test/* ]]; then
                TEST_FILES="$TEST_FILES $1"
                shift
            else
                echo -e "${RED}❌ Unknown option: $1${NC}"
                usage
                exit 1
            fi
            ;;
    esac
done

if [ "$SHOW_HELP" = true ]; then
    usage
    exit 0
fi

# Print welcome
print_header
echo "Environment: $TEST_ENV"
echo "Docker Image: $DOCKER_IMAGE"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Environment detection and setup
detect_environment() {
    echo -e "${CYAN}🔍 Detecting environment...${NC}"
    
    # Check if Docker is available
    if command -v docker &> /dev/null && docker info &> /dev/null; then
        echo -e "${GREEN}✓ Docker available${NC}"
        DOCKER_AVAILABLE=true
        
        # Check if ScalarLM container is running
        if docker ps --format "table {{.Names}}" | grep -q "scalarlm-cray"; then
            echo -e "${GREEN}✓ ScalarLM container running${NC}"
            CONTAINER_RUNNING=true
        else
            echo -e "${YELLOW}⚠ ScalarLM container not running${NC}"
            CONTAINER_RUNNING=false
        fi
        
        # Check if Docker image exists
        if docker image inspect "$DOCKER_IMAGE" &>/dev/null; then
            echo -e "${GREEN}✓ Docker image '$DOCKER_IMAGE' found${NC}"
            DOCKER_IMAGE_AVAILABLE=true
        else
            echo -e "${YELLOW}⚠ Docker image '$DOCKER_IMAGE' not found${NC}"
            DOCKER_IMAGE_AVAILABLE=false
        fi
    else
        echo -e "${YELLOW}⚠ Docker not available${NC}"
        DOCKER_AVAILABLE=false
        DOCKER_IMAGE_AVAILABLE=false
        CONTAINER_RUNNING=false
    fi
    
    # Check if ScalarLM server is running
    if [ "$SERVER_CHECK" = true ]; then
        if curl -s http://localhost:8000/health &>/dev/null; then
            echo -e "${GREEN}✓ ScalarLM server running (http://localhost:8000)${NC}"
            SERVER_RUNNING=true
        else
            echo -e "${YELLOW}⚠ ScalarLM server not running${NC}"
            SERVER_RUNNING=false
        fi
    else
        SERVER_RUNNING=false
    fi
    
    echo ""
}

# Test execution functions
run_deployment_tests() {
    echo -e "${BOLD}${CYAN}🚀 DEPLOYMENT TESTS${NC}"
    echo "Running essential tests for fast feedback..."
    echo ""
    
    # Health check if server is running
    if [ "$SERVER_RUNNING" = true ]; then
        echo -e "${BLUE}▶ Health Check${NC}"
        python "$PROJECT_ROOT/test/deployment/health.py"
        
        echo -e "${BLUE}▶ Generate Test${NC}"
        python "$PROJECT_ROOT/test/deployment/generate.py"
        
        echo -e "${BLUE}▶ Train Test${NC}"
        python "$PROJECT_ROOT/test/deployment/train.py"
    fi
    
    echo -e "${GREEN}✓ Deployment tests completed${NC}"
}

run_integration_tests() {
    echo -e "${BOLD}${CYAN}🔗 INTEGRATION TESTS${NC}"
    
    # Check if ScalarLM container is running
    if docker ps --format "table {{.Names}}" | grep -q "scalarlm-cray"; then
        echo "Running integration tests against live ScalarLM container..."
        
        # Ensure test dependencies are installed in the container
        echo "Installing test dependencies in container..."
        docker exec scalarlm-cray-1 pip install -q -r /app/cray/test/requirements-pytest.txt
        
        # Set environment for tests to run against running container
        export PYTHONPATH="$PROJECT_ROOT/infra:$PROJECT_ROOT/sdk:${PYTHONPATH:-}"
        
        # Determine what tests to run
        if [ -n "$TEST_FILES" ]; then
            # Run specific test files inside container
            echo "Running specific test files in container: $TEST_FILES"
            if [ "${VERBOSE:-false}" = true ]; then
                docker exec scalarlm-cray-1 python -m pytest $TEST_FILES -v
            else
                docker exec scalarlm-cray-1 python -m pytest $TEST_FILES
            fi
        else
            # Run all integration tests inside container
            if [ "${VERBOSE:-false}" = true ]; then
                docker exec scalarlm-cray-1 python -m pytest /app/cray/test/integration/ -v
            else
                docker exec scalarlm-cray-1 python -m pytest /app/cray/test/integration/
            fi
        fi
    else
        echo -e "${RED}❌ Integration tests require running ScalarLM container${NC}"
        echo "Start with: docker-compose up -d cray"
        echo "Available containers:"
        docker ps --format "table {{.Names}}\t{{.Status}}"
        return 1
    fi
}



# Main execution
detect_environment

# Track results
TOTAL_SUITES=0
PASSED_SUITES=0
FAILED_SUITES=0
FAILED_NAMES=()

run_test_suite() {
    local name="$1"
    local func="$2"
    
    echo ""
    echo -e "${BOLD}${BLUE}$(printf '=%.0s' $(seq 1 60))${NC}"
    echo -e "${BOLD}${BLUE} Running: $name${NC}"
    echo -e "${BOLD}${BLUE}$(printf '=%.0s' $(seq 1 60))${NC}"
    
    TOTAL_SUITES=$((TOTAL_SUITES + 1))
    
    if $func; then
        echo -e "${GREEN}✅ $name - PASSED${NC}"
        PASSED_SUITES=$((PASSED_SUITES + 1))
    else
        echo -e "${RED}❌ $name - FAILED${NC}"
        FAILED_SUITES=$((FAILED_SUITES + 1))
        FAILED_NAMES+=("$name")
        
        if [ "${FAIL_FAST:-false}" = true ]; then
            echo -e "${RED}Stopping due to --fail-fast${NC}"
            exit 1
        fi
    fi
}

# Execute selected test suites
if [ "$RUN_ALL" = true ]; then
    run_test_suite "Deployment Tests" run_deployment_tests
    run_test_suite "Integration Tests" run_integration_tests
elif [ "$RUN_DEPLOYMENT" = true ]; then
    run_test_suite "Deployment Tests" run_deployment_tests
elif [ "$RUN_INTEGRATION" = true ]; then
    run_test_suite "Integration Tests" run_integration_tests
fi

# Final summary
echo ""
echo -e "${BOLD}${BLUE}$(printf '=%.0s' $(seq 1 60))${NC}"
echo -e "${BOLD}${BLUE}                    📊 FINAL SUMMARY${NC}"
echo -e "${BOLD}${BLUE}$(printf '=%.0s' $(seq 1 60))${NC}"
echo ""
echo "Total Suites: $TOTAL_SUITES"
echo -e "Passed:       ${GREEN}$PASSED_SUITES${NC}"
echo -e "Failed:       ${RED}$FAILED_SUITES${NC}"

if [ ${#FAILED_NAMES[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed Suites:${NC}"
    for name in "${FAILED_NAMES[@]}"; do
        echo "  - $name"
    done
fi

echo ""
if [ $FAILED_SUITES -eq 0 ]; then
    echo -e "${BOLD}${GREEN}🎉 ALL TESTS PASSED! 🎉${NC}"
    exit 0
else
    echo -e "${BOLD}${RED}💥 SOME TESTS FAILED 💥${NC}"
    exit 1
fi
