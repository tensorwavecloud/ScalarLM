#!/bin/bash
# Integration test runner for ScalarLM deployment tests
# Runs executable test scripts that require a running ScalarLM server
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Set PYTHONPATH to include infra and SDK directories
export PYTHONPATH="$PROJECT_DIR/infra:$PROJECT_DIR/sdk:${PYTHONPATH:-}"

echo "🚀 Running ScalarLM deployment integration tests"
echo "   Project: $PROJECT_DIR"
echo "   PYTHONPATH: $PYTHONPATH"
echo ""
echo "⚠️  NOTE: These tests require a running ScalarLM server!"
echo "   Start server with: ./scalarlm up cpu"
echo ""

cd "$PROJECT_DIR"

# Find all Python files in test/deployment directory, excluding utilities
test_files=()
for file in test/deployment/*.py; do
    # Skip utility files that aren't tests
    if [[ "$file" != *"slurm_utils.py" && "$file" != *"__pycache__"* ]]; then
        test_files+=("$file")
    fi
done

if [ ${#test_files[@]} -eq 0 ]; then
    echo "❌ No test files found in test/deployment/"
    exit 1
fi

# Track test results
total_tests=0
passed_tests=0
failed_tests=0
failed_test_names=()

echo "🔍 Found ${#test_files[@]} deployment test(s) to run:"
for test_file in "${test_files[@]}"; do
    if [[ -f "$test_file" ]]; then
        echo "   - $test_file"
    fi
done
echo ""

# Run each test file
for test_file in "${test_files[@]}"; do
    if [[ -f "$test_file" ]]; then
        test_name=$(basename "$test_file")
        echo "🧪 Running $test_name..."
        
        total_tests=$((total_tests + 1))
        
        # Run the test and capture exit code
        if python "$test_file"; then
            echo "✅ $test_name - PASSED"
            passed_tests=$((passed_tests + 1))
        else
            echo "❌ $test_name - FAILED"
            failed_tests=$((failed_tests + 1))
            failed_test_names+=("$test_name")
        fi
        echo ""
    fi
done

# Print summary
echo "="*60
echo "📊 DEPLOYMENT TEST SUMMARY"
echo "="*60
echo "Total tests:  $total_tests"
echo "Passed:       $passed_tests"
echo "Failed:       $failed_tests"

if [ $failed_tests -gt 0 ]; then
    echo ""
    echo "❌ Failed tests:"
    for failed_test in "${failed_test_names[@]}"; do
        echo "   - $failed_test"
    done
    echo ""
    echo "💡 Make sure ScalarLM server is running: ./scalarlm up cpu"
    exit 1
else
    echo ""
    echo "🎉 All deployment tests passed!"
    exit 0
fi