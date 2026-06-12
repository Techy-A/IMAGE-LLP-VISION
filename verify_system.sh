#!/bin/bash

# System Verification Script for IMAGE-LLP-VISION
# This script checks what's working WITHOUT needing dependencies installed

set -e  # Exit on error

cd "$(dirname "$0")"

echo "=========================================="
echo "IMAGE-LLP-VISION System Verification"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass_count=0
fail_count=0
warn_count=0

# Test 1: Check Python version
echo "1️⃣  Checking Python version..."
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$python_version" == "3.10" ]] || [[ "$python_version" == "3.11" ]]; then
    echo -e "   ${GREEN}✅ Python $python_version (compatible)${NC}"
    ((pass_count++))
elif [[ "$python_version" == "3.13" ]]; then
    echo -e "   ${YELLOW}⚠️  Python $python_version (needs 3.10 or 3.11 for PyTorch)${NC}"
    ((warn_count++))
else
    echo -e "   ${RED}❌ Python $python_version (unsupported)${NC}"
    ((fail_count++))
fi
echo ""

# Test 2: Check required files exist
echo "2️⃣  Checking required files exist..."
required_files=(
    "config.yaml"
    "requirements.txt"
    "main.py"
    "src/csv_loader.py"
    "src/data_models.py"
    "src/arena.py"
    "src/elo_system.py"
    "src/ensemble.py"
)

all_exist=true
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "   ${GREEN}✅${NC} $file"
    else
        echo -e "   ${RED}❌${NC} $file"
        all_exist=false
    fi
done

if [ "$all_exist" = true ]; then
    ((pass_count++))
else
    ((fail_count++))
fi
echo ""

# Test 3: Check module structure
echo "3️⃣  Checking module structure..."
echo "   Core modules:"
core_count=$(find src -maxdepth 1 -name "*.py" -not -name "__*" | wc -l | tr -d ' ')
echo "      Found: $core_count files"

echo "   Model adapters:"
model_count=$(find src/models -name "*_adapter.py" 2>/dev/null | wc -l | tr -d ' ')
echo "      Found: $model_count adapters"

echo "   Scorers:"
scorer_count=$(find src/scoring -name "*.py" -not -name "__*" -not -name "base_*" 2>/dev/null | wc -l | tr -d ' ')
echo "      Found: $scorer_count scorers"

if [[ $core_count -ge 10 ]] && [[ $model_count -ge 4 ]] && [[ $scorer_count -ge 5 ]]; then
    echo -e "   ${GREEN}✅ Module structure looks complete${NC}"
    ((pass_count++))
else
    echo -e "   ${YELLOW}⚠️  Some modules might be missing${NC}"
    ((warn_count++))
fi
echo ""

# Test 4: Syntax check (compile)
echo "4️⃣  Checking Python syntax (compilation)..."
error_count=0
for pyfile in src/*.py src/models/*.py src/scoring/*.py src/visualization/*.py; do
    if [ -f "$pyfile" ] 2>/dev/null; then
        if python3 -m py_compile "$pyfile" 2>/dev/null; then
            : # Success, do nothing
        else
            echo -e "   ${RED}❌${NC} Syntax error in: $pyfile"
            ((error_count++))
        fi
    fi
done 2>/dev/null

if [ $error_count -eq 0 ]; then
    echo -e "   ${GREEN}✅ All Python files compile successfully${NC}"
    ((pass_count++))
else
    echo -e "   ${RED}❌ $error_count files have syntax errors${NC}"
    ((fail_count++))
fi
echo ""

# Test 5: Check test files exist
echo "5️⃣  Checking test coverage..."
test_count=$(find tests -name "test_*.py" 2>/dev/null | wc -l | tr -d ' ')
echo "   Found: $test_count test files"
if [ $test_count -ge 8 ]; then
    echo -e "   ${GREEN}✅ Good test coverage${NC}"
    ((pass_count++))
else
    echo -e "   ${YELLOW}⚠️  Limited test coverage${NC}"
    ((warn_count++))
fi
echo ""

# Test 6: Check configuration
echo "6️⃣  Checking configuration file..."
if grep -q "models:" config.yaml && grep -q "scoring:" config.yaml; then
    echo -e "   ${GREEN}✅ config.yaml has required sections${NC}"
    ((pass_count++))
else
    echo -e "   ${RED}❌ config.yaml missing required sections${NC}"
    ((fail_count++))
fi
echo ""

# Test 7: Check if dependencies are installed (optional)
echo "7️⃣  Checking dependencies (optional)..."
deps_installed=true
for dep in pandas torch transformers; do
    if python3 -c "import $dep" 2>/dev/null; then
        echo -e "   ${GREEN}✅${NC} $dep"
    else
        echo -e "   ${YELLOW}⚠️${NC}  $dep (not installed)"
        deps_installed=false
    fi
done

if [ "$deps_installed" = true ]; then
    echo -e "   ${GREEN}✅ All dependencies installed${NC}"
    ((pass_count++))
else
    echo -e "   ${YELLOW}⚠️  Some dependencies missing (expected if not set up yet)${NC}"
    ((warn_count++))
fi
echo ""

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo -e "${GREEN}✅ Passed: $pass_count${NC}"
echo -e "${YELLOW}⚠️  Warnings: $warn_count${NC}"
echo -e "${RED}❌ Failed: $fail_count${NC}"
echo ""

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}🎉 System structure is valid!${NC}"
    echo ""
    echo "Next steps:"
    if [ $warn_count -gt 0 ]; then
        echo "1. Install Python 3.10 or 3.11 (you have $python_version)"
        echo "2. Create virtual environment: python3.11 -m venv .venv"
        echo "3. Activate: source .venv/bin/activate"
        echo "4. Install dependencies: pip install -r requirements.txt"
        echo "5. Run tests: pytest tests/ -v"
    else
        echo "✅ Ready to run tests: pytest tests/ -v"
    fi
else
    echo -e "${RED}⚠️  Some critical issues found. Review errors above.${NC}"
    exit 1
fi
