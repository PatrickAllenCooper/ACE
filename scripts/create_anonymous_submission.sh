#!/bin/bash
#
# Create Anonymous Submission Repository for ICML 2026
# 
# This script creates a clean, anonymous copy of the ACE codebase
# suitable for paper submission.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================================="
echo "ACE Anonymous Submission Repository Generator"
echo "=================================================="
echo ""

# Configuration
ANON_DIR="../ACE-anonymous-submission"
ANON_NAME="Anonymous Researcher"
ANON_EMAIL="anonymous@institution.edu"

# Check if target directory exists
if [ -d "$ANON_DIR" ]; then
    echo -e "${YELLOW}Warning: Directory $ANON_DIR already exists${NC}"
    read -p "Do you want to remove it and continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$ANON_DIR"
    else
        echo "Aborted."
        exit 1
    fi
fi

echo "Creating anonymous submission directory: $ANON_DIR"
mkdir -p "$ANON_DIR"

# Initialize new git repository with anonymous config
echo "Initializing anonymous git repository..."
cd "$ANON_DIR"
git init
git config user.name "$ANON_NAME"
git config user.email "$ANON_EMAIL"

# Return to original directory
cd - > /dev/null

echo "Copying files (excluding git history and temporary files)..."

# Copy all relevant files, excluding git history and build artifacts
rsync -av \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='htmlcov' \
    --exclude='.coverage' \
    --exclude='*.egg-info' \
    --exclude='dist' \
    --exclude='build' \
    --exclude='.DS_Store' \
    --exclude='*.swp' \
    --exclude='*~' \
    --exclude='ANONYMIZE.md' \
    --exclude='SUBMISSION_CHECKLIST.md' \
    --exclude='create_anonymous_submission.sh' \
    --exclude='create_anonymous_submission.ps1' \
    ./ "$ANON_DIR/"

# Create initial commit in anonymous repo
echo "Creating initial anonymous commit..."
cd "$ANON_DIR"
git add .
git commit -m "Initial commit - ACE: Active Causal Experimentalist codebase"

# Create submission archive
echo "Creating submission archive..."
ARCHIVE_NAME="ACE-submission-$(date +%Y%m%d).zip"
git archive --format=zip --output="../$ARCHIVE_NAME" HEAD

cd - > /dev/null

echo ""
echo -e "${GREEN}[SUCCESS] Anonymous submission repository created successfully!${NC}"
echo ""
echo "Location: $ANON_DIR"
echo "Archive: ../$ARCHIVE_NAME"
echo ""
echo "Verification steps:"
echo "  1. Check git config:"
echo "     cd $ANON_DIR && git config user.name && git config user.email"
echo ""
echo "  2. Check git log:"
echo "     cd $ANON_DIR && git log --pretty=format:'%an <%ae> - %s'"
echo ""
echo "  3. Review files for any identifying information"
echo ""
echo -e "${YELLOW}IMPORTANT: Review the anonymous repository before submission!${NC}"
echo ""
