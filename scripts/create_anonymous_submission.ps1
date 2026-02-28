# Create Anonymous Submission Repository for ICML 2026
# PowerShell version
#
# This script creates a clean, anonymous copy of the ACE codebase
# suitable for paper submission.

$ErrorActionPreference = "Stop"

Write-Host "=================================================="
Write-Host "ACE Anonymous Submission Repository Generator"
Write-Host "=================================================="
Write-Host ""

# Configuration
$ANON_DIR = "..\ACE-anonymous-submission"
$ANON_NAME = "Anonymous Researcher"
$ANON_EMAIL = "anonymous@institution.edu"

# Check if target directory exists
if (Test-Path $ANON_DIR) {
    Write-Host "Warning: Directory $ANON_DIR already exists" -ForegroundColor Yellow
    $response = Read-Host "Do you want to remove it and continue? (y/n)"
    if ($response -ne "y") {
        Write-Host "Aborted."
        exit 1
    }
    Remove-Item -Recurse -Force $ANON_DIR
}

Write-Host "Creating anonymous submission directory: $ANON_DIR"
New-Item -ItemType Directory -Path $ANON_DIR | Out-Null

# Initialize new git repository with anonymous config
Write-Host "Initializing anonymous git repository..."
Set-Location $ANON_DIR
git init
git config user.name "$ANON_NAME"
git config user.email "$ANON_EMAIL"
Set-Location ..

Write-Host "Copying files (excluding git history and temporary files)..."

# Files and directories to exclude
$excludes = @(
    '.git',
    '*.pyc',
    '__pycache__',
    '.pytest_cache',
    'htmlcov',
    '.coverage',
    '*.egg-info',
    'dist',
    'build',
    '.DS_Store',
    '*.swp',
    '*~',
    'ANONYMIZE.md',
    'create_anonymous_submission.sh',
    'create_anonymous_submission.ps1'
)

# Copy files using robocopy (Windows native)
$excludeArgs = $excludes | ForEach-Object { "/XF $_" }
robocopy . "$ANON_DIR" /E /XD .git __pycache__ .pytest_cache htmlcov dist build /XF *.pyc .coverage *.egg-info .DS_Store *.swp *~ ANONYMIZE.md SUBMISSION_CHECKLIST.md create_anonymous_submission.sh create_anonymous_submission.ps1 /NFL /NDL /NJH /NJS

# Create initial commit in anonymous repo
Write-Host "Creating initial anonymous commit..."
Set-Location $ANON_DIR
git add .
git commit -m "Initial commit - ACE: Active Causal Experimentalist codebase"

# Create submission archive
Write-Host "Creating submission archive..."
$ARCHIVE_NAME = "ACE-submission-$(Get-Date -Format 'yyyyMMdd').zip"
git archive --format=zip --output="..\$ARCHIVE_NAME" HEAD

Set-Location ..

Write-Host ""
Write-Host "[SUCCESS] Anonymous submission repository created successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Location: $ANON_DIR"
Write-Host "Archive: .\$ARCHIVE_NAME"
Write-Host ""
Write-Host "Verification steps:"
Write-Host "  1. Check git config:"
Write-Host "     cd $ANON_DIR; git config user.name; git config user.email"
Write-Host ""
Write-Host "  2. Check git log:"
Write-Host "     cd $ANON_DIR; git log --pretty=format:'%an <%ae> - %s'"
Write-Host ""
Write-Host "  3. Review files for any identifying information"
Write-Host ""
Write-Host "IMPORTANT: Review the anonymous repository before submission!" -ForegroundColor Yellow
Write-Host ""
