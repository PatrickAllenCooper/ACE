@echo off
REM Local experiment execution for Windows
REM Runs all remaining experiments on NVIDIA RTX 3080

echo ============================================
echo LOCAL EXPERIMENT EXECUTION
echo ============================================
echo.
echo This will run ALL remaining experiments locally:
echo 1. Complex 15-node SCM with PPO (20h)
echo 2. Remaining ablations (4-5h)
echo 3. ACE without oracle (5h)
echo.
echo Total estimated time: 30 hours
echo.
echo Press Ctrl+C to cancel, or
pause

cd /d "%~dp0"

python -u scripts/runners/run_all_experiments_local.py

echo.
echo ============================================
echo Experiments complete
echo ============================================
pause
