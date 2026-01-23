#!/bin/bash
echo "Checking ACE repository version..."
echo ""
echo "run_all.sh check:"
if grep -q "Job Orchestrator" run_all.sh; then
    echo "  ✓ NEW VERSION (job orchestrator)"
else
    echo "  ✗ OLD VERSION (direct execution)"
fi
echo ""
echo "jobs/ directory check:"
if [ -d "jobs" ]; then
    echo "  ✓ jobs/ directory exists"
    ls jobs/
else
    echo "  ✗ jobs/ directory missing"
fi
echo ""
echo "To update: git pull"
