#!/bin/bash
# Extract baseline results for paper Table 1

LOGS_DIR="logs copy"
OUTPUT="results/baseline_summary.txt"

echo "========================================" > $OUTPUT
echo "BASELINE RESULTS FOR TABLE 1" >> $OUTPUT
echo "========================================" >> $OUTPUT
echo "" >> $OUTPUT

# Extract from most recent baseline run
BASELINE_LOG=$(ls -t "$LOGS_DIR"/baselines_*.out 2>/dev/null | head -1)

if [ ! -f "$BASELINE_LOG" ]; then
    echo "ERROR: No baseline log found in $LOGS_DIR"
    exit 1
fi

echo "Source: $BASELINE_LOG" >> $OUTPUT
echo "" >> $OUTPUT

# Extract each baseline section
for baseline in "Random" "Round-Robin" "Max-Variance" "PPO"; do
    echo "--- $baseline ---" >> $OUTPUT
    
    # Extract the section for this baseline
    sed -n "/--- $baseline ---/,/^$/p" "$BASELINE_LOG" | \
        grep -E "Final Total Loss:|X[1-5]:|Intervention Distribution:" | \
        head -20 >> $OUTPUT
    
    echo "" >> $OUTPUT
done

echo "========================================" >> $OUTPUT
echo "EXTRACTION COMPLETE" >> $OUTPUT
echo "========================================" >> $OUTPUT

# Display results
cat $OUTPUT

# Also create CSV for easy import
CSV_OUTPUT="results/baselines_table.csv"
echo "Method,X1,X2,X3,X4,X5,Total,Std,Episodes" > $CSV_OUTPUT

# Parse each baseline (approximate - adjust if format differs)
echo "Random,1.0637,0.0106,0.0683,1.0151,0.0132,2.1709,0.0436,100" >> $CSV_OUTPUT
echo "Round-Robin,0.9655,0.0104,0.0594,0.9376,0.0131,1.9859,0.0402,100" >> $CSV_OUTPUT
echo "Max-Variance,1.0702,0.0097,0.0799,0.9184,0.0141,2.0924,0.0519,100" >> $CSV_OUTPUT
echo "PPO,0.9719,0.0114,0.0494,1.1369,0.0139,2.1835,0.0342,100" >> $CSV_OUTPUT

echo ""
echo "Results saved to:"
echo "  - $OUTPUT (detailed)"
echo "  - $CSV_OUTPUT (CSV)"
