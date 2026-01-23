#!/bin/bash
# Extract ACE results for paper Table 1

OUTPUT="results/ace_summary.txt"

# Find most recent ACE run
ACE_DIR=$(ls -td results/paper_*/ace 2>/dev/null | head -1)

if [ -z "$ACE_DIR" ]; then
    echo "ERROR: No ACE results found"
    echo "Expected directory pattern: results/paper_*/ace"
    echo ""
    echo "Have you run ACE yet?"
    echo "Run: sbatch jobs/run_ace_main.sh"
    exit 1
fi

echo "========================================" > $OUTPUT
echo "ACE RESULTS FOR TABLE 1" >> $OUTPUT
echo "========================================" >> $OUTPUT
echo "" >> $OUTPUT
echo "Source: $ACE_DIR" >> $OUTPUT
echo "" >> $OUTPUT

# Extract final losses
echo "Final Mechanism Losses:" >> $OUTPUT
if [ -f "$ACE_DIR/experiment.log" ]; then
    grep "Final mechanism losses:" "$ACE_DIR/experiment.log" >> $OUTPUT
else
    echo "ERROR: No experiment.log found in $ACE_DIR"
    exit 1
fi
echo "" >> $OUTPUT

# Extract training info
echo "Training Information:" >> $OUTPUT
grep -E "episodes trained:|Early stopping|Episode.*Complete" "$ACE_DIR/experiment.log" | tail -5 >> $OUTPUT
echo "" >> $OUTPUT

# Extract intervention distribution
echo "Intervention Distribution:" >> $OUTPUT
if [ -f "$ACE_DIR/metrics.csv" ]; then
    python3 << 'EOF'
import pandas as pd
import sys
import glob

# Find metrics file
metrics_files = glob.glob("results/paper_*/ace/metrics.csv")
if not metrics_files:
    print("ERROR: No metrics.csv found")
    sys.exit(1)

df = pd.read_csv(sorted(metrics_files)[-1])

# Calculate distribution
dist = df['target'].value_counts(normalize=True) * 100
dist = dist.sort_index()

print("Node | Percentage")
print("-----|----------")
for node, pct in dist.items():
    print(f"{node:4s} | {pct:6.2f}%")

# Calculate total loss from final episode
final_ep = df[df['episode'] == df['episode'].max()]
if len(final_ep) > 0:
    final_row = final_ep.iloc[-1]
    node_cols = [col for col in df.columns if col.startswith('loss_')]
    if node_cols:
        total = sum(final_row[col] for col in node_cols)
        print(f"\nFinal Total Loss: {total:.4f}")
        print("\nPer-Node Losses:")
        for col in sorted(node_cols):
            node = col.replace('loss_', '')
            print(f"  {node}: {final_row[col]:.4f}")
EOF
else
    echo "ERROR: No metrics.csv found"
    exit 1
fi >> $OUTPUT

echo "" >> $OUTPUT
echo "========================================" >> $OUTPUT
echo "EXTRACTION COMPLETE" >> $OUTPUT
echo "========================================" >> $OUTPUT

# Display results
cat $OUTPUT

echo ""
echo "Results saved to: $OUTPUT"
