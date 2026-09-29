# Archived B action-value audit (29 September 2026)

This is an evaluation-only audit of the 20 archived connected-SCM systems in `results/local_connected_binary_tree_pair_confirmation_20260928`. It issued zero simulator queries and made zero model API calls. `scripts/analysis/audit_connected_action_values.py` checked the archived action, metric, and system file hashes against each cell receipt before reading the executed actions. A second run produced byte-identical summary and per-seed CSV files.

At the fixed cost cap, `coverage_single` made 10 actions across 10 motifs and spent 400; `coverage_pair` made 5 actions across 5 motifs and spent 360. Neither arm revisited a motif at another actuator-value vector in any of the 20 systems. `risk_pair` made 5 actions, spent 360, visited 2.95 motifs on average, and revisited at least one motif at distinct value vectors in all 20 systems. Its direct motif-0 pair design had rank at least two in all 20 systems. The corresponding coverage-pair direct motif-0 design had rank one.

The archived comparison therefore changes **which motifs are sampled and which intervention values are repeated**. It cannot isolate a benefit of adaptive motif selection. This trace-level statement does not imply that the full learner has no interaction information: passive observations and downstream responses may contribute other design rows. A new control should vary actuator values at revisited motifs while matching pair-action cost and total response count; its rule and systems must be frozen before testing. The earlier joint-versus-single gate remains failed.

Files: `per_seed.csv` contains the 60 method-system rows; `summary.csv` contains the three arm summaries; `complete.json` records output hashes and source revision. Reproduce with:

```bash
python scripts/analysis/audit_connected_action_values.py \
  --root results/local_connected_binary_tree_pair_confirmation_20260928 \
  --output /tmp/connected_action_values_replay
```
