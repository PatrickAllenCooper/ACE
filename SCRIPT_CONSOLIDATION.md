# Script Consolidation - Migration Guide

## Overview

Consolidated **21 shell scripts** into **1 unified CLI** (`ace.sh`) + **5 SLURM job templates** (`jobs/*.sh`)

**Before:** 21 scattered scripts 
**After:** 6 total files (1 CLI + 5 job templates)

---

## Migration Mapping

### Old Scripts -> New Commands

| Old Script | New Command | Status |
|------------|-------------|--------|
| `run_all.sh` | `./ace.sh run all` | [DONE] Replaced |
| `scripts/run_all_multi_seed.sh` | `./ace.sh run-multi-seed [n]` | [DONE] Replaced |
| `scripts/run_ablations.sh` | `./ace.sh run-ablations` | [DONE] Replaced |
| `scripts/run_obs_ratio_ablation.sh` | `./ace.sh run-obs-ablation` | [DONE] Replaced |
| `scripts/run_ace_experiments.sh` | `./ace.sh run ace` | [DONE] Replaced |
| `scripts/process_all_results.sh` | `./ace.sh process <dir>` | [DONE] Replaced |
| `scripts/consolidate_multi_runs.sh` | `./ace.sh consolidate-multi-seed <dir>` | [DONE] Replaced |
| `scripts/extract_ace.sh` | `./ace.sh extract ace <dir>` | [DONE] Replaced |
| `scripts/extract_baselines.sh` | `./ace.sh extract baselines <dir>` | [DONE] Replaced |
| `scripts/verify_claims.sh` | `./ace.sh verify <dir>` | [DONE] Replaced |
| `scripts/sync_results_from_hpc.sh` | `./ace.sh sync-hpc` | [DONE] Replaced |
| `scripts/test_jan21_fixes.sh` | `./ace.sh test` | [DONE] Replaced |
| `scripts/pipeline_test.sh` | `./ace.sh test` | [DONE] Replaced |
| `scripts/cleanup.sh` | `./ace.sh clean` | [DONE] Replaced |
| `scripts/check_version.sh` | `./ace.sh check-version` | [DONE] Replaced |
| `scripts/launch_training.sh` | *Deprecated* | [WARNING] Remove |

**Keep (SLURM job templates):**
- `jobs/run_ace_main.sh` - Called by `sbatch`
- `jobs/run_baselines.sh` - Called by `sbatch`
- `jobs/run_complex_scm.sh` - Called by `sbatch`
- `jobs/run_duffing.sh` - Called by `sbatch`
- `jobs/run_phillips.sh` - Called by `sbatch`

---

## Quick Reference

### Running Experiments

```bash
# Old way
./run_all.sh

# New way
./ace.sh run all
```

```bash
# Old way
./scripts/run_all_multi_seed.sh

# New way
./ace.sh run-multi-seed 5
```

```bash
# Old way
./scripts/run_ablations.sh

# New way
./ace.sh run-ablations
```

### Post-Processing

```bash
# Old way
./scripts/process_all_results.sh results/paper_20260121_123456

# New way
./ace.sh process results/paper_20260121_123456
```

```bash
# Old way
./scripts/sync_results_from_hpc.sh

# New way
./ace.sh sync-hpc
```

### Utilities

```bash
# Old way
./scripts/cleanup.sh

# New way
./ace.sh clean
```

```bash
# Old way
./scripts/pipeline_test.sh

# New way
./ace.sh test
```

---

## Benefits of Consolidation

1. **Single entry point** - One command to remember (`./ace.sh`)
2. **Consistent interface** - All operations follow same pattern
3. **Better discoverability** - `./ace.sh help` shows everything
4. **Easier maintenance** - Changes in one place
5. **Cleaner repository** - 21 files -> 6 files (72% reduction)
6. **Type safety** - Built-in validation and error messages
7. **Better logging** - Color-coded output for all operations

---

## Complete Command Reference

```bash
# Show all available commands
./ace.sh help

# Run experiments
./ace.sh run ace # Single experiment
./ace.sh run all # All 5 experiments
./ace.sh run-multi-seed 5 # Multi-seed validation
./ace.sh run-ablations # Ablation studies
./ace.sh run-obs-ablation # Obs ratio ablation

# Post-process results
./ace.sh process <dir> # Full post-processing
./ace.sh extract ace <dir> # Extract ACE metrics
./ace.sh extract baselines <dir> # Extract baseline metrics
./ace.sh verify <dir> # Verify claims
./ace.sh consolidate-multi-seed <dir> # Consolidate multi-seed

# HPC workflows
./ace.sh sync-hpc # Sync from HPC

# Utilities
./ace.sh test # Run tests
./ace.sh clean # Clean up
./ace.sh check-version # Check versions
```

---

## Files to Remove

Once migration is verified, these scripts can be safely removed:

```bash
# Root level
rm run_all.sh

# Scripts directory (keep Python scripts and job templates)
rm scripts/run_all_multi_seed.sh
rm scripts/run_ablations.sh
rm scripts/run_obs_ratio_ablation.sh
rm scripts/run_ace_experiments.sh
rm scripts/process_all_results.sh
rm scripts/consolidate_multi_runs.sh
rm scripts/extract_ace.sh
rm scripts/extract_baselines.sh
rm scripts/verify_claims.sh
rm scripts/sync_results_from_hpc.sh
rm scripts/test_jan21_fixes.sh
rm scripts/pipeline_test.sh
rm scripts/cleanup.sh
rm scripts/check_version.sh
rm scripts/launch_training.sh
```

**Total to remove:** 16 shell scripts 
**Total to keep:** 1 CLI + 5 job templates = 6 files

---

## Verification

Test the new CLI:

```bash
# Verify syntax
bash -n ace.sh

# Show help
./ace.sh help

# Run pipeline tests
./ace.sh test

# Check environment
./ace.sh check-version
```

---

## Migration Checklist

- [x] Create unified CLI (`ace.sh`)
- [x] Verify syntax and help message
- [x] Test basic commands
- [ ] Update README.md
- [ ] Update guidance_doc.txt
- [ ] Remove old scripts
- [ ] Update tests to use new CLI
- [ ] Test complete workflow with new CLI
- [ ] Commit and push changes

---

## Rollback Plan

If issues arise, old scripts are preserved in git history:

```bash
# Rollback to previous version
git checkout HEAD~1 -- run_all.sh scripts/

# Or restore specific script
git checkout HEAD~1 -- scripts/run_all_multi_seed.sh
```
