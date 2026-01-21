# Results Directory
## Evidence Log for Paper Claims

This directory contains documented evidence for all claims made in `paper/paper.tex`.

---

## Purpose

**Document every experimental result immediately after obtaining it.**

This creates a clear audit trail: raw experimental data → documented evidence → paper claims → reviewable science.

**Current Status (Jan 21, 2026):**
- ✅ Baselines documented (Round-Robin 1.99, Random 2.17, Max-Var 2.09, PPO 2.18*)
- ✅ Root learner documented (98.7% improvement)
- ✅ Additional domains documented (Duffing, Phillips complete)
- ⏳ ACE main results pending (need fresh run with Jan 21 fixes)

See `RESULTS_LOG.md` for all documented findings.

---

## Directory Structure

```
results/
├── README.md                          # This file
├── RESULTS_LOG.md                     # Running log of all findings
├── claim_evidence/                    # Evidence for specific claims
│   ├── collider_identification.md     # Evidence for collider superiority
│   ├── dpo_vs_ppo.md                  # Evidence for DPO > PPO
│   ├── early_stopping.md              # Evidence for 80% speedup
│   └── ...
├── experimental_runs/                 # Links/summaries of key runs
│   ├── run_20260120_summary.md
│   ├── run_20260121_summary.md
│   └── ...
└── figures/                           # Generated figures for paper
    ├── learning_curves.png
    ├── intervention_distribution.png
    └── ...
```

---

## Workflow

### When You Get A Result:

1. **Document It Immediately**
   ```bash
   # Add entry to RESULTS_LOG.md with:
   # - Date
   # - What was tested
   # - Key findings
   # - Which paper claim it supports
   ```

2. **Extract Key Numbers**
   ```bash
   # If it's a table value, add to claim_evidence/
   # Include: metric, value, source file, date
   ```

3. **Generate Figures**
   ```bash
   # Save publication-ready figures to figures/
   # Include metadata about what run generated it
   ```

4. **Link to Paper**
   ```bash
   # Note which paper section/line this supports
   # Example: "Supports line 439: superior collider identification"
   ```

---

## Example Entry Format

### In RESULTS_LOG.md:
```markdown
## 2026-01-21: Baseline Comparison Complete

**Run:** `logs copy/baselines_20260120_142711_23026272`
**Date:** January 21, 2026
**Status:** ✅ Complete

**Key Findings:**
- Round-Robin: 1.9859 ± 0.0402 (BEST baseline)
- Random: 2.1709 ± 0.0436
- Max-Variance: 2.0924 ± 0.0519
- PPO: 2.1835 ± 0.0342 (has bug - needs rerun)

**Paper Claims Supported:**
- Line 362-378: All four baselines implemented ✅
- Line 734: "DPO outperforms PPO" (tentative - PPO has bug)

**Action Items:**
- [ ] Fix PPO shape mismatch bug
- [ ] Rerun PPO baseline
- [ ] Update paper Table 1 with these values
```

---

## Templates

### Quick Result Entry Template
```markdown
## YYYY-MM-DD: [Brief Title]

**Run ID:** [path/to/logs]
**Experiment:** [5-node synthetic / Duffing / Phillips / etc.]
**Status:** [✅ Complete / ⏳ Running / ❌ Failed]

**Key Metrics:**
- Metric 1: [value]
- Metric 2: [value]

**Paper Claims Supported:**
- Line XXX: "[claim text]" → [✅ Supported / ⚠️ Partial / ❌ Not supported]

**Evidence Files:**
- [path/to/data.csv]
- [path/to/figure.png]

**Notes:**
[Any caveats, issues, or next steps]
```

### Claim Evidence Template
```markdown
# Evidence for: [Specific Claim]

**Paper Location:** Line XXX
**Claim:** "[exact quote from paper]"

---

## Supporting Evidence

### Run 1: [Date]
- **Source:** [path/to/logs]
- **Result:** [value]
- **Status:** [✅/⚠️/❌]

### Run 2: [Date]
- **Source:** [path/to/logs]
- **Result:** [value]
- **Status:** [✅/⚠️/❌]

---

## Summary
- **Overall Status:** [✅ Well supported / ⚠️ Needs more data / ❌ Not supported]
- **Confidence:** [High / Medium / Low]
- **Action Items:** [What's needed to strengthen this claim]
```

---

## Integration with Paper

### Before Writing a Claim:
1. Check if you have evidence in `results/`
2. If not, run experiment first
3. Document result before writing claim

### While Writing Paper:
1. Reference specific evidence files
2. Use actual numbers from RESULTS_LOG.md
3. Don't use placeholders - use real data or mark as TODO

### Before Submission:
1. Review RESULTS_LOG.md
2. Verify every claim has supporting evidence
3. Check that all evidence files are linked
4. Ensure no placeholders remain in paper

---

## Quick Commands

### Add a new result:
```bash
# Open the log
code results/RESULTS_LOG.md

# Append new entry (use template above)
```

### Check which claims need evidence:
```bash
# Search paper for placeholders
grep -n "\[.*\]" paper/paper.tex

# Cross-reference with RESULTS_LOG.md
```

### Generate summary for reviewers:
```bash
# Create evidence summary
python scripts/generate_evidence_summary.py
```

---

## Best Practices

1. **Document IMMEDIATELY** - Don't wait until paper writing
2. **Include RAW PATHS** - Always link back to source data
3. **Note CAVEATS** - If result has issues, document them
4. **Version EVERYTHING** - Note git commit, date, run ID
5. **Cross-REFERENCE** - Link result → claim → paper line

---

## Common Mistakes to Avoid

❌ **DON'T:**
- Write paper claims before having results
- Use aspirational/hoped-for results
- Forget to document negative results
- Skip documenting partial results

✅ **DO:**
- Document every experimental run
- Note when results contradict expectations
- Update log when reruns change numbers
- Keep evidence files organized

---

## Status Dashboard

Create a quick view of what's documented:

| Paper Claim | Evidence Status | Last Updated | Notes |
|-------------|----------------|--------------|-------|
| Collider ID superior | ⏳ Pending | - | Need fresh ACE run |
| DPO > PPO | ⚠️ Tentative | 2026-01-21 | PPO has bug |
| 80% speedup | ❌ Contradicted | 2026-01-21 | Current run slow |
| Root learner works | ✅ Strong | 2026-01-21 | 98.7% improvement |
| ... | ... | ... | ... |

---

**Remember:** Good science is reproducible science. Document everything!
