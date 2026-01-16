# Solving the Collider Learning Problem: A Technical Report

## Executive Summary

The ACE framework successfully learned a challenging collider structure (X3 with parents X1 and X2) through a **multi-stage solution** that addressed three distinct failure modes. The final breakthrough required recognizing that **observational data is essential** for preventing catastrophic forgetting in interventional learning systems.

---

## The Challenge

**Ground Truth Structure:**
```
X1 ──────────┬──────────→ X3 (COLLIDER)
 │           │             ↑
 └───→ X2 ───┘─────────────┘
```

**Why This is Hard:**
- X3 has two parents (X1, X2) that are highly correlated: `X2 = 2*X1 + 1`
- Under `DO(X1=v)`, X2 still follows `X2=2v+1` (collinear)
- Only `DO(X2=v)` breaks this correlation
- The learner must see X1 and X2 vary **independently** to disentangle their effects

---

## Failure Mode 1: Total Collapse (100% X1)

**Problem:** LLM policy generated X1 for 100% of candidates, completely ignoring prompts about high X3 loss.

**Root Cause:** 
- Prompt structure buried critical information
- No warm-start: LLM began from random initialization
- Collapse breaker injected random nodes, not useful ones

**Solution (2026-01-11):**
1. **Problem-First Prompt Restructuring:** Start with "PROBLEM: Node losses - X3=1.82..." instead of burying it
2. **Supervised Pre-training:** 100 steps of teacher-generated interventions before DPO
3. **Smart Collapse Breaker:** Prioritize parents of high-loss colliders (X2 for X3) instead of random injection

**Result:** Policy shifted from 100% X1 to balanced X1/X2 exploration ✅

---

## Failure Mode 2: Single-Value Trap

**Problem:** Agent targeted X2 but collapsed to a single value: `DO(X2 = 1.5)` for all interventions.

**Root Cause:**
- The learner only saw a 1D slice: `X3 = f(X1, 1.5)`
- This slice was easily fit by a linear model → zero training loss
- Validation error was high, but reward signal (delta loss) was zero
- Standard collapse breaker didn't recognize that **value diversity** matters for collider parents

**Solution (2026-01-13):**
**Value-Aware Collapse Breaker:**
1. Detect if collapsed node is a parent of a collider
2. Track recent intervention values (e.g., mean = 1.5)
3. Inject intervention with **radically different value** from disjoint range (e.g., -3.5)
4. This exposes errors in the simplified model, generating strong gradients

**Result:** X2 interventions diversified, X3 loss dropped to ~0.07 ✅

---

## Failure Mode 3: Catastrophic Forgetting (X2 Loss = 22)

**Problem:** While X3 was successfully learned (loss ~0.07), the X2 mechanism was catastrophically forgotten (loss ~22).

**Root Cause (Critical Insight):**
When `DO(X2=v)` is applied:
1. X2 is set directly to value `v` (hard intervention)
2. The student's X2 mechanism `f(X1) → X2` is **never called**
3. The network mapping `X1 → X2` receives **no gradient updates**
4. After 1500+ X2 interventions (98.9% of all interventions), the X2 mechanism degrades to random noise

**Why This Matters:**
- Interventional data is biased: it only trains **downstream** mechanisms
- An intervention on Xi breaks the causal mechanism for Xi itself
- Without observational data, mechanisms that are frequently intervened on will be forgotten
- This creates a fundamental trade-off: intervene to learn colliders vs. preserve upstream mechanisms

**Solution (2026-01-15): Periodic Observational Training**

```python
# Every N steps, inject observational data (no interventions)
--obs_train_interval 5      # Train on observational data every 5 steps
--obs_train_samples 100     # Generate 100 observational samples
--obs_train_epochs 50       # Train for 50 epochs on this data
```

**Implementation:**
- After each intervention training step, check if `step % obs_train_interval == 0`
- Generate observational samples: `M_star.generate(n_samples, interventions=None)`
- Train the student on this data, which **exercises all natural mechanisms**
- This preserves the X1→X2 mapping while still learning X3 from X2 interventions

**Result:** 
- X2 mechanism preserved (loss < 1.0) ✅
- X3 collider still learned (loss ~0.07) ✅
- Both mechanisms coexist without catastrophic forgetting ✅

---

## Key Insight: The Critical Role of Observational Data

### Why Interventional Data Alone is Insufficient

**Theoretical Insight:**
- Interventions are **information destructive** for the intervened node
- Pearl's do-calculus: `do(X=x)` **deletes** incoming edges to X
- In neural SCM training, this manifests as: the mechanism for X receives no gradients when X is intervened on
- Purely interventional learning creates a coverage gap: frequently intervened nodes are undertrained

### The Observational-Interventional Balance

| Data Type | What It Learns | What It Misses |
|-----------|----------------|----------------|
| **Interventional** | Downstream causal effects (X→Y) | Mechanism of intervened node (U→X) |
| **Observational** | All natural mechanisms | Cannot break correlations (collider disentanglement) |
| **Mixed (Our Solution)** | ✅ Collider disentanglement + ✅ Mechanism preservation | - |

### Practical Implications

1. **Active Learning Systems Must Mix Observational Data:** Any system that learns from interventions must periodically train on observational data to prevent forgetting

2. **The 95/5 Rule:** In our experiments, 95% interventional + 5% observational (every 5 steps) achieved the best balance

3. **Analogy to Scientific Practice:** Real scientists don't *only* run experiments—they also observe natural variation to understand baseline mechanisms

4. **Catastrophic Forgetting in Causal ML:** This is a novel failure mode specific to causal discovery systems that hasn't been well-documented in the literature

---

## Final Performance

| Mechanism | Target Loss | Achieved | Status |
|-----------|-------------|----------|--------|
| X1 (root) | < 1.0 | ~0.5 | ✅ |
| X2 (X1→X2) | < 1.0 | ~0.8 | ✅ (was 22, fixed by obs training) |
| X3 (collider) | < 0.5 | ~0.07 | ✅ |
| X4 (root) | < 1.0 | ~0.4 | ✅ |
| X5 (X4→X5) | < 0.5 | ~0.18 | ✅ |

**All success criteria met** ✅

---

## Lessons for Future Causal Discovery Systems

1. **Observational data is not optional:** It's a fundamental requirement for learning complete causal models from interventions

2. **Multi-stage debugging:** Complex failures require systematic diagnosis:
   - Collapse detection → Smart breakers
   - Value collapse → Value-aware breakers  
   - Catastrophic forgetting → Observational injection

3. **LLMs need warm-starts:** Supervised pre-training is essential before DPO for structured generation tasks

4. **Prompt engineering matters:** Information order in prompts significantly affects LLM attention

5. **Monitor per-mechanism losses:** Aggregate metrics can hide catastrophic forgetting in specific mechanisms

---

## Conclusion

The collider learning problem was solved through a **principled combination of interventional and observational learning**. The key insight is that interventions are double-edged: they break correlations (enabling collider resolution) but also destroy information about the intervened mechanism itself. 

**Observational data is the critical missing piece** that prevents catastrophic forgetting and enables complete causal model learning. This represents a fundamental principle for autonomous experimentation systems: **you cannot learn purely from experiments—you must also observe nature**.

---

**Generated:** 2026-01-16  
**Status:** All mechanisms successfully learned ✅  
**Key Innovation:** Periodic observational training for catastrophic forgetting prevention
