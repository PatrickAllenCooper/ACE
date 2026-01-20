# ACE Project - Suggested Revisions
**Priority-Ordered Action Items**  
**Date:** January 20, 2026

---

## PRIORITY 1: Critical Issues (Implement Before Next HPC Run)

### 1.1 Fix Root Node Learning
**Problem:** X1 and X4 (root nodes) have borderline losses (0.88, 0.94)  
**Root Cause:** Interventions override root distributions; only observational data shows true roots

**Actions:**
- [ ] Increase observational training frequency: `--obs_train_interval 3` (currently 5)
- [ ] Increase observational samples: `--obs_train_samples 200` (currently 100)
- [ ] Increase observational epochs: `--obs_train_epochs 100` (currently 50)
- [ ] Add root-specific distribution fitting phase after each episode
- [ ] Implement weighted loss: 3x weight for root nodes, 1x for intermediates

**Expected Impact:** X1 loss: 0.88 → <0.5, X4 loss: 0.94 → <0.5

**Implementation:**
```python
# In ace_experiments.py, modify observational training section:
if step % args.obs_train_interval == 0:
    obs_samples = args.obs_train_samples  # Increase to 200
    obs_epochs = args.obs_train_epochs     # Increase to 100
    
    # Add root-specific training
    if step % (args.obs_train_interval * 2) == 0:
        root_fitting_phase(M_student, M_star, root_nodes=['X1', 'X4'],
                          n_samples=500, epochs=100)
```

---

### 1.2 Address LLM Policy Collapse
**Problem:** LLM generates X2 for 99.1% of candidates; requires constant hard-cap enforcement

**Actions:**
- [ ] Increase undersampled bonus: `--undersampled_bonus 200.0` (currently 100.0)
- [ ] Reduce hard cap threshold: 70% → 60% (requires code change)
- [ ] Increase supervised pre-training frequency: `--pretrain_interval 25` (currently 50)
- [ ] Increase supervised training steps: `--pretrain_steps 200` (currently 100)
- [ ] Add explicit diversity penalty to reward function
- [ ] Modify prompt to explicitly discourage X2 over-selection

**Expected Impact:** X2 concentration: 69.4% → <60%, X4 interventions: 0.9% → >10%

**Implementation:**
```python
# In ace_experiments.py, modify reward calculation:
diversity_penalty = -50.0 * max(0, recent_target_concentration - 0.5)
final_reward = base_reward + diversity_penalty

# Modify hard cap threshold:
HARD_CAP_THRESHOLD = 0.60  # Was 0.70

# Update prompt template to include:
"""
CRITICAL: The intervention distribution should be balanced.
Current concentration: X2={X2_pct}% (TARGET: <50%)
AVOID over-selecting X2. Explore X1, X4, X5 more frequently.
"""
```

---

### 1.3 Validate ACE Advantage Over Baselines
**Problem:** ACE requires 200 episodes for 8% improvement over PPO (100 episodes)

**Actions:**
- [ ] Run all baselines for 200 episodes to match ACE
- [ ] Compare final losses with fair compute budget
- [ ] Run PPO with hyperparameter tuning
- [ ] Document wall-clock time and compute costs

**Expected Outcome:** Determine if ACE's advantage is due to adaptive policy or simply more training

**Commands:**
```bash
# Run 200-episode baselines
python baselines.py --all_with_ppo --episodes 200 --output results/baselines_200ep

# Tuned PPO
python baselines.py --baseline ppo --episodes 200 \
  --ppo_lr 1e-4 --ppo_clip 0.1 --ppo_epochs 8
```

---

## PRIORITY 2: Validation & Enhancement (1-2 Week Timeline)

### 2.1 Test ACE on Complex 15-Node SCM
**Goal:** Demonstrate strategic advantage on harder problems

**Actions:**
- [ ] Implement ACE (LLM+DPO) policy for complex SCM
- [ ] Run 200-episode experiment
- [ ] Compare against random/smart_random/greedy_collider baselines
- [ ] Analyze collider learning performance specifically

**Success Criterion:** ACE outperforms baselines by >20% on final loss

**Command:**
```bash
python -m experiments.complex_scm --policy ace --episodes 200 \
  --output results/complex_scm_ace
```

---

### 2.2 Implement Multi-Objective Reward
**Goal:** Explicitly optimize for both loss reduction AND intervention diversity

**Actions:**
- [ ] Design multi-objective reward function
- [ ] Add Pareto frontier tracking
- [ ] Implement adaptive weighting (start with diversity, shift to loss)

**Implementation:**
```python
# New reward structure
loss_reward = -delta_loss * 100.0
diversity_reward = entropy(intervention_distribution) * 50.0
coverage_reward = num_unique_targets_explored * 10.0

total_reward = (
    0.6 * loss_reward +      # Primary objective
    0.3 * diversity_reward +  # Encourage exploration
    0.1 * coverage_reward     # Bonus for new targets
)
```

---

### 2.3 Computational Optimization
**Goal:** Reduce compute time by 30-50% without sacrificing performance

**Actions:**
- [ ] Implement early stopping with patience
- [ ] Adaptive episode length (reduce steps as learning stabilizes)
- [ ] Cache LLM embeddings
- [ ] Profile code for bottlenecks

**Implementation:**
```python
# Early stopping
--patience_episodes 20
--min_improvement 0.01

# Adaptive steps
--adaptive_steps
--initial_steps 25
--min_steps 10
--convergence_threshold 0.05
```

---

## PRIORITY 3: Research Enhancements (1-2 Month Timeline)

### 3.1 Ablation Studies
**Goal:** Understand which components contribute to performance

**Experiments:**
- [ ] Smart breaker vs no breaker
- [ ] DPO vs PPO vs Supervised-only
- [ ] Observational training frequency sweep (0, 1, 3, 5, 10 steps)
- [ ] Collider bonus ablation
- [ ] Value diversity breaker impact

**Expected Outcome:** Identify minimal effective feature set

---

### 3.2 Hybrid Training Regime
**Goal:** Better balance interventional and observational learning

**Actions:**
- [ ] Curriculum learning: start observational → gradual interventions
- [ ] Weighted loss by node type (3x roots, 2x intermediates, 1x leaves)
- [ ] Periodic root distribution matching (every 10 episodes)
- [ ] Separate learner models for roots vs mechanisms

**Implementation:**
```python
# Curriculum schedule
episodes_0_50:   80% observational, 20% interventional
episodes_51_100: 50% observational, 50% interventional
episodes_101+:   20% observational, 80% interventional

# Weighted loss
loss_weights = {
    'root': 3.0,      # X1, X4
    'intermediate': 2.0,  # X2
    'collider': 1.5,  # X3
    'leaf': 1.0       # X5
}
```

---

### 3.3 LLM Architecture Experiments
**Goal:** Find optimal model size/architecture for intervention generation

**Actions:**
- [ ] Test smaller models (Qwen2.5-0.5B) with knowledge distillation
- [ ] Compare Qwen vs Phi-3 vs Gemma families
- [ ] Investigate prompt engineering techniques
- [ ] Fine-tune LLM on synthetic intervention tasks

**Models to Test:**
- Qwen/Qwen2.5-0.5B (smaller, faster)
- microsoft/Phi-3-mini-4k (efficient)
- google/gemma-2b (strong reasoning)

---

### 3.4 Theoretical Analysis
**Goal:** Provide formal guarantees and sample complexity bounds

**Research Questions:**
- [ ] Formal proof of identifiability under intervention regimes
- [ ] Sample complexity bounds for root vs intermediate nodes
- [ ] Optimal intervention allocation (MAB/Bayesian optimization)
- [ ] Convergence guarantees for DPO in causal discovery

**Deliverables:**
- Theoretical paper section
- Sample complexity plots
- Comparison to information-theoretic lower bounds

---

## PRIORITY 4: Paper Preparation

### 4.1 Additional Baselines
**Goal:** Comprehensive comparison with SOTA methods

**Baselines to Add:**
- [ ] NOTEARS (continuous optimization for DAG learning)
- [ ] GES (greedy equivalence search)
- [ ] PC algorithm (constraint-based)
- [ ] DirectLiNGAM (linear non-Gaussian)
- [ ] Active learning baselines (BALD, etc.)

---

### 4.2 Visualization & Analysis
**Goal:** Publication-quality figures and comprehensive analysis

**Actions:**
- [ ] Intervention distribution heatmaps over time
- [ ] Learning curves for all methods
- [ ] Mechanism quality visualizations
- [ ] Causal graph recovery accuracy
- [ ] Ablation study summary plots

---

### 4.3 Real-World Domains
**Goal:** Demonstrate practical applicability

**Additional Experiments:**
- [ ] Gene regulatory networks (biology)
- [ ] Climate causality (environmental science)
- [ ] Marketing attribution (business)
- [ ] Healthcare treatment effects (medicine)

---

## Quick Reference: Command-Line Changes for Next Run

```bash
# Immediate fixes for next HPC run
python ace_experiments.py \
  --episodes 200 \
  --obs_train_interval 3 \           # Was 5
  --obs_train_samples 200 \          # Was 100
  --obs_train_epochs 100 \           # Was 50
  --undersampled_bonus 200.0 \       # Was 100.0
  --pretrain_interval 25 \           # Was 50
  --pretrain_steps 200 \             # Was 100
  --smart_breaker \
  --output results/ace_improved
  
# Also run extended baselines
python baselines.py --all_with_ppo --episodes 200 \
  --obs_train_interval 3 \
  --output results/baselines_200ep

# Complex SCM validation
python -m experiments.complex_scm --policy ace --episodes 200
```

---

## Success Metrics for Next Run

### Must Achieve (Critical)
- [ ] X1 loss < 0.5 (currently 0.88)
- [ ] X4 loss < 0.5 (currently 0.94)
- [ ] X2 intervention concentration < 60% (currently 69.4%)
- [ ] X4 receives >10% of interventions (currently 0.9%)

### Should Achieve (Important)
- [ ] ACE total loss < 1.5 (currently 1.92)
- [ ] ACE outperforms 200-ep baselines by >15%
- [ ] DPO preference margin remains positive
- [ ] LLM generates <80% X2 (currently 99.1%)

### Nice to Have (Aspirational)
- [ ] All node losses < 0.3
- [ ] Balanced intervention distribution (all nodes 15-25%)
- [ ] <6 hours total runtime (currently 9h 11m)
- [ ] ACE on complex SCM outperforms baselines by >30%

---

## Timeline & Milestones

### Week 1 (Jan 20-26)
- Implement Priority 1.1 and 1.2 fixes
- Run validation experiments (Priority 1.3)
- Analyze results and compare to baselines

### Week 2 (Jan 27 - Feb 2)
- Implement Priority 2 enhancements
- Run complex SCM experiments
- Begin ablation studies

### Week 3-4 (Feb 3-16)
- Multi-objective reward implementation
- Computational optimizations
- Additional baseline comparisons

### Month 2 (Feb 17 - Mar 16)
- Priority 3 research enhancements
- Theoretical analysis
- Real-world domain experiments

### Month 3 (Mar 17 - Apr 16)
- Paper writing
- Final experiments
- Visualization & presentation materials

---

**Document Version:** 1.0  
**Last Updated:** January 20, 2026  
**Next Review:** After Priority 1 implementations complete
