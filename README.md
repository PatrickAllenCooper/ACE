# ACE: Adversarial Causal Experimentalist

**A Framework for Automated Causal Discovery via DSL-Mediated Self-Play and Direct Preference Optimization (DPO).**

## Overview

**ACE** (Adversarial Self-Alignment for Causal Experimentalism) is a research framework that reformulates causal discovery not as a static curve-fitting problem, but as an interactive, sequential decision-making game.

The goal is to train an AI agent (the **Experimentalist**) to autonomously design experiments that efficiently uncover the hidden mechanisms of a ground-truth system (the **Environment**). By treating the "Learner's" ignorance as an adversary, the agent uses **Direct Preference Optimization (DPO)** to align itself with the objective of maximizing "Scientific Surprise" (Information Gain).

## Key Features

* **Interactive Causal Discovery:** The agent actively queries the environment using interventions ($do(X=x)$), rather than learning from passive observational data.
* **Episodic Discovery Protocol:** Training is organized into "Episodes" where a fresh Learner attempts to solve the system from scratch, forcing the Agent to learn generalizable strategies rather than memorizing a single solution.
* **Dual-Policy Architecture:** Supports both custom scratch-trained Transformers and pretrained LLM Adapters (e.g., **Qwen-2.5**) to guide experimentation.
* **Rigorous DSL:** All interventions are grounded in a Domain Specific Language (DSL) to ensure valid, physically realizable experiments (e.g., `DO X1 = 2.5`).
* **Teacher Injection:** A bootstrapping mechanism that injects valid "Teacher" commands during early training to overcome the cold-start problem and prevent reward hacking.

## System Architecture

The framework consists of three primary interacting components:

1.  **The Environment ($M^*$):** A ground-truth Structural Causal Model (SCM) that generates data. It supports complex non-linear mechanisms (e.g., Sine waves, quadratic functions) to rigorously test discovery capabilities.
2.  **The Learner ($M_\theta$):** A learnable SCM parameterized by Neural Networks (MLPs). It attempts to approximate the Environment's mechanisms based on data gathered by the Agent.
3.  **The Experimentalist ($\pi_\phi$):** The policy (Agent) that observes the Learner's current state (weights, uncertainty) and proposes the next experiment to run.

## Important training detail: do-interventions vs mechanism fitting

When an experiment uses a hard intervention `DO Xi = c`, the structural equation for `Xi` is replaced by an assignment for those samples. As a result, **the learner must not train `Xi`'s mechanism** on batches where `Xi` was directly intervened on, or it will push the mechanism toward a constant / parent-ignoring mapping and fail to learn functional associations.

Implementation notes:

- `ExperimentExecutor.run_experiment(...)` returns a batch dict of the form:
  - `{"data": <dict[node] -> Tensor>, "intervened": <node name or None>}`
- `SCMLearner.train_step(...)` masks out samples for the **intervened node only** while still using the batch to train downstream mechanisms (descendants).

## Collider Learning Challenge

**Colliders** (nodes with multiple parents, e.g., X3 ← X1, X3 ← X2) are particularly difficult for the agent to learn because:

1. **Training-Validation Mismatch**: The learner is validated on *independent* parent samples, but training data often has correlated parents (e.g., X2 = f(X1))
2. **Greedy Exploitation**: Intervening on upstream nodes (X1) immediately improves loss by helping their direct children (X2), leading to over-sampling
3. **Missed Critical Experiments**: Learning X3 = f(X1, X2) requires seeing X1 and X2 vary *independently*, which only happens when intervening directly on X2

### Solution: Reward Rescaling & Enhanced Incentives

To address the collider learning problem, the framework implements:

1. **Scaled Rewards** (v2025-01-02): Raw rewards reduced from `delta * 100` to `delta * 10` to make exploration bonuses competitive
2. **Disentanglement Bonus**: Strong incentive (100x child loss) for interventions that break parent correlations in collider structures
3. **Parent Balance Bonus**: Encourages balanced coverage of all parents of multi-parent nodes
4. **Comprehensive Diagnostics**: Per-node loss tracking (`node_losses.csv`) and intervention coverage analysis (`intervention_coverage.csv`) for debugging

This approach ensures the agent explores interventions on *all* parents of colliders, enabling proper mechanism learning.

## Early stopping (episode-level)

To avoid wasted steps when progress stalls (often accompanied by rapid reward collapse), the training loop supports episode-level early stopping based on mechanism validation loss:

- `--patience_steps`: stop an episode after this many non-improving steps
- `--min_delta`: minimum improvement required to reset patience
- `--warmup_steps`: steps before early stopping can trigger

## Anti-Collapse Mechanisms

The agent can fall into "collapse" where it repeatedly targets the same node (e.g., X1@96%), preventing discovery of complex structures like colliders. The framework includes multiple anti-collapse mechanisms:

### 1. Collapse Detection & Penalties
- `--collapse_threshold` (default: 0.30): Fraction threshold for detecting collapse (lowered from 0.50)
- `--collapse_penalty` (default: 150.0): Quadratic penalty applied when collapse detected (scales with severity) **[Updated 2025-01-02]**

### 2. Under-Sampling Incentives
- `--undersampled_bonus` (default: 100.0): Strong bonus for severely neglected nodes (e.g., X2 when X1 dominates) **[Updated 2025-01-02]**
- `--cov_bonus` (default: 60.0): Coverage bonus scale increased for stronger exploration **[Updated 2025-01-02]**
- `--parent_balance_bonus` (default: 80.0): Bonus for balanced interventions among parents of multi-parent nodes **[New 2025-01-02]**

### 3. Mandatory Diversity Constraint
- `--diversity_constraint`: Enable hard constraint that rejects over-sampled nodes when collapse > threshold
- `--diversity_threshold` (default: 0.60): Threshold for mandatory diversity enforcement

### 4. Forced Periodic Exploration
- Every 10 steps, if collapse > 50%, the system automatically targets the least-sampled node
- Helps discover collider structures (e.g., X3 with parents X1, X2) by ensuring balanced parent coverage

### 5. Diagnostic Outputs (New 2025-01-02)
The framework now generates detailed diagnostic files for analyzing learning failures:
- `node_losses.csv`: Per-node mechanism losses at each step
- `intervention_coverage.csv`: Intervention balance for collider parent nodes
- Detailed bonus component breakdown in logs (every 50 steps)

### 6. Collapse Breaker (Fail-Safe)
- **Problem:** Even with penalties, a collapsed policy might *only* propose the collapsed node, forcing the selector to pick it.
- **Solution:** If collapse > 50% and *all* candidates target the collapsed node, the system forcefully **injects a random alternative**.
- **Result:** The alternative (neutral score) automatically wins against the penalized candidates (negative score), mechanically breaking the loop.

### 7. Principled Normalization (New 2026-01-03)
- **Problem:** Base nodes (e.g., X1) structurally affect more children than downstream nodes (e.g., X3), leading to artificially higher "Impact Scores" even when mechanisms are equally broken.
- **Solution:** Impact scoring now uses **Average Child Loss** (normalized by child count) instead of Total Child Loss. This levels the playing field, ensuring agents choose based on *mechanism urgency* rather than graph position.

## Running

Example (custom policy, no pretrained LLM):

```bash
python ace_experiments.py --custom --episodes 100 --steps 25 --candidates 4 --output "experiment_results"
```

Example (pretrained LLM policy):

```bash
python ace_experiments.py --model "Qwen/Qwen2.5-1.5B" --episodes 100 --output "experiment_results"
```

Example (with diversity constraint to prevent collapse):

```bash
python ace_experiments.py --model "Qwen/Qwen2.5-1.5B" --episodes 100 --diversity_constraint --output "experiment_results"
```

### Debugging Parsing Issues

If you encounter low parsing rates (commands not being recognized), use the `--debug_parsing` flag for detailed diagnostics:

```bash
python ace_experiments.py --model "Qwen/Qwen2.5-1.5B" --episodes 100 --debug_parsing --output "experiment_results"
```

This enables:
- Detailed parse attempt logging showing what the model generates vs what's expected
- Sample failed parse outputs every 10 episodes
- Real-time parsing statistics every 20 steps

The DSL parser now supports:
- Case-insensitive matching (`DO`, `do`, `Do` all work)
- Scientific notation (e.g., `1e-5`, `2.5E+3`)
- Various decimal formats (e.g., `.5`, `0.5`, `5.`, `5.0`)
- Automatic value clipping to valid ranges

## Guidance

Project guidance and design notes live in `guidance_documents/guidance_doc.txt`.
