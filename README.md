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

## Early stopping (episode-level)

To avoid wasted steps when progress stalls (often accompanied by rapid reward collapse), the training loop supports episode-level early stopping based on mechanism validation loss:

- `--patience_steps`: stop an episode after this many non-improving steps
- `--min_delta`: minimum improvement required to reset patience
- `--warmup_steps`: steps before early stopping can trigger

## Running

Example (custom policy, no pretrained LLM):

```bash
python ace_experiments.py --custom --episodes 100 --steps 25 --candidates 4 --output "experiment_results"
```

Example (pretrained LLM policy):

```bash
python ace_experiments.py --model "Qwen/Qwen2.5-1.5B" --episodes 100 --output "experiment_results"
```

## Guidance

Project guidance and design notes live in `guidance_documents/guidance_doc.txt`.
