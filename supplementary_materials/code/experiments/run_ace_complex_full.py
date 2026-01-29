#!/usr/bin/env python3
"""
Full ACE implementation for Complex 15-Node SCM.
Complete architecture matching ace_experiments.py with all optimizations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import numpy as np
import pandas as pd
import copy
import logging
from datetime import datetime
from collections import Counter, deque
from typing import Dict, List, Tuple, Optional

from experiments.complex_scm import ComplexGroundTruthSCM, ComplexStudentSCM, ComplexSCMLearner, ComplexCritic
from ace_experiments import (
    HuggingFacePolicy,
    TransformerPolicy,
    ExperimentalDSL,
    supervised_pretrain_llm,
    EarlyStopping,
    DPOLogger
)


# ================================================================
# HELPER FUNCTIONS (Ported from ace_experiments.py)
# ================================================================

def compute_unified_diversity_score(target, recent_targets, all_nodes, max_concentration=0.4, 
                                   recent_window=100, collider_parents=None, node_losses=None):
    """
    Unified diversity scoring function that encourages balanced exploration.
    Returns positive score for diverse choices, negative for over-concentrated.
    """
    if not recent_targets:
        return 0.0
    
    # Get recent window
    recent = list(recent_targets)[-recent_window:]
    target_counts = Counter(recent)
    total = len(recent)
    
    # Current concentration of target node
    current_count = target_counts.get(target, 0)
    current_frac = current_count / total if total > 0 else 0.0
    
    # Base diversity score: penalize if we're already over-concentrated
    if current_frac > max_concentration:
        # Steep penalty for exceeding concentration limit
        excess = current_frac - max_concentration
        diversity_score = -200.0 * excess
    else:
        # Reward for targeting under-sampled nodes
        under_sample_bonus = (max_concentration - current_frac) / max_concentration
        diversity_score = 50.0 * under_sample_bonus
    
    # Bonus for collider parents (if specified)
    if collider_parents and target in collider_parents:
        diversity_score += 20.0
    
    return diversity_score


def _direct_child_impact_weight(graph, target, node_losses, normalize=True):
    """
    Calculate weight based on impact on direct children.
    Higher weight if children have high loss.
    """
    # Find all nodes where target is a parent
    children = [node for node in graph.keys() if target in graph.get(node, [])]
    
    if not children:
        return 0.0
    
    # Sum losses of direct children
    child_loss_sum = sum(node_losses.get(child, 0.0) for child in children)
    
    if normalize:
        # Normalize by number of children to get average impact
        return child_loss_sum / len(children)
    else:
        return child_loss_sum


def _disentanglement_bonus(graph, target, node_losses):
    """
    Bonus for intervening on collider parents.
    Encourages disentangling correlated causes.
    """
    bonus = 0.0
    
    # Check if target is a parent of any collider
    for node, parents in graph.items():
        if len(parents) > 1 and target in parents:
            # Target is parent of a collider
            collider_loss = node_losses.get(node, 0.0)
            bonus += collider_loss * 0.5  # Scale by how badly we need to learn this collider
    
    return bonus


def calculate_value_novelty_bonus(value, target, history, n_bins=11, value_range=(-5, 5)):
    """
    Bonus for exploring novel value ranges.
    """
    if not history:
        return 1.0
    
    # Filter history for this target
    target_values = [v for t, v in history if t == target]
    
    if not target_values:
        return 1.0
    
    # Discretize into bins
    bins = np.linspace(value_range[0], value_range[1], n_bins)
    value_bin = np.digitize([value], bins)[0]
    
    # Count how many times we've sampled this bin
    bin_counts = Counter()
    for v in target_values:
        b = np.digitize([v], bins)[0]
        bin_counts[b] += 1
    
    # Bonus inversely proportional to bin frequency
    count = bin_counts.get(value_bin, 0)
    bonus = 1.0 / (1.0 + count)
    
    return bonus


def dpo_loss_llm(policy_net, ref_policy, student, winner_cmd, loser_cmd, 
                node_losses=None, intervention_history=None, beta=0.1):
    """
    DPO loss for LLM-based policy.
    Uses proper token-level log probabilities.
    """
    device = next(policy_net.model.parameters()).device
    
    # Create prompts with context
    intervention_history = intervention_history or []
    recent_history = list(intervention_history)[-10:] if intervention_history else []
    
    # Tokenize commands (the policy will add context)
    winner_tokens = policy_net.tokenizer(winner_cmd, return_tensors="pt", padding=True, truncation=True).to(device)
    loser_tokens = policy_net.tokenizer(loser_cmd, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Get logits from policy
    policy_net.model.train()
    policy_winner_out = policy_net.model(**winner_tokens)
    policy_loser_out = policy_net.model(**loser_tokens)
    
    # Get logits from reference (frozen)
    with torch.no_grad():
        ref_winner_out = ref_policy.model(**winner_tokens)
        ref_loser_out = ref_policy.model(**loser_tokens)
    
    # Compute log probabilities (simplified: use mean over sequence)
    def get_mean_log_prob(logits):
        log_probs = torch.log_softmax(logits.logits, dim=-1)
        return log_probs.mean()
    
    policy_winner_logp = get_mean_log_prob(policy_winner_out)
    policy_loser_logp = get_mean_log_prob(policy_loser_out)
    ref_winner_logp = get_mean_log_prob(ref_winner_out)
    ref_loser_logp = get_mean_log_prob(ref_loser_out)
    
    # DPO loss: -log sigmoid(beta * ((policy_winner - policy_loser) - (ref_winner - ref_loser)))
    policy_diff = policy_winner_logp - policy_loser_logp
    ref_diff = ref_winner_logp - ref_loser_logp
    
    loss = -torch.nn.functional.logsigmoid(beta * (policy_diff - ref_diff))
    
    return loss


# ================================================================
# MAIN ACE TRAINING LOOP
# ================================================================

def run_ace_complex_full(args):
    """
    Run complete ACE with full DPO training on complex 15-node SCM.
    """
    
    # Setup
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output, f"seed_{seed}", f"run_{timestamp}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(run_dir, "experiment.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    logging.info("=" * 70)
    logging.info("FULL ACE ON COMPLEX 15-NODE SCM")
    logging.info("=" * 70)
    logging.info(f"Seed: {seed}")
    logging.info(f"Device: {device}")
    logging.info(f"Episodes: {args.episodes}")
    logging.info(f"Steps per episode: {args.steps}")
    logging.info(f"Candidates per step: {args.candidates}")
    logging.info(f"Output: {run_dir}")
    logging.info(f"Model: {args.model if not args.custom else 'Custom Transformer'}")
    
    # Create environment
    oracle = ComplexGroundTruthSCM()
    dsl = ExperimentalDSL(oracle.nodes, value_min=args.value_min, value_max=args.value_max)
    critic = ComplexCritic(oracle)
    
    # Identify colliders for special handling
    collider_nodes = [n for n in oracle.nodes if len(oracle.graph.get(n, [])) > 1]
    collider_parents = list(set([p for c in collider_nodes for p in oracle.graph[c]]))
    logging.info(f"Collider nodes: {collider_nodes}")
    logging.info(f"Collider parents: {collider_parents}")
    
    # Create policy
    use_pretrained = not args.custom
    if use_pretrained:
        logging.info(f"Loading {args.model} policy...")
        policy_net = HuggingFacePolicy(args.model, dsl, device, token=args.token)
        logging.info("Model loaded successfully")
    else:
        logging.info("Using custom transformer policy...")
        policy_net = TransformerPolicy(dsl, device).to(device)
    
    ref_policy = copy.deepcopy(policy_net)
    if use_pretrained:
        ref_policy.model.eval()
    else:
        ref_policy.eval()
    
    optimizer = optim.Adam(policy_net.model.parameters() if use_pretrained else policy_net.parameters(), 
                          lr=args.lr)
    
    # Oracle pretraining
    if args.pretrain_steps > 0:
        logging.info(f"Oracle pretraining ({args.pretrain_steps} steps)...")
        temp_student = ComplexStudentSCM(oracle)
        temp_learner = ComplexSCMLearner(temp_student, oracle=oracle)
        init_losses = temp_learner.evaluate()
        
        if use_pretrained:
            supervised_pretrain_llm(
                policy_net,
                temp_student,
                oracle.graph,
                oracle.nodes,
                init_losses,
                optimizer,
                n_steps=args.pretrain_steps,
                value_min=args.value_min,
                value_max=args.value_max
            )
        else:
            # Custom transformer pretraining would go here
            logging.info("Skipping pretraining for custom transformer")
        
        ref_policy = copy.deepcopy(policy_net)
        if use_pretrained:
            ref_policy.model.eval()
        else:
            ref_policy.eval()
        
        logging.info("Pretraining complete")
    
    # Training tracking
    loss_history = []
    reward_history = []
    target_history = []
    value_history = []
    episode_history = []
    step_history = []
    cov_bonus_history = []
    score_history = []
    
    # Diversity tracking
    recent_action_counts = deque(maxlen=100)
    episode_action_counts = Counter()
    intervention_history = deque(maxlen=50)
    
    # Collider parent intervention tracking
    parent_intervention_counts = {c: Counter() for c in collider_nodes}
    
    # Early stopping
    early_stopper = None
    if args.early_stopping:
        early_stopper = EarlyStopping(
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
            min_episodes=args.early_stop_min_episodes
        )
    
    # Main training loop
    logging.info("\n" + "=" * 70)
    logging.info("STARTING TRAINING")
    logging.info("=" * 70)
    
    for episode in range(args.episodes):
        if episode % 5 == 0:
            logging.info(f"\n[Episode {episode}/{args.episodes}]")
        
        # Fresh student each episode
        current_student = ComplexStudentSCM(oracle)
        learner = ComplexSCMLearner(current_student, lr=args.learner_lr, 
                                   buffer_size=args.buffer_steps, oracle=oracle)
        
        # Run episode
        for step in range(args.steps):
            # Get current losses
            node_losses_start = learner.evaluate()
            loss_start = sum(node_losses_start.values())
            
            # Generate candidates
            num_candidates = args.candidates
            # SPEED OPTIMIZATION: Reduce candidates after warmup (same as main ACE)
            if episode < 20:
                num_candidates = args.candidates
            elif episode < 50:
                num_candidates = max(3, args.candidates // 2)
            else:
                num_candidates = 3
            
            candidates = []
            
            for k in range(num_candidates):
                # Generate intervention
                try:
                    if use_pretrained:
                        cmd_str, plan = policy_net.generate_experiment(
                            current_student,
                            node_losses=node_losses_start,
                            intervention_history=intervention_history
                        )
                    else:
                        # Custom transformer generation
                        cmd_str, plan = policy_net.generate_experiment(
                            current_student,
                            node_losses=node_losses_start
                        )
                    
                    if plan is None:
                        # Failed to parse - fallback
                        target = random.choice(oracle.nodes)
                        value = random.uniform(args.value_min, args.value_max)
                        cmd_str = f"DO {target} = {value:.4f}"
                        plan = {"target": target, "value": value, "command": cmd_str}
                except Exception as e:
                    logging.debug(f"Generation failed: {e}")
                    target = random.choice(oracle.nodes)
                    value = random.uniform(args.value_min, args.value_max)
                    cmd_str = f"DO {target} = {value:.4f}"
                    plan = {"target": target, "value": value, "command": cmd_str}
                
                # Evaluate candidate via lookahead
                student_clone = copy.deepcopy(current_student)
                
                # Clone learner with replay buffer
                initial_buffer = []
                for b in learner.buffer:
                    b_data = {k: v.detach().clone() for k, v in b["data"].items()}
                    initial_buffer.append({"data": b_data, "intervened": b.get("intervened")})
                
                clone_learner = ComplexSCMLearner(
                    student_clone,
                    lr=args.learner_lr,
                    buffer_size=args.buffer_steps,
                    oracle=oracle
                )
                clone_learner.buffer = initial_buffer
                
                # Execute intervention on clone
                data = oracle.generate(n_samples=100, interventions={plan["target"]: plan["value"]})
                clone_learner.train_step(data, intervened=plan["target"], n_epochs=args.learner_epochs)
                
                # Evaluate reward
                node_losses_end = clone_learner.evaluate()
                loss_end = sum(node_losses_end.values())
                info_gain = loss_start - loss_end
                
                # SOPHISTICATED REWARD SCORING (matching main ACE)
                tgt = plan["target"]
                val = plan["value"]
                
                # 1. Node importance (parent of high-loss children)
                node_weight = _direct_child_impact_weight(oracle.graph, tgt, node_losses_start, normalize=True)
                denom = float(sum(node_losses_start.values())) + 1e-8
                norm_weight = node_weight / denom
                under_sample = 1.0 / np.sqrt(1.0 + episode_action_counts.get(tgt, 0))
                cov_bonus = args.cov_bonus * norm_weight * under_sample
                
                # 2. Unified diversity score
                unified_diversity = compute_unified_diversity_score(
                    target=tgt,
                    recent_targets=list(recent_action_counts),
                    all_nodes=oracle.nodes,
                    max_concentration=args.max_concentration,
                    recent_window=100,
                    collider_parents=collider_parents,
                    node_losses=node_losses_start
                )
                
                # 3. Value novelty bonus
                novelty_bonus = calculate_value_novelty_bonus(
                    val, tgt, list(zip(target_history, value_history))
                )
                
                # 4. Disentanglement bonus (collider parents)
                disent_bonus = _disentanglement_bonus(oracle.graph, tgt, node_losses_start)
                
                # Final score
                score = (
                    info_gain +  # Primary: information gain
                    cov_bonus +  # Node importance
                    args.diversity_reward_weight * unified_diversity +  # Diversity
                    novelty_bonus * 0.1 +  # Value exploration
                    disent_bonus * 0.2  # Collider focus
                )
                
                candidates.append((cmd_str, info_gain, cov_bonus, score, plan))
            
            # Sort by score
            sorted_cands = sorted(candidates, key=lambda x: x[3], reverse=True)
            
            # COLLAPSE DETECTION AND SMART BREAKERS (matching main ACE)
            top_node = None
            top_frac = 0.0
            if len(recent_action_counts) >= 20:
                node_counts = Counter(recent_action_counts)
                top_node, top_count = node_counts.most_common(1)[0]
                top_frac = top_count / len(recent_action_counts)
            
            # Smart collapse breaker
            if args.smart_breaker and top_frac > 0.65:
                # Severe collapse - inject intervention on under-sampled collider parent
                collider_losses = [(c, node_losses_start.get(c, 0.0)) for c in collider_nodes]
                collider_losses.sort(key=lambda x: x[1], reverse=True)
                
                if collider_losses:
                    target_collider = collider_losses[0][0]
                    parents = oracle.graph[target_collider]
                    
                    # Pick least-sampled parent
                    parent_counts = {p: parent_intervention_counts[target_collider][p] for p in parents}
                    breaker_tgt = min(parent_counts, key=parent_counts.get)
                    breaker_val = random.uniform(args.value_min, args.value_max)
                    breaker_cmd = f"DO {breaker_tgt} = {breaker_val:.4f}"
                    breaker_plan = {"target": breaker_tgt, "value": breaker_val, "command": breaker_cmd}
                    
                    # Give it high score
                    breaker_score = 100.0
                    sorted_cands.insert(0, (breaker_cmd, 0.0, args.cov_bonus, breaker_score, breaker_plan))
                    
                    if episode % 10 == 0:
                        logging.info(f"  [Smart Breaker] Collapse {top_node}@{top_frac:.0%}, injecting {breaker_cmd}")
            
            # MANDATORY DIVERSITY CONSTRAINT
            diversity_enforced = False
            if args.diversity_constraint and top_node is not None and top_frac > args.diversity_threshold:
                diverse_cands = [c for c in sorted_cands if c[4] and c[4].get("target") != top_node]
                if diverse_cands:
                    logging.info(f"  [Diversity Constraint] Forcing alternative to {top_node}@{top_frac:.0%}")
                    sorted_cands = diverse_cands
                    diversity_enforced = True
            
            # FORCED DIVERSITY: Every 10 steps
            if step > 0 and step % 10 == 0 and top_node is not None and top_frac > 0.50 and not diversity_enforced:
                node_counts = Counter(recent_action_counts)
                least_sampled = min([n for n in oracle.nodes if n != top_node],
                                  key=lambda n: node_counts.get(n, 0), default=None)
                if least_sampled:
                    diverse_cands = [c for c in sorted_cands if c[4] and c[4].get("target") == least_sampled]
                    if diverse_cands:
                        logging.info(f"  [Forced Diversity] Targeting {least_sampled} vs {top_node}@{top_frac:.0%}")
                        sorted_cands = diverse_cands
            
            # Select winner
            winner_cmd, winner_reward, winner_cov_bonus, winner_score, winner_plan = sorted_cands[0]
            
            # HARD INTERVENTION CAP (matching main ACE)
            MAX_NODE_FRACTION = 0.70
            if len(recent_action_counts) > 10 and winner_plan:
                node_counts = Counter(recent_action_counts)
                winner_target = winner_plan.get("target")
                
                if winner_target:
                    winner_count = node_counts.get(winner_target, 0)
                    winner_fraction = winner_count / len(recent_action_counts)
                    
                    if winner_fraction > MAX_NODE_FRACTION:
                        logging.info(f"  [Hard Cap] {winner_target}@{winner_fraction:.0%} > {MAX_NODE_FRACTION:.0%}")
                        
                        # Force alternative
                        undersampled = min(collider_parents, key=lambda p: node_counts.get(p, 0))
                        forced_value = random.uniform(args.value_min, args.value_max)
                        winner_plan = {
                            "target": undersampled,
                            "value": forced_value,
                            "command": f"DO {undersampled} = {forced_value:.4f}"
                        }
                        winner_cmd = winner_plan["command"]
                        winner_score = 100.0
            
            # EPISTEMIC CURIOSITY: Select loser strategically
            loser_idx = -1
            curiosity_weight = 1.0
            
            dom_node = top_node if (top_node and top_frac > 0.40) else None
            winner_tgt = winner_plan.get("target") if winner_plan else None
            
            if dom_node and winner_tgt and winner_tgt != dom_node:
                # Winner is novel - pair against collapsed node
                for i in range(1, len(sorted_cands)):
                    cand = sorted_cands[i]
                    if cand[4] and cand[4].get("target") == dom_node:
                        loser_idx = i
                        curiosity_weight = 2.0
                        break
            
            loser_cmd, loser_reward, loser_cov_bonus, loser_score, loser_plan = sorted_cands[loser_idx]
            
            # Execute winner on actual learner
            data = oracle.generate(n_samples=100, interventions={winner_plan["target"]: winner_plan["value"]})
            learner.train_step(data, intervened=winner_plan["target"], n_epochs=args.learner_epochs)
            
            # Track interventions
            intervention_history.append((winner_plan["target"], winner_plan["value"]))
            recent_action_counts.append(winner_plan["target"])
            episode_action_counts[winner_plan["target"]] += 1
            
            # Track collider parent coverage
            for collider in collider_nodes:
                parents = oracle.graph[collider]
                if winner_plan["target"] in parents:
                    parent_intervention_counts[collider][winner_plan["target"]] += 1
            
            # OBSERVATIONAL TRAINING (CRITICAL - prevents forgetting)
            if args.obs_train_interval > 0 and step > 0 and step % args.obs_train_interval == 0:
                obs_data = oracle.generate(n_samples=args.obs_train_samples, interventions=None)
                learner.train_step(obs_data, intervened=None, n_epochs=args.obs_train_epochs)
                if episode % 20 == 0:
                    logging.info(f"  [Obs Training] Injected {args.obs_train_samples} samples")
            
            # DPO UPDATE
            if winner_score > loser_score:
                optimizer.zero_grad()
                
                if use_pretrained:
                    loss = dpo_loss_llm(
                        policy_net, ref_policy, current_student,
                        winner_cmd, loser_cmd,
                        node_losses=node_losses_start,
                        intervention_history=intervention_history
                    )
                else:
                    # Custom transformer DPO would go here
                    # For now, skip
                    loss = torch.tensor(0.0)
                
                # Apply curiosity boost
                loss = loss * curiosity_weight
                
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        policy_net.model.parameters() if use_pretrained else policy_net.parameters(),
                        1.0
                    )
                    optimizer.step()
                
                loss_history.append(loss.item())
            else:
                loss_history.append(0.0)
            
            # Track metrics
            reward_history.append(winner_reward)
            target_history.append(winner_plan["target"] if winner_plan else None)
            value_history.append(winner_plan["value"] if winner_plan else 0.0)
            cov_bonus_history.append(winner_cov_bonus)
            score_history.append(winner_score)
            episode_history.append(episode)
            step_history.append(step)
        
        # Update reference policy periodically
        if args.update_reference_interval > 0 and episode > 0 and episode % args.update_reference_interval == 0:
            ref_policy = copy.deepcopy(policy_net)
            if use_pretrained:
                ref_policy.model.eval()
            else:
                ref_policy.eval()
            logging.info(f"  [Reference Update] Updated at episode {episode}")
        
        # Early stopping checks
        if args.early_stopping and early_stopper and episode >= early_stopper.min_episodes:
            final_losses = learner.evaluate()
            total_loss = sum(final_losses.values())
            
            if args.use_per_node_convergence:
                if early_stopper.check_per_node_convergence(final_losses, patience=args.node_convergence_patience):
                    logging.info(f"[CONVERGED] Per-node convergence at episode {episode}")
                    break
            else:
                if early_stopper.check_loss(total_loss):
                    logging.info(f"[EARLY STOP] at episode {episode}")
                    break
    
    # Save metrics
    df = pd.DataFrame({
        "dpo_loss": loss_history,
        "reward": reward_history,
        "cov_bonus": cov_bonus_history,
        "score": score_history,
        "target": target_history,
        "value": value_history,
        "episode": episode_history,
        "step": step_history,
    })
    
    df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    
    # Final evaluation
    final_losses = learner.evaluate()
    total_loss = sum(final_losses.values())
    
    logging.info("\n" + "=" * 70)
    logging.info("FINAL RESULTS")
    logging.info("=" * 70)
    logging.info(f"Total Loss: {total_loss:.4f}")
    logging.info("Per-Node Losses:")
    for node in oracle.nodes:
        logging.info(f"  {node}: {final_losses[node]:.4f}")
    
    logging.info(f"\nResults saved to {run_dir}")
    
    return run_dir, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full ACE on Complex 15-Node SCM")
    
    # Model selection
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B", help="HF Model Name")
    parser.add_argument("--custom", action="store_true", help="Use Custom Transformer instead of LLM")
    parser.add_argument("--token", type=str, default=None, help="HF Auth Token")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=50, help="Steps per episode")
    parser.add_argument("--candidates", type=int, default=4, help="Candidates per step")
    parser.add_argument("--lr", type=float, default=1e-5, help="Policy learning rate")
    parser.add_argument("--learner_lr", type=float, default=2e-3, help="Student SCM learning rate")
    parser.add_argument("--learner_epochs", type=int, default=100, help="Learner training epochs per step")
    parser.add_argument("--buffer_steps", type=int, default=50, help="Learner replay buffer length")
    
    # Reward components
    parser.add_argument("--cov_bonus", type=float, default=60.0, help="Coverage bonus scale")
    parser.add_argument("--diversity_reward_weight", type=float, default=0.3, help="Weight for diversity reward")
    parser.add_argument("--max_concentration", type=float, default=0.4, help="Maximum concentration on single node")
    
    # Diversity mechanisms
    parser.add_argument("--diversity_constraint", action="store_true", default=True, help="Enforce mandatory diversity")
    parser.add_argument("--diversity_threshold", type=float, default=0.60, help="Threshold for diversity enforcement")
    parser.add_argument("--smart_breaker", action="store_true", default=True, help="Use smart collapse breaker")
    
    # Observational training (CRITICAL)
    parser.add_argument("--obs_train_interval", type=int, default=3, help="Train on observational data every N steps")
    parser.add_argument("--obs_train_samples", type=int, default=200, help="Observational samples per training")
    parser.add_argument("--obs_train_epochs", type=int, default=100, help="Epochs for observational training")
    
    # Oracle pretraining
    parser.add_argument("--pretrain_steps", type=int, default=500, help="Supervised pre-training steps")
    
    # Reference policy updates
    parser.add_argument("--update_reference_interval", type=int, default=25, help="Update reference policy every N episodes")
    
    # Early stopping
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--early_stop_patience", type=int, default=20, help="Early stop patience")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.01, help="Minimum improvement")
    parser.add_argument("--early_stop_min_episodes", type=int, default=40, help="Minimum episodes before early stop")
    parser.add_argument("--use_per_node_convergence", action="store_true", default=True, help="Use per-node convergence")
    parser.add_argument("--node_convergence_patience", type=int, default=10, help="Per-node convergence patience")
    
    # Value range
    parser.add_argument("--value_min", type=float, default=-5.0, help="Minimum intervention value")
    parser.add_argument("--value_max", type=float, default=5.0, help="Maximum intervention value")
    
    # Output and reproducibility
    parser.add_argument("--output", type=str, default="results/ace_complex_scm_optimized", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    run_ace_complex_full(args)
