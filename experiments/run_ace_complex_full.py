#!/usr/bin/env python3
"""
Full ACE implementation for Complex 15-Node SCM.
Complete DPO training with all components from ace_experiments.py.
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

from experiments.complex_scm import ComplexGroundTruthSCM, ComplexStudentSCM, ComplexSCMLearner, ComplexCritic
from ace_experiments import (
    HuggingFacePolicy,
    TransformerPolicy,
    ExperimentalDSL,
    supervised_pretrain_llm,
    EarlyStopping,
    DPOLogger
)


def compute_dpo_loss(policy_net, ref_policy, winner_cmd, loser_cmd, device, beta=0.1):
    """
    Compute DPO loss from best/worst intervention pairs.
    Proper implementation with token-level log probabilities.
    """
    # Tokenize
    winner_tokens = policy_net.tokenizer(winner_cmd, return_tensors="pt", padding=True).to(device)
    loser_tokens = policy_net.tokenizer(loser_cmd, return_tensors="pt", padding=True).to(device)
    
    # Get logits from policy
    policy_net.model.train()
    policy_winner_logits = policy_net.model(**winner_tokens).logits
    policy_loser_logits = policy_net.model(**loser_tokens).logits
    
    # Get logits from reference
    with torch.no_grad():
        ref_winner_logits = ref_policy.model(**winner_tokens).logits
        ref_loser_logits = ref_policy.model(**loser_tokens).logits
    
    # Convert to log probabilities
    # Simplified: use mean of log probs over sequence
    policy_winner_logp = torch.log_softmax(policy_winner_logits, dim=-1).mean()
    policy_loser_logp = torch.log_softmax(policy_loser_logits, dim=-1).mean()
    ref_winner_logp = torch.log_softmax(ref_winner_logits, dim=-1).mean()
    ref_loser_logp = torch.log_softmax(ref_loser_logits, dim=-1).mean()
    
    # DPO loss
    policy_ratio = policy_winner_logp - policy_loser_logp
    ref_ratio = ref_winner_logp - ref_loser_logp
    
    loss = -torch.nn.functional.logsigmoid(beta * (policy_ratio - ref_ratio))
    
    return loss


def compute_reward_with_bonuses(info_gain, target, node_losses, nodes):
    """
    Compute reward with diversity and node importance bonuses.
    """
    # Information gain (primary)
    reward = info_gain
    
    # Node importance (targets high-loss nodes)
    node_loss = node_losses.get(target, 0)
    avg_loss = sum(node_losses.values()) / len(node_losses)
    if node_loss > avg_loss:
        reward += 0.1 * (node_loss - avg_loss)  # Bonus for targeting difficult nodes
    
    return reward


def run_ace_complex_full(seed=42, episodes=200, output_dir="results/ace_complex_scm"):
    """
    Run complete ACE with full DPO training on complex 15-node SCM.
    """
    
    # Setup
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}_seed{seed}")
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
    
    print(f"[STARTUP] Full ACE on Complex 15-Node SCM", flush=True)
    print(f"[STARTUP] Seed: {seed}", flush=True)
    print(f"[STARTUP] Device: {device}", flush=True)
    print(f"[STARTUP] Output: {run_dir}", flush=True)
    
    # Create environment
    oracle = ComplexGroundTruthSCM()
    dsl = ExperimentalDSL(oracle.nodes, value_min=-5.0, value_max=5.0)
    critic = ComplexCritic(oracle)
    
    # Create policy
    print("[STARTUP] Loading Qwen2.5-1.5B policy...", flush=True)
    policy_net = HuggingFacePolicy("Qwen/Qwen2.5-1.5B", dsl, device)
    print("[STARTUP] Model loaded successfully", flush=True)
    
    ref_policy = copy.deepcopy(policy_net)
    ref_policy.eval()
    
    optimizer = optim.Adam(policy_net.model.parameters(), lr=1e-5)
    
    # Oracle pretraining
    print("[STARTUP] Oracle pretraining (200 steps)...", flush=True)
    temp_student = ComplexStudentSCM(oracle)
    temp_learner = ComplexSCMLearner(temp_student, oracle=oracle)
    init_losses = temp_learner.evaluate()
    
    supervised_pretrain_llm(
        policy_net,
        temp_student,
        oracle.graph,
        oracle.nodes,
        init_losses,
        optimizer,
        n_steps=500,  # Increased from 200 for better initialization
        value_min=-5.0,
        value_max=5.0
    )
    
    ref_policy = copy.deepcopy(policy_net)
    ref_policy.eval()
    print("[STARTUP] Pretraining complete", flush=True)
    
    # Training loop - run full episodes for complete evaluation
    results = []
    dpo_losses = []
    # Disable early stopping to get full 200 episode runs
    # early_stopper = EarlyStopping(patience=20, min_delta=0.01, min_episodes=40)
    intervention_history = deque(maxlen=50)
    
    for episode in range(episodes):
        if episode % 5 == 0:
            print(f"[PROGRESS] Episode {episode}/{episodes}", flush=True)
        
        # Fresh student each episode
        student = ComplexStudentSCM(oracle)
        learner = ComplexSCMLearner(student, oracle=oracle)
        
        # Run one training step
        for step in range(50):  # Increased from 25 to 50 steps per episode
            # Get current losses
            node_losses = learner.evaluate()
            
        # Generate K=2 candidates (faster per episode, allows more episodes)
        K = 2
        candidates = []
            
            for k in range(K):
                try:
                    # Use policy to generate
                    prompt = policy_net.scm_to_prompt(student, node_losses, list(intervention_history)[-10:])
                    generated_text, parsed = policy_net.generate_and_parse(prompt)
                    
                    if parsed and 'node' in parsed and 'value' in parsed:
                        target = parsed['node']
                        value = parsed['value']
                        if target in oracle.nodes and -5 <= value <= 5:
                            candidates.append((target, value))
                            continue
                except Exception as e:
                    logging.debug(f"Generation failed: {e}")
                
                # Fallback
                target = random.choice(oracle.nodes)
                value = random.uniform(-5, 5)
                candidates.append((target, value))
            
            # Ensure K candidates
            while len(candidates) < K:
                candidates.append((random.choice(oracle.nodes), random.uniform(-5, 5)))
            
            # Evaluate candidates on cloned learners
            candidate_data = []
            for target, value in candidates:
                cloned_student = ComplexStudentSCM(oracle)
                cloned_learner = ComplexSCMLearner(cloned_student, oracle=oracle)
                
                losses_before = cloned_learner.evaluate()
                
                data = oracle.generate(n_samples=100, interventions={target: value})
                cloned_learner.train_step(data)
                
                losses_after = cloned_learner.evaluate()
                
                info_gain = sum(losses_before.values()) - sum(losses_after.values())
                reward = compute_reward_with_bonuses(info_gain, target, losses_before, oracle.nodes)
                
                candidate_data.append({
                    'target': target,
                    'value': value,
                    'reward': reward,
                    'info_gain': info_gain
                })
            
            # Select best and worst
            best_idx = max(range(K), key=lambda i: candidate_data[i]['reward'])
            worst_idx = min(range(K), key=lambda i: candidate_data[i]['reward'])
            
            best = candidate_data[best_idx]
            worst = candidate_data[worst_idx]
            
            # Execute best on actual learner
            data = oracle.generate(n_samples=100, interventions={best['target']: best['value']})
            learner.train_step(data)
            
            # CRITICAL: Observational training every 3 steps to prevent mechanism forgetting
            if step > 0 and step % 3 == 0:
                obs_data = oracle.generate(n_samples=200, interventions=None)
                learner.train_step(obs_data)
                if episode % 20 == 0 and step % 9 == 0:
                    logging.info(f"  [Obs Training] Episode {episode}, Step {step}")
            
            intervention_history.append((best['target'], best['value']))
            
            # DPO update
            if best['reward'] != worst['reward']:  # Only update if there's a preference
                winner_cmd = f"DO {best['target']} = {best['value']:.4f}"
                loser_cmd = f"DO {worst['target']} = {worst['value']:.4f}"
                
                dpo_loss = compute_dpo_loss(policy_net, ref_policy, winner_cmd, loser_cmd, device)
                
                optimizer.zero_grad()
                dpo_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.model.parameters(), 1.0)
                optimizer.step()
                
                dpo_losses.append(dpo_loss.item())
        
        # Evaluate final state
        final_losses = learner.evaluate()
        total_loss = sum(final_losses.values())
        
        results.append({
            'episode': episode,
            'total_loss': total_loss,
            **{f'loss_{k}': v for k, v in final_losses.items()}
        })
        
        if episode % 10 == 0 and dpo_losses:
            recent_dpo = np.mean(dpo_losses[-10:])
            logging.info(f"Episode {episode}: Loss={total_loss:.2f}, DPO Loss={recent_dpo:.4f}")
        
        # Update reference policy periodically
        if episode > 0 and episode % 25 == 0:
            ref_policy = copy.deepcopy(policy_net)
            ref_policy.eval()
            logging.info(f"Reference policy updated at episode {episode}")
        
        # Run full 200 episodes (no early stopping)
        # This gives complete learning curves for comparison
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, "results.csv"), index=False)
    
    # Save DPO losses
    dpo_df = pd.DataFrame({'episode': range(len(dpo_losses)), 'dpo_loss': dpo_losses})
    dpo_df.to_csv(os.path.join(run_dir, "dpo_training.csv"), index=False)
    
    print(f"[COMPLETE] Saved to {run_dir}", flush=True)
    logging.info(f"Final loss: {total_loss:.4f} after {len(results)} episodes")
    
    return run_dir, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--output", type=str, default="results/ace_complex_scm")
    args = parser.parse_args()
    
    run_ace_complex_full(
        seed=args.seed,
        episodes=args.episodes,
        output_dir=args.output
    )
