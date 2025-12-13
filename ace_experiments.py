import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import copy
import random
import argparse
import logging
import pandas as pd
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter, deque

# ----------------------------------------------------------------
# 1. CORE SCM CLASSES
# ----------------------------------------------------------------
class CausalModel:
    def __init__(self, edges):
        self.graph = nx.DiGraph(edges)
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Graph must be a DAG")
        self.nodes = sorted(list(self.graph.nodes))
        self.topo_order = list(nx.topological_sort(self.graph))
        
    def get_parents(self, node):
        return list(self.graph.predecessors(node))

class GroundTruthSCM(CausalModel):
    def __init__(self):
        edges = [('X1', 'X2'), ('X2', 'X3'), ('X1', 'X3'), ('X4', 'X5')]
        super().__init__(edges)
        
    def mechanisms(self, data, node, n_samples=1):
        n = next(iter(data.values())).shape[0] if data else n_samples
        noise = torch.randn(n) * 0.1
        
        if node == 'X1': return torch.randn(n)
        if node == 'X4': return torch.randn(n) + 2.0
        if node == 'X2': return 2.0 * data['X1'] + 1.0 + noise
        if node == 'X3': return 0.5 * data['X1'] - data['X2'] + torch.sin(data['X2']) + noise
        if node == 'X5': return 0.2 * (data['X4'] ** 2) + noise
        return noise

    def generate(self, n_samples=1, interventions=None):
        data = {}
        interventions = interventions or {}
        for node in self.topo_order:
            if node in interventions:
                val = interventions[node]
                data[node] = torch.full((n_samples,), float(val))
            else:
                parents = self.get_parents(node)
                p_data = {p: data[p] for p in parents}
                data[node] = self.mechanisms(p_data, node, n_samples=n_samples)
        return data

class StudentSCM(CausalModel, nn.Module):
    def __init__(self, gt_instance):
        super().__init__(list(gt_instance.graph.edges))
        nn.Module.__init__(self)
        self.mechanisms = nn.ModuleDict()
        
        for node in self.nodes:
            parents = self.get_parents(node)
            if parents:
                self.mechanisms[node] = nn.Sequential(
                    nn.Linear(len(parents), 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
            else:
                self.mechanisms[node] = nn.ParameterDict({
                    'mu': nn.Parameter(torch.zeros(1)),
                    'sigma': nn.Parameter(torch.ones(1))
                })

    def forward(self, n_samples=1, interventions=None):
        data = {}
        interventions = interventions or {}
        for node in self.topo_order:
            if node in interventions:
                data[node] = torch.full((n_samples,), float(interventions[node]))
            else:
                parents = self.get_parents(node)
                if not parents:
                    data[node] = self.mechanisms[node]['mu'].expand(n_samples)
                else:
                    p_tensor = torch.stack([data[p] for p in parents], dim=1)
                    data[node] = self.mechanisms[node](p_tensor).squeeze()
        return data

# ----------------------------------------------------------------
# 2. EXPERIMENTAL ENGINE
# ----------------------------------------------------------------
class ExperimentExecutor:
    def __init__(self, ground_truth_scm):
        self.env = ground_truth_scm
        
    def run_experiment(self, intervention_plan):
        if intervention_plan is None:
            return self.env.generate(n_samples=100)
        
        target = intervention_plan.get('target')
        value = intervention_plan.get('value')
        n_samples = intervention_plan.get('samples', 100)
        
        if target:
            return self.env.generate(n_samples, interventions={target: value})
        else:
            return self.env.generate(n_samples)

class SCMLearner:
    def __init__(self, student_scm, lr=0.01, buffer_steps=50, initial_buffer=None):
        self.student = student_scm
        self.optimizer = optim.Adam(self.student.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = list(initial_buffer) if initial_buffer is not None else []
        self.buffer_steps = buffer_steps
        
    def train_step(self, data, n_epochs=50):
        self.student.train()
        
        # Update buffer
        self.buffer.append(data)
        if len(self.buffer) > self.buffer_steps:
            self.buffer.pop(0)
            
        # Collate data
        combined_data = {}
        # Assumes all data dicts have the same nodes (which they should)
        nodes = list(data.keys())
        for node in nodes:
            tensors = [d[node] for d in self.buffer]
            combined_data[node] = torch.cat(tensors, dim=0)
            
        losses = []
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            total_loss = 0
            for node in self.student.nodes:
                parents = self.student.get_parents(node)
                y_true = combined_data[node]
                if not parents:
                    y_pred = self.student.mechanisms[node]['mu'].expand_as(y_true)
                else:
                    p_tensor = torch.stack([combined_data[p] for p in parents], dim=1)
                    y_pred = self.student.mechanisms[node](p_tensor).squeeze()
                loss = self.loss_fn(y_pred, y_true)
                total_loss += loss
            total_loss.backward()
            self.optimizer.step()
            losses.append(total_loss.item())
        return losses[-1]

# ----------------------------------------------------------------
# 3. POLICY & DSL
# ----------------------------------------------------------------
class ExperimentalDSL:
    def __init__(self, nodes):
        self.nodes = nodes
        self.vocab = ["<PAD>", "<SOS>", "<EOS>", "DO", "MEASURE", "=", "-"] + \
                     nodes + [str(i) for i in range(-5, 6)]
        self.token2id = {t: i for i, t in enumerate(self.vocab)}
        self.id2token = {i: t for i, t in enumerate(self.vocab)}
        
    def parse_to_dict(self, command_str):
        try:
            clean_cmd = command_str.strip()
            # Use fullmatch to ensure no trailing garbage (e.g., "2.4444444...")
            match = re.fullmatch(r"DO\s+(X\d+)\s*=\s*(-?\d+(?:\.\d+)?)", clean_cmd)
            if match:
                node = match.group(1)
                value = float(match.group(2))
                if node not in self.nodes: return None 
                if not (-10 <= value <= 10): return None
                return {'target': node, 'value': value, 'samples': 200}
            return None
        except:
            return None

    def encode(self, command_str):
        tokens = command_str.split()
        return torch.tensor([self.token2id.get(t, 0) for t in tokens])

    def decode(self, token_tensor):
        tokens = [self.id2token.get(t.item(), "") for t in token_tensor]
        return " ".join([t for t in tokens if t not in ["<PAD>", "<SOS>", "<EOS>"]])

class StateEncoder(nn.Module):
    def __init__(self, n_nodes, device, d_model=128):
        super().__init__()
        self.n_nodes = n_nodes
        self.device = device
        self.node_projector = nn.Linear(3, d_model) 
        self.pos_embedding = nn.Parameter(torch.randn(1, n_nodes, d_model))
        
    def forward(self, scm_student):
        node_feats = []
        for node in scm_student.nodes:
            if isinstance(scm_student.mechanisms[node], nn.Sequential):
                w_mag = scm_student.mechanisms[node][0].weight.abs().mean().item()
            else:
                w_mag = scm_student.mechanisms[node]['mu'].item()
            uncert = 0.5
            idx = int(node[1]) 
            node_feats.append([w_mag, uncert, float(idx)])
        x = torch.tensor(node_feats, dtype=torch.float32).unsqueeze(0)
        return self.node_projector(x.to(self.device)) + self.pos_embedding.to(self.device)

class TransformerPolicy(nn.Module):
    def __init__(self, dsl, device, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.dsl = dsl
        self.device = device
        self.d_model = d_model
        self.state_encoder = StateEncoder(len(dsl.nodes), device, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.token_embedding = nn.Embedding(len(dsl.vocab), d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_dec = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, len(dsl.vocab))
        
    def forward(self, scm_student, target_seq=None):
        src = self.state_encoder(scm_student)
        memory = self.transformer_enc(src)
        if target_seq is not None:
            tgt = self.token_embedding(target_seq)
            out = self.transformer_dec(tgt, memory)
            return self.output_head(out)
        return memory

    def generate_experiment(self, scm_student, max_len=5):
        self.eval()
        memory = self.forward(scm_student)
        curr_token = torch.tensor([[self.dsl.token2id["<SOS>"]]], device=self.device)
        generated_ids = []
        for _ in range(max_len):
            tgt_emb = self.token_embedding(curr_token)
            out = self.transformer_dec(tgt_emb, memory)
            logits = self.output_head(out[:, -1, :])
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            if self.dsl.id2token.get(next_id) == "<EOS>": break
            generated_ids.append(next_id)
            curr_token = torch.cat([curr_token, torch.tensor([[next_id]], device=self.device)], dim=1)
        cmd_str = self.dsl.decode(torch.tensor(generated_ids)) # Use decode logic from DSL
        # Simple decode for custom (space separated)
        parsed = self.dsl.parse_to_dict(cmd_str)
        return cmd_str, parsed
        
    def decode_tensor(self, indices):
         # Helper for transformer to string
         tokens = [self.dsl.id2token[i.item()] for i in indices]
         clean = [t for t in tokens if t not in ["<PAD>", "<SOS>", "<EOS>"]]
         return " ".join(clean)

class HuggingFacePolicy(nn.Module):
    def __init__(self, model_name, dsl, device, token=None):
        super().__init__()
        self.dsl = dsl
        self.device = device
        logging.info(f"Loading LLM: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=token).to(device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def scm_to_prompt(self, scm):
        edges = [f"{u}->{v}" for u, v in scm.graph.edges]
        edge_str = ", ".join(edges)
        knowledge = []
        for node in scm.nodes:
            if isinstance(scm.mechanisms[node], nn.Sequential):
                mag = scm.mechanisms[node][0].weight.abs().mean().item()
                knowledge.append(f"{node}:{mag:.2f}")
        know_str = " | ".join(knowledge)
        return f"Graph: [{edge_str}]. Weights: [{know_str}]. Task: Propose intervention. Syntax: DO X[Node] = [Value]. Command: DO"

    def forward(self, scm_student, target_text_list=None):
        prompt_text = self.scm_to_prompt(scm_student)
        if target_text_list is None: target_text_list = [""]
        full_texts = [prompt_text + " " + t for t in target_text_list]
        inputs = self.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model(**inputs)
        return outputs.logits, inputs.input_ids

    def generate_experiment(self, scm_student, max_new_tokens=32):
        prompt_text = self.scm_to_prompt(scm_student)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=self.tokenizer.eos_token_id, 
                do_sample=True, 
                temperature=0.7,
                repetition_penalty=1.2
            )
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            cmd_str = full_text.split("Command:")[-1].strip()
            # Handle potential multiline garbage
            cmd_str = cmd_str.split('\n')[0].strip()
            if not cmd_str.startswith("DO"): cmd_str = "DO " + cmd_str
            return cmd_str, self.dsl.parse_to_dict(cmd_str)
        except:
            return full_text, None

# ----------------------------------------------------------------
# 4. LOSS & CRITIC
# ----------------------------------------------------------------
class ScientificCritic:
    def __init__(self, test_oracle):
        self.test_oracle = test_oracle
        self.val_data = self.test_oracle.generate(n_samples=500)
        
    def evaluate_model(self, student_scm):
        total, _ = self.evaluate_model_detailed(student_scm)
        return total

    def evaluate_model_detailed(self, student_scm):
        """Returns (total_loss, node_losses) on a fixed validation set."""
        student_scm.eval()
        total_loss = 0.0
        node_losses = {}
        with torch.no_grad():
            for node in student_scm.nodes:
                y_true = self.val_data[node]
                parents = student_scm.get_parents(node)
                if not parents:
                    y_pred = student_scm.mechanisms[node]['mu'].expand_as(y_true)
                else:
                    p_tensor = torch.stack([self.val_data[p] for p in parents], dim=1)
                    y_pred = student_scm.mechanisms[node](p_tensor).squeeze()
                loss = F.mse_loss(y_pred, y_true).item()
                node_losses[node] = loss
                total_loss += loss
        return total_loss, node_losses

    def calculate_reward(self, loss_before, loss_after):
        delta = loss_before - loss_after
        reward = delta * 100.0
        # Clip to keep extremely large deltas from destabilizing updates/metrics.
        return float(np.clip(reward, -10.0, 4000.0))

def dpo_loss(policy_model, ref_model, scm_state, winner_seq, loser_seq, beta=0.1):
    # Custom Transformer Tensor Loss
    def get_log_probs(model, state, seq):
        input_seq = seq[:, :-1]
        logits = model(state, input_seq) 
        log_probs = F.log_softmax(logits, dim=-1)
        target_tokens = seq[:, 1:].unsqueeze(-1) 
        token_log_probs = torch.gather(log_probs, -1, target_tokens).squeeze(-1)
        return token_log_probs.sum(dim=-1)

    policy_win_lp = get_log_probs(policy_model, scm_state, winner_seq)
    policy_lose_lp = get_log_probs(policy_model, scm_state, loser_seq)
    with torch.no_grad():
        ref_win_lp = get_log_probs(ref_model, scm_state, winner_seq)
        ref_lose_lp = get_log_probs(ref_model, scm_state, loser_seq)

    logits = (policy_win_lp - ref_win_lp) - (policy_lose_lp - ref_lose_lp)
    return -F.logsigmoid(beta * logits).mean()

def dpo_loss_llm(policy_model, ref_model, scm_state, win_text, lose_text, beta=0.1):
    # LLM Text Loss
    def get_log_probs(model, text):
        logits, input_ids = model(scm_state, [text])
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum(dim=-1)

    policy_win_lp = get_log_probs(policy_model, win_text)
    policy_lose_lp = get_log_probs(policy_model, lose_text)
    with torch.no_grad():
        ref_win_lp = get_log_probs(ref_model, win_text)
        ref_lose_lp = get_log_probs(ref_model, lose_text)

    logits = (policy_win_lp - ref_win_lp) - (policy_lose_lp - ref_lose_lp)
    return -F.logsigmoid(beta * logits).mean()

# ----------------------------------------------------------------
# 5. UTILS
# ----------------------------------------------------------------
def get_random_valid_command(nodes):
    target = random.choice(nodes)
    # Broad coverage across the allowed range.
    val = random.uniform(-5.0, 5.0)
    return f"DO {target} = {val:.4f}"

def save_plots(results_dir, loss_history, reward_history, targets, values, nodes):
    # 1. Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label="DPO Loss", color='purple')
    plt.title("Agent Alignment")
    plt.subplot(1, 2, 2)
    plt.plot(reward_history, label="Info Gain", color='green')
    plt.title("Discovery Progress")
    plt.savefig(os.path.join(results_dir, "training_curves.png"))
    plt.close()

    # 2. Strategy Analysis
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    node_counts = {node: targets.count(node) for node in nodes}
    sns.barplot(x=list(node_counts.keys()), y=list(node_counts.values()), palette="viridis")
    plt.title("Target Preference")
    
    plt.subplot(1, 2, 2)
    sns.histplot(values, bins=20, kde=True, color="orange")
    plt.title("Value Distribution")
    plt.savefig(os.path.join(results_dir, "strategy_analysis.png"))
    plt.close()

def visualize_contrast_save(oracle, student, results_dir):
    # Simple contrast plot saver
    try:
        relationships = []
        for child in oracle.nodes:
            parents = oracle.get_parents(child)
            if not parents: relationships.append(('Root', child))
            else:
                for parent in parents: relationships.append((parent, child))
        
        n_plots = len(relationships)
        cols = 3
        rows = (n_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
        axes = axes.flatten()
        x_range = torch.linspace(-4, 4, 100)
        
        for i, rel in enumerate(relationships):
            ax = axes[i]
            if rel[0] == 'Root':
                node = rel[1]
                true_data = oracle.generate(1000)[node].detach().numpy()
                with torch.no_grad():
                    pred_data = student.forward(1000)[node].detach().numpy()
                ax.hist(true_data, bins=30, density=True, alpha=0.3, color='black', label='Truth')
                ax.hist(pred_data, bins=30, density=True, alpha=0.3, color='red', label='Student')
                ax.set_title(f"Root: {node}")
            else:
                parent, child = rel
                parents_list = oracle.get_parents(child)
                data_context = {}
                for p in parents_list:
                    data_context[p] = x_range if p == parent else torch.zeros(100)
                
                y_true = oracle.mechanisms(data_context, child).detach().numpy()
                with torch.no_grad():
                    if len(parents_list) > 0:
                        p_tensor = torch.stack([data_context[p] for p in parents_list], dim=1)
                        y_pred = student.mechanisms[child](p_tensor).squeeze().detach().numpy()
                    else:
                        y_pred = student.mechanisms[child]['mu'].expand(100).detach().numpy()

                ax.plot(x_range.numpy(), y_true, 'k--', lw=3, label='Truth')
                ax.plot(x_range.numpy(), y_pred, 'r-', lw=2, alpha=0.8, label='Student')
                ax.set_title(f"{parent} -> {child}")
        
        for j in range(i + 1, len(axes)): axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "mechanism_contrast.png"))
        plt.close()
    except Exception as e:
        logging.error(f"Failed to save contrast plot: {e}")

# ----------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ACE Causal Discovery Experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B", help="HF Model Name")
    parser.add_argument("--custom", action="store_true", help="Use Custom Transformer instead of LLM")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=25, help="Max steps per episode")
    parser.add_argument("--candidates", type=int, default=4, help="Candidates per step")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning Rate")
    parser.add_argument("--learner_lr", type=float, default=1e-2, help="Learner (student SCM) learning rate")
    parser.add_argument("--buffer_steps", type=int, default=50, help="Learner replay buffer length")
    parser.add_argument("--cov_bonus", type=float, default=25.0, help="Coverage bonus scale (discourages target collapse)")
    parser.add_argument("--eps_explore", type=float, default=0.10, help="Exploration probability (coverage-seeking)")
    parser.add_argument("--token", type=str, default=None, help="HF Auth Token")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    # Setup Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Logging Setup
    logging.basicConfig(
        filename=os.path.join(run_dir, "experiment.log"),
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting ACE Experiment on {device}")
    logging.info(f"Config: {args}")

    # 1. Setup Environment
    M_star = GroundTruthSCM()
    executor = ExperimentExecutor(M_star)
    temp_nodes = sorted(list(M_star.graph.nodes))
    dsl = ExperimentalDSL(temp_nodes)
    critic = ScientificCritic(M_star)

    # 2. Setup Agent
    use_pretrained = not args.custom
    if use_pretrained:
        hf_token = args.token or os.getenv("HF_TOKEN")
        policy_net = HuggingFacePolicy(args.model, dsl, device, token=hf_token)
    else:
        policy_net = TransformerPolicy(dsl, device).to(device)

    ref_policy = copy.deepcopy(policy_net)
    ref_policy.eval()
    optimizer_agent = optim.Adam(policy_net.parameters(), lr=args.lr)

    # 3. Training Loop
    loss_history = []
    reward_history = []
    score_history = []
    cov_bonus_history = []
    target_history = []
    value_history = []
    episode_history = []
    step_history = []
    
    logging.info(f"--- Starting Discovery Loop ({args.episodes} Episodes) ---")

    recent_action_counts = deque(maxlen=500)

    for episode in range(args.episodes):
        current_student = StudentSCM(M_star)
        learner = SCMLearner(current_student, lr=args.learner_lr, buffer_steps=args.buffer_steps)
        episode_action_counts = Counter()

        if episode % 10 == 0:
            logging.info(f"--- Episode {episode+1} Start ---")

        for step in range(args.steps):
            loss_start, node_losses_start = critic.evaluate_model_detailed(current_student)
            if loss_start < 0.5:
                if episode % 10 == 0:
                     logging.info(f"  > Solved at Step {step}! Loss: {loss_start:.4f}")
                break

            candidates = []
            for k in range(args.candidates):
                cmd_str, plan = policy_net.generate_experiment(current_student)
                if plan is None:
                    candidates.append((cmd_str, -10.0, 0.0, -10.0, None))
                else:
                    student_clone = copy.deepcopy(current_student)
                    # Clone learner with the same replay buffer to make candidate scoring realistic.
                    initial_buffer = [{kk: vv.detach().clone() for kk, vv in d.items()} for d in learner.buffer]
                    clone_learner = SCMLearner(
                        student_clone,
                        lr=args.learner_lr,
                        buffer_steps=args.buffer_steps,
                        initial_buffer=initial_buffer,
                    )
                    data_t = executor.run_experiment(plan)
                    clone_learner.train_step(data_t)
                    loss_end = critic.evaluate_model(student_clone)
                    reward = critic.calculate_reward(loss_start, loss_end)
                    tgt = plan.get("target")
                    node_weight = float(node_losses_start.get(tgt, 0.0))
                    denom = float(sum(node_losses_start.values())) + 1e-8
                    norm_weight = node_weight / denom
                    under_sample = 1.0 / np.sqrt(1.0 + episode_action_counts.get(tgt, 0))
                    cov_bonus = args.cov_bonus * norm_weight * under_sample
                    score = reward + cov_bonus
                    candidates.append((cmd_str, reward, cov_bonus, score, plan))

            valid_cmds = [c for c, r, cb, s, p in candidates if r > -9.0]
            if not valid_cmds:
                teacher_cmd = get_random_valid_command(dsl.nodes)
                teacher_plan = dsl.parse_to_dict(teacher_cmd)
                tgt = teacher_plan.get("target") if teacher_plan else None
                node_weight = float(node_losses_start.get(tgt, 0.0))
                denom = float(sum(node_losses_start.values())) + 1e-8
                norm_weight = node_weight / denom
                under_sample = 1.0 / np.sqrt(1.0 + episode_action_counts.get(tgt, 0))
                cov_bonus = args.cov_bonus * norm_weight * under_sample
                score = 0.1 + cov_bonus
                candidates.append((teacher_cmd, 0.1, cov_bonus, score, teacher_plan))

            # Rank by score (reward + coverage bonus).
            sorted_cands = sorted(candidates, key=lambda x: x[3], reverse=True)
            if random.random() < args.eps_explore:
                explore_pool = [c for c in candidates if c[1] > -9.0]
                explore_sorted = sorted(explore_pool, key=lambda x: x[2], reverse=True)
                winner_cmd, winner_reward, winner_cov_bonus, winner_score, winner_plan = explore_sorted[0]
            else:
                winner_cmd, winner_reward, winner_cov_bonus, winner_score, winner_plan = sorted_cands[0]
            loser_cmd, loser_reward, loser_cov_bonus, loser_score, _ = sorted_cands[-1]

            # Update
            if winner_score > loser_score:
                optimizer_agent.zero_grad()
                if use_pretrained:
                    loss = dpo_loss_llm(policy_net, ref_policy, current_student, winner_cmd, loser_cmd)
                else:
                    # Tensor path for custom
                    win_seq = dsl.encode(winner_cmd).unsqueeze(0).to(device)
                    lose_seq = dsl.encode(loser_cmd).unsqueeze(0).to(device)
                    max_len = max(win_seq.shape[1], lose_seq.shape[1])
                    win_pad = F.pad(win_seq, (0, max_len - win_seq.shape[1]), value=dsl.token2id["<PAD>"])
                    lose_pad = F.pad(lose_seq, (0, max_len - lose_seq.shape[1]), value=dsl.token2id["<PAD>"])
                    loss = dpo_loss(policy_net, ref_policy, current_student, win_pad, lose_pad)
                    
                loss.backward()
                optimizer_agent.step()
                loss_history.append(loss.item())
                reward_history.append(winner_reward)
                score_history.append(winner_score)
                cov_bonus_history.append(winner_cov_bonus)
                if winner_plan:
                    target_history.append(winner_plan.get("target"))
                    value_history.append(winner_plan.get("value"))
                else:
                    target_history.append(None)
                    value_history.append(None)
                episode_history.append(episode)
                step_history.append(step)
            
            # Execute
            if winner_plan:
                tgt = winner_plan.get("target")
                episode_action_counts[tgt] += 1
                recent_action_counts.append(tgt)

                # Simple collapse indicator for logs.
                if len(recent_action_counts) >= 50:
                    rc = Counter(recent_action_counts)
                    top_node, top_count = rc.most_common(1)[0]
                    top_frac = top_count / len(recent_action_counts)
                else:
                    top_node, top_frac = None, 0.0

                if winner_score > 0.2 or (episode % 10 == 0 and step == 0):
                    logging.info(
                        f"  Step {step}: '{winner_cmd}' (Reward: {winner_reward:.2f}, "
                        f"Cov: {winner_cov_bonus:.2f}, Score: {winner_score:.2f}, "
                        f"RecentTop: {top_node}@{top_frac:.0%})"
                    )
                real_data = executor.run_experiment(winner_plan)
                learner.train_step(real_data)

    # 4. Final Evaluation
    logging.info("--- Running Final Evaluation ---")
    eval_targets = []
    eval_values = []
    
    # Use the LAST student state for evaluation visualization
    visualize_contrast_save(M_star, current_student, run_dir)
    
    n_eval = 100
    n_parsed = 0
    for _ in range(n_eval):
        _, plan = policy_net.generate_experiment(current_student)
        if plan:
            n_parsed += 1
            eval_targets.append(plan["target"])
            eval_values.append(plan["value"])

    # If parsing fails completely (common when the LLM drifts off-DSL), fall back to
    # the observed executed interventions from training so plots are still informative.
    if not eval_targets:
        fallback_targets = [t for t in target_history if t]
        fallback_values = [v for v, t in zip(value_history, target_history) if t]
        eval_targets = fallback_targets[-500:]
        eval_values = fallback_values[-500:]
        logging.info(
            f"Final eval produced 0 parsed commands; falling back to last {len(eval_targets)} training interventions for plots."
        )
    else:
        logging.info(f"Final eval parse success: {n_parsed}/{n_eval} ({(n_parsed / max(n_eval, 1)):.0%})")
    
    save_plots(run_dir, loss_history, reward_history, eval_targets, eval_values, dsl.nodes)
    
    # Save Metrics
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
    
    logging.info(f"Experiment Complete. Results saved to {run_dir}")

if __name__ == "__main__":
    main()
