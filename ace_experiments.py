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
import sys
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
        """
        Runs an experiment and returns a batch dict:
          - data: dict[node] -> Tensor[n]
          - intervened: node name that was directly intervened on (or None)

        IMPORTANT: The learner must NOT train a node's structural mechanism on samples
        where that node was directly intervened on (do-operator breaks the mechanism).
        """
        if intervention_plan is None:
            return {"data": self.env.generate(n_samples=100), "intervened": None}
        
        target = intervention_plan.get('target')
        value = intervention_plan.get('value')
        n_samples = intervention_plan.get('samples', 100)
        
        if target:
            return {
                "data": self.env.generate(n_samples, interventions={target: value}),
                "intervened": target,
            }
        else:
            return {"data": self.env.generate(n_samples), "intervened": None}

class SCMLearner:
    def __init__(self, student_scm, lr=0.01, buffer_steps=50, initial_buffer=None):
        self.student = student_scm
        self.optimizer = optim.Adam(self.student.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = list(initial_buffer) if initial_buffer is not None else []
        self.buffer_steps = buffer_steps

    def _normalize_batch(self, batch):
        """
        Accepts either:
          - raw data dict[node]->Tensor
          - {"data": data_dict, "intervened": node|None}
        """
        if isinstance(batch, dict) and "data" in batch:
            return {"data": batch["data"], "intervened": batch.get("intervened")}
        # Back-compat: treat plain dict as observational batch.
        if isinstance(batch, dict) and batch and all(isinstance(v, torch.Tensor) for v in batch.values()):
            return {"data": batch, "intervened": None}
        raise ValueError("Unsupported batch format for SCMLearner.train_step")
        
    def train_step(self, data, n_epochs=50):
        self.student.train()

        # --- FIX: FAST ADAPTATION PHASE ---
        # Train strictly on the NEW data first to ensure immediate mechanism updates.
        # This synchronizes the Loss Drop with the Action, fixing Reward Misattribution.
        batch_new = self._normalize_batch(data)
        
        # Fast adaptation loop (e.g. 20% of total epochs, minimum 5)
        fast_epochs = max(5, int(n_epochs * 0.2)) 
        for _ in range(fast_epochs):
            self.optimizer.zero_grad()
            
            # Fast adaptation: loss on NEW batch only
            total_loss = 0
            # Reuse the loss calculation logic for a single batch
            # We must manually extract the logic or helper it, but for minimal change we inline here
            # Since batch_new is just one dict, we don't need the complex collate logic
            
            # Simple batch processing for fast adaptation
            for node in self.student.nodes:
                parents = self.student.get_parents(node)
                y_true = batch_new["data"][node]
                
                # Check intervention mask
                mask = None
                if batch_new.get("intervened") == node:
                    mask = torch.zeros(y_true.shape[0], dtype=torch.bool)
                else:
                    mask = torch.ones(y_true.shape[0], dtype=torch.bool)
                
                if mask.sum().item() == 0:
                    continue
                    
                if not parents:
                    y_pred = self.student.mechanisms[node]['mu'].expand_as(y_true)
                else:
                    p_tensor = torch.stack([batch_new["data"][p] for p in parents], dim=1)
                    y_pred = self.student.mechanisms[node](p_tensor).squeeze()
                    
                loss = self.loss_fn(y_pred[mask], y_true[mask])
                total_loss += loss
            
            total_loss.backward()
            self.optimizer.step()
            
        # --- END FIX ---

        # Standard Replay Phase (Consolidation)
        self.buffer.append(batch_new)
        if len(self.buffer) > self.buffer_steps:
            self.buffer.pop(0)
            
        # Collate data
        combined_data = {}
        combined_mask = {}
        # Assumes all data dicts have the same nodes (which they should)
        nodes = list(batch_new["data"].keys())
        for node in nodes:
            tensors = [b["data"][node] for b in self.buffer]
            combined_data[node] = torch.cat(tensors, dim=0)

            # Mask out samples where THIS node was directly intervened on.
            masks = []
            for b in self.buffer:
                n = b["data"][node].shape[0]
                if b.get("intervened") == node:
                    masks.append(torch.zeros(n, dtype=torch.bool))
                else:
                    masks.append(torch.ones(n, dtype=torch.bool))
            combined_mask[node] = torch.cat(masks, dim=0)
            
        losses = []
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            total_loss = 0
            for node in self.student.nodes:
                parents = self.student.get_parents(node)
                y_true = combined_data[node]
                mask = combined_mask.get(node, None)
                if mask is not None and mask.sum().item() == 0:
                    # If we only ever intervened on this node in the replay window, skip it.
                    continue
                if not parents:
                    y_pred = self.student.mechanisms[node]['mu'].expand_as(y_true)
                else:
                    p_tensor = torch.stack([combined_data[p] for p in parents], dim=1)
                    y_pred = self.student.mechanisms[node](p_tensor).squeeze()
                if mask is not None:
                    loss = self.loss_fn(y_pred[mask], y_true[mask])
                else:
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
    def __init__(self, nodes, value_min=-5.0, value_max=5.0):
        self.nodes = nodes
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.vocab = ["<PAD>", "<SOS>", "<EOS>", "DO", "MEASURE", "=", "-"] + \
                     nodes + [str(i) for i in range(-5, 6)]
        self.token2id = {t: i for i, t in enumerate(self.vocab)}
        self.id2token = {i: t for i, t in enumerate(self.vocab)}
        
    def parse_to_dict(self, command_str):
        try:
            clean_cmd = command_str.strip()
            # Use fullmatch to ensure no trailing garbage
            # Case-insensitive matching for "DO"
            # Enhanced regex to handle scientific notation (e.g., 1e-5, 2.5E+3)
            match = re.fullmatch(r"(?i)DO\s+(X\d+)\s*=\s*(-?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)", clean_cmd)
            if match:
                node = match.group(1).upper()  # Normalize to uppercase
                value = float(match.group(2))
                if node not in self.nodes: return None 
                # Keep values in a well-covered, informative range (matches teacher sampling).
                if not (self.value_min <= value <= self.value_max): return None
                return {'target': node, 'value': value, 'samples': 200}
            return None
        except:
            return None

    def parse_to_dict_lenient(self, text, clip_out_of_range=True):
        """
        Lenient parse that extracts the first valid DO command substring from `text`.
        If clip_out_of_range is True, clamps the value into [value_min, value_max] instead of failing.
        """
        try:
            if text is None:
                logging.debug(f"PARSE: text is None")
                return None
            # Case-insensitive search for DO command
            # Enhanced regex to handle scientific notation and various number formats
            m = re.search(r"(?i)DO\s+(X\d+)\s*=\s*(-?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)", str(text))
            if not m:
                logging.debug(f"PARSE: No regex match for pattern in text: '{str(text)[:100]}'")
                return None
            node = m.group(1).upper()  # Normalize to uppercase
            value = float(m.group(2))
            if node not in self.nodes:
                logging.debug(f"PARSE: Node '{node}' not in valid nodes: {self.nodes}")
                return None
            if clip_out_of_range:
                value = max(self.value_min, min(self.value_max, value))
            else:
                if not (self.value_min <= value <= self.value_max):
                    logging.debug(f"PARSE: Value {value} out of range [{self.value_min}, {self.value_max}]")
                    return None
            return {"target": node, "value": value, "samples": 200}
        except Exception as e:
            logging.debug(f"PARSE: Exception {e}")
            return None

    def encode(self, command_str):
        tokens = command_str.split()
        return torch.tensor([self.token2id.get(t, 0) for t in tokens], dtype=torch.long)

    def decode(self, token_tensor):
        tokens = [self.id2token.get(t.item(), "") for t in token_tensor]
        return " ".join([t for t in tokens if t not in ["<PAD>", "<SOS>", "<EOS>"]])

class StateEncoder(nn.Module):
    def __init__(self, n_nodes, device, d_model=128):
        super().__init__()
        self.n_nodes = n_nodes
        self.device = device
        # Added validation loss to features: [w_mag, uncert, loss, idx]
        self.node_projector = nn.Linear(4, d_model) 
        self.pos_embedding = nn.Parameter(torch.randn(1, n_nodes, d_model))
        
    def forward(self, scm_student, node_losses=None):
        node_feats = []
        for node in scm_student.nodes:
            if isinstance(scm_student.mechanisms[node], nn.Sequential):
                w_mag = scm_student.mechanisms[node][0].weight.abs().mean().item()
            else:
                w_mag = scm_student.mechanisms[node]['mu'].item()
            uncert = 0.5
            # Inject validation loss if available, else 0.0
            loss_val = node_losses.get(node, 0.0) if node_losses else 0.0
            idx = int(node[1]) 
            node_feats.append([w_mag, uncert, loss_val, float(idx)])
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
        
    def forward(self, scm_student, target_seq=None, node_losses=None):
        src = self.state_encoder(scm_student, node_losses=node_losses)
        memory = self.transformer_enc(src)
        if target_seq is not None:
            tgt = self.token_embedding(target_seq)
            out = self.transformer_dec(tgt, memory)
            return self.output_head(out)
        return memory

    def generate_experiment(self, scm_student, node_losses=None, max_len=5):
        self.eval()
        memory = self.forward(scm_student, node_losses=node_losses)
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
            
    def scm_to_prompt(self, scm, node_losses=None):
        edges = [f"{u}->{v}" for u, v in scm.graph.edges]
        edge_str = ", ".join(edges)
        knowledge = []
        for node in scm.nodes:
            if isinstance(scm.mechanisms[node], nn.Sequential):
                mag = scm.mechanisms[node][0].weight.abs().mean().item()
                knowledge.append(f"{node}:{mag:.2f}")
        know_str = " | ".join(knowledge)
        
        # Inject validation losses into prompt
        loss_info = ""
        if node_losses:
            loss_strs = [f"{n}:{node_losses.get(n,0.0):.2f}" for n in scm.nodes]
            loss_info = "Losses: [" + " | ".join(loss_strs) + "]. "
            
        # Enhanced prompt with explicit format examples
        valid_nodes = ", ".join(scm.nodes)
        return (
            f"Graph: [{edge_str}]. Weights: [{know_str}]. {loss_info}"
            f"Task: Propose a causal intervention to discover mechanisms. "
            f"Valid nodes: {valid_nodes}. "
            f"Value range: [{self.dsl.value_min}, {self.dsl.value_max}]. "
            f"Format: DO X[digit] = [number]. "
            f"Examples: 'DO X1 = 2.5', 'DO X3 = -1.8'. "
            f"Output only the command, nothing else.\n"
            f"Command: DO"
        )

    def forward(self, scm_student, target_text_list=None, node_losses=None):
        prompt_text = self.scm_to_prompt(scm_student, node_losses=node_losses)
        if target_text_list is None: target_text_list = [""]
        full_texts = [prompt_text + " " + t for t in target_text_list]
        inputs = self.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model(**inputs)
        return outputs.logits, inputs.input_ids

    def generate_experiment(self, scm_student, node_losses=None, max_new_tokens=32):
        prompt_text = self.scm_to_prompt(scm_student, node_losses=node_losses)
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
        
        # Extract only the generated portion (after the prompt)
        # The model may echo the prompt, so we need to remove it
        if full_text.startswith(prompt_text):
            generated_text = full_text[len(prompt_text):].strip()
        else:
            # If prompt not found at start, try to find "Command:" marker
            if "Command:" in full_text:
                generated_text = full_text.split("Command:")[-1].strip()
            else:
                generated_text = full_text
        
        try:
            # Try parsing the generated text
            plan = self.dsl.parse_to_dict_lenient(generated_text, clip_out_of_range=True)
            
            # Fallback: try parsing the full text if generated_text didn't work
            if plan is None and generated_text != full_text:
                plan = self.dsl.parse_to_dict_lenient(full_text, clip_out_of_range=True)
            
            if plan is None:
                # Enhanced logging for parse failures
                logging.debug(f"PARSE_FAIL: generated='{generated_text[:150]}', full='{full_text[:150]}'")
                return generated_text if generated_text else full_text, None

            # Canonicalize the command string to exactly match the DSL.
            cmd_str = f"DO {plan['target']} = {float(plan['value']):.4f}"
            return cmd_str, plan
        except Exception as e:
            logging.debug(f"PARSE_EXCEPTION: {e}, generated='{generated_text[:150] if generated_text else None}'")
            return generated_text if generated_text else full_text, None

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

    def evaluate_mechanisms_detailed(self, student_scm, n=500):
        """
        Evaluates each non-root mechanism on an *interventional-style* validation set where
        parents are sampled independently over a broad range. This better reflects mechanism
        fidelity than purely observational rollouts.
        """
        student_scm.eval()
        total_loss = 0.0
        node_losses = {}

        # Evaluate roots only on mean (Student roots don't model stochasticity).
        with torch.no_grad():
            for node in student_scm.nodes:
                parents = student_scm.get_parents(node)
                if parents:
                    continue
                # Compare predicted mean to empirical mean from oracle validation data.
                y_true = self.val_data[node]
                y_pred = student_scm.mechanisms[node]["mu"].expand_as(y_true)
                loss = F.mse_loss(y_pred, y_true).item()
                node_losses[node] = loss

        # Evaluate mechanisms with parents using independent parent samples.
        with torch.no_grad():
            for node in student_scm.nodes:
                parents = student_scm.get_parents(node)
                if not parents:
                    continue
                parent_ctx = {p: (torch.rand(n) * 8.0 - 4.0) for p in parents}  # U[-4, 4]
                y_true = self.test_oracle.mechanisms(parent_ctx, node, n_samples=n)
                p_tensor = torch.stack([parent_ctx[p] for p in parents], dim=1)
                y_pred = student_scm.mechanisms[node](p_tensor).squeeze()
                loss = F.mse_loss(y_pred, y_true).item()
                node_losses[node] = loss

        # Weight towards child mechanisms (non-roots) by default; roots contribute weakly.
        for node, loss in node_losses.items():
            w = 0.2 if not student_scm.get_parents(node) else 1.0
            total_loss += w * loss

        return total_loss, node_losses

    def calculate_reward(self, loss_before, loss_after):
        delta = loss_before - loss_after
        # CRITICAL FIX: Scale reward down to make bonuses competitive
        # Old: reward = delta * 100.0 (rewards >> bonuses, causing collapse)
        # New: reward = delta * 10.0 (rewards ~ bonuses, enabling exploration)
        reward = delta * 10.0
        # Clip to keep extremely large deltas from destabilizing updates/metrics.
        # Also clip negative rewards to avoid severe punishment for exploration
        return float(np.clip(reward, -2.0, 400.0))

def dpo_loss(policy_model, ref_model, scm_state, winner_seq, loser_seq, node_losses=None, beta=0.1):
    # Custom Transformer Tensor Loss
    def get_log_probs(model, state, seq, losses):
        input_seq = seq[:, :-1]
        logits = model(state, input_seq, node_losses=losses) 
        log_probs = F.log_softmax(logits, dim=-1)
        target_tokens = seq[:, 1:].unsqueeze(-1) 
        token_log_probs = torch.gather(log_probs, -1, target_tokens).squeeze(-1)
        return token_log_probs.sum(dim=-1)

    policy_win_lp = get_log_probs(policy_model, scm_state, winner_seq, node_losses)
    policy_lose_lp = get_log_probs(policy_model, scm_state, loser_seq, node_losses)
    with torch.no_grad():
        ref_win_lp = get_log_probs(ref_model, scm_state, winner_seq, node_losses)
        ref_lose_lp = get_log_probs(ref_model, scm_state, loser_seq, node_losses)

    logits = (policy_win_lp - ref_win_lp) - (policy_lose_lp - ref_lose_lp)
    return -F.logsigmoid(beta * logits).mean()

def dpo_loss_llm(policy_model, ref_model, scm_state, win_text, lose_text, node_losses=None, beta=0.1):
    # LLM Text Loss
    def get_log_probs(model, text, losses):
        logits, input_ids = model(scm_state, [text], node_losses=losses)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum(dim=-1)

    policy_win_lp = get_log_probs(policy_model, win_text, node_losses)
    policy_lose_lp = get_log_probs(policy_model, lose_text, node_losses)
    with torch.no_grad():
        ref_win_lp = get_log_probs(ref_model, win_text, node_losses)
        ref_lose_lp = get_log_probs(ref_model, lose_text, node_losses)

    logits = (policy_win_lp - ref_win_lp) - (policy_lose_lp - ref_lose_lp)
    return -F.logsigmoid(beta * logits).mean()

# ----------------------------------------------------------------
# 5. UTILS
# ----------------------------------------------------------------
def get_random_valid_command_range(nodes, value_min=-5.0, value_max=5.0):
    target = random.choice(nodes)
    val = random.uniform(float(value_min), float(value_max))
    return f"DO {target} = {val:.4f}"

def _impact_weight(graph, target, node_losses):
    """
    Impact-aware weight for an intervention on `target`.

    Since do-interventions replace the mechanism of the intervened node, intervening on a node
    does not directly help fit that node's mechanism. Instead, it provides informative parent
    coverage for *descendants*.
    """
    try:
        desc = nx.descendants(graph, target)
    except Exception:
        desc = set()
    if not desc:
        return 0.0
    return float(sum(float(node_losses.get(n, 0.0)) for n in desc))

def _direct_child_impact_weight(graph, target, node_losses, normalize=True):
    """
    Direct-child impact weight for an intervention on `target`.

    Intervening on a parent is most directly useful for identifying the mechanisms of its
    immediate children (especially multi-parent children like X3).
    
    If normalize=True, returns Average Child Loss (structure-independent).
    If normalize=False, returns Total Child Loss (favors nodes with many children).
    """
    try:
        children = list(graph.successors(target))
    except Exception:
        children = []
    if not children:
        return 0.0

    total = 0.0
    for child in children:
        w = 1.0
        try:
            n_par = len(list(graph.predecessors(child)))
        except Exception:
            n_par = 0
        if n_par >= 2:
            w *= 2.0
        total += w * float(node_losses.get(child, 0.0))
        
    if normalize:
        return float(total) / len(children)
    return float(total)

def _disentanglement_bonus(graph, target, node_losses):
    """
    Bonus for interventions that break dependencies between parents of a common child.
    (Triangle Breaking / Disentanglement).
    
    If target T and another node P are both parents of C, and P->T exists, 
    then intervening on T breaks the correlation induced by P->T. 
    This allows the learner to see T and P varying independently, which is 
    CRITICAL for learning the mechanism of C (especially if C = f(P, T) is complex).
    
    CRITICAL FIX: Increased bonus magnitude to compete with scaled-down rewards.
    """
    total_bonus = 0.0
    try:
        # Find all children where 'target' is a parent
        children = list(graph.successors(target))
        for child in children:
            # Check parents of this child
            parents = list(graph.predecessors(child))
            if len(parents) < 2:
                continue
                
            # Check if any OTHER parent 'P' has a connection to 'target'
            # Specifically, we care if P -> target (so target is a mediator).
            # Intervening on target breaks P -> target.
            for p in parents:
                if p == target: continue
                
                # Check if P is a parent of target (P -> T -> C structure)
                if graph.has_edge(p, target):
                    # We found a triangle P -> T -> C (and P -> C).
                    # Intervening on T is highly valuable for C.
                    child_loss = float(node_losses.get(child, 0.0))
                    # CRITICAL FIX: Increased from 20.0 to 100.0 to compete with rewards
                    total_bonus += 100.0 * child_loss 
                    
    except Exception:
        pass
        
    return total_bonus

def get_teacher_command_impact(nodes, graph, node_losses, value_min=-5.0, value_max=5.0):
    """
    Teacher injection that prefers intervening on nodes that can help the most (by descendant loss).
    Falls back to random if all impacts are zero.
    """
    impacts = []
    for n in nodes:
        # Prefer direct-child impact to reduce collapse onto distant ancestors.
        # Use normalize=True to prevent teacher bias towards high-degree nodes (X1).
        impacts.append(_direct_child_impact_weight(graph, n, node_losses, normalize=True))
    total = sum(impacts)
    if total <= 0:
        return get_random_valid_command_range(nodes, value_min=value_min, value_max=value_max)
    # Sample proportional to impact (soft preference, avoids determinism).
    r = random.random() * total
    acc = 0.0
    for n, w in zip(nodes, impacts):
        acc += w
        if acc >= r:
            val = random.uniform(float(value_min), float(value_max))
            return f"DO {n} = {val:.4f}"
    return get_random_valid_command_range(nodes, value_min=value_min, value_max=value_max)

def _bin_index(value, value_min, value_max, n_bins):
    vmin = float(value_min)
    vmax = float(value_max)
    if vmax <= vmin:
        return 0
    x = float(value)
    # Clamp then bucketize.
    x = max(vmin, min(vmax, x))
    t = (x - vmin) / (vmax - vmin)
    idx = int(t * n_bins)
    if idx == n_bins:
        idx = n_bins - 1
    return max(0, min(n_bins - 1, idx))

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
    # Seaborn >=0.14 deprecates `palette` without `hue`.
    x = list(node_counts.keys())
    y = list(node_counts.values())
    sns.barplot(x=x, y=y, hue=x, palette="viridis", legend=False)
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
    parser.add_argument("--learner_lr", type=float, default=2e-3, help="Learner (student SCM) learning rate")
    parser.add_argument("--learner_epochs", type=int, default=100, help="Learner training epochs per step")
    parser.add_argument("--buffer_steps", type=int, default=50, help="Learner replay buffer length")
    parser.add_argument("--patience_steps", type=int, default=6, help="Early stop episode if no loss improvement for this many steps")
    parser.add_argument("--min_delta", type=float, default=1e-3, help="Minimum loss improvement to reset early stop patience")
    parser.add_argument("--warmup_steps", type=int, default=3, help="Do not early-stop before this many steps")
    # CRITICAL FIX: All bonus/penalty parameters rebalanced to compete with scaled-down rewards
    parser.add_argument("--cov_bonus", type=float, default=60.0, help="Coverage bonus scale (INCREASED to compete with reduced rewards)")
    parser.add_argument("--eps_explore", type=float, default=0.10, help="Exploration probability (coverage-seeking)")
    parser.add_argument("--undersampled_bonus", type=float, default=100.0, help="Strong bonus for severely under-sampled nodes (INCREASED from 50.0)")
    parser.add_argument("--diversity_constraint", action="store_true", help="Enforce mandatory diversity: reject candidates targeting over-sampled nodes when collapse detected")
    parser.add_argument("--diversity_threshold", type=float, default=0.60, help="Threshold for mandatory diversity enforcement (e.g., 60% triggers constraint)")
    parser.add_argument("--val_bonus", type=float, default=1.5, help="Value novelty bonus scale (discourages repeated values)")
    parser.add_argument("--value_min", type=float, default=-5.0, help="Minimum intervention value accepted by the DSL")
    parser.add_argument("--value_max", type=float, default=5.0, help="Maximum intervention value accepted by the DSL")
    parser.add_argument("--n_value_bins", type=int, default=11, help="Discretization bins for value coverage bonus")
    parser.add_argument("--bin_bonus", type=float, default=8.0, help="Value coverage bonus scale (encourages spanning the value range)")
    parser.add_argument("--disentangle_bonus", type=float, default=20.0, help="[DEPRECATED - bonus now computed internally] Bonus for breaking parent-parent correlations")
    parser.add_argument("--collapse_threshold", type=float, default=0.30, help="RecentTop fraction above which collapse penalty applies (lowered to detect collapse earlier)")
    parser.add_argument("--collapse_penalty", type=float, default=150.0, help="Penalty scale (INCREASED from 80.0 for stronger deterrence)")
    parser.add_argument("--leaf_penalty", type=float, default=40.0, help="Penalty for intervening on leaf nodes (INCREASED from 25.0)")
    parser.add_argument("--parent_balance_bonus", type=float, default=80.0, help="Bonus for balanced interventions among parents (INCREASED from 20.0 for colliders)")
    parser.add_argument("--token", type=str, default=None, help="HF Auth Token")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--debug_parsing", action="store_true", help="Enable detailed parse debug logging")
    args = parser.parse_args()

    # Setup Directories
    run_started_at = datetime.now()
    timestamp = run_started_at.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Logging Setup
    log_level = logging.DEBUG if args.debug_parsing else logging.INFO
    logging.basicConfig(
        filename=os.path.join(run_dir, "experiment.log"),
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(log_level)
    logging.getLogger('').addHandler(console)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting ACE Experiment on {device}")
    logging.info(f"Config: {args}")
    logging.info(f"Run started at: {run_started_at.isoformat(timespec='seconds')}")
    logging.info(f"Run directory: {run_dir}")
    logging.info(f"Command: {' '.join(sys.argv)}")

    # 1. Setup Environment
    M_star = GroundTruthSCM()
    executor = ExperimentExecutor(M_star)
    temp_nodes = sorted(list(M_star.graph.nodes))
    dsl = ExperimentalDSL(temp_nodes, value_min=args.value_min, value_max=args.value_max)
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

    # Training-time parsing / fallback instrumentation.
    train_candidates_total = 0
    train_candidates_parsed = 0
    train_candidates_invalid = 0
    train_steps_total = 0
    train_steps_with_any_valid = 0
    train_steps_teacher_fallback = 0
    recent_parse_failures = deque(maxlen=10)  # Keep track of recent parsing failures
    
    # NEW: Per-node loss tracking for collider diagnostics
    node_loss_tracking = []
    
    # NEW: Intervention coverage tracking for multi-parent nodes
    intervention_coverage_tracking = []
    parent_intervention_counts = {n: Counter() for n in M_star.nodes if len(list(M_star.graph.predecessors(n))) >= 2}
    
    logging.info(f"--- Starting Discovery Loop ({args.episodes} Episodes) ---")
    
    # Identify collider nodes for special tracking
    collider_nodes = [n for n in M_star.nodes if len(list(M_star.graph.predecessors(n))) >= 2]
    if collider_nodes:
        logging.info(f"Collider nodes identified (multi-parent): {collider_nodes}")

    recent_action_counts = deque(maxlen=500)
    recent_values_by_target = {n: deque(maxlen=200) for n in dsl.nodes}
    recent_value_bins_by_target = {n: deque(maxlen=200) for n in dsl.nodes}

    for episode in range(args.episodes):
        current_student = StudentSCM(M_star)
        learner = SCMLearner(current_student, lr=args.learner_lr, buffer_steps=args.buffer_steps)
        episode_action_counts = Counter()
        best_mech_loss = float("inf")
        no_improve_steps = 0

        if episode % 10 == 0:
            # Report cumulative parsing statistics every 10 episodes
            parse_rate = train_candidates_parsed / max(train_candidates_total, 1) if train_candidates_total > 0 else 0.0
            fallback_rate = train_steps_teacher_fallback / max(train_steps_total, 1) if train_steps_total > 0 else 0.0
            logging.info(
                f"--- Episode {episode+1} Start --- "
                f"[Cumulative Parse: {train_candidates_parsed}/{train_candidates_total} ({parse_rate:.1%}), "
                f"Teacher Fallback: {train_steps_teacher_fallback}/{train_steps_total} ({fallback_rate:.1%})]"
            )
            # Show recent parse failure examples for diagnosis
            if recent_parse_failures:
                sample_failures = list(recent_parse_failures)[:3]
                logging.info(f"  Recent parse failure examples:")
                for idx, failure in enumerate(sample_failures):
                    logging.info(f"    [{idx+1}] '{failure[:200]}'")


        for step in range(args.steps):
            train_steps_total += 1
            # Evaluate mechanisms on interventional-style validation for better causal fidelity.
            loss_start, node_losses_start = critic.evaluate_mechanisms_detailed(current_student)
            
            # NEW: Track per-node losses for diagnosis
            node_loss_record = {
                "episode": episode,
                "step": step,
                "total_loss": loss_start,
            }
            for node, loss_val in node_losses_start.items():
                node_loss_record[f"loss_{node}"] = loss_val
            node_loss_tracking.append(node_loss_record)
            
            if loss_start < best_mech_loss - args.min_delta:
                best_mech_loss = loss_start
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                if step >= args.warmup_steps and no_improve_steps >= args.patience_steps:
                    if episode % 10 == 0:
                        logging.info(
                            f"  > Early-stopping episode at Step {step} (no improvement for {no_improve_steps} steps). "
                            f"BestLoss: {best_mech_loss:.4f}, CurrentLoss: {loss_start:.4f}"
                        )
                    break
            if loss_start < 0.5:
                if episode % 10 == 0:
                     logging.info(f"  > Solved at Step {step}! Loss: {loss_start:.4f}")
                break

            candidates = []
            # Recent-top collapse indicator (used for candidate scoring penalties).
            if len(recent_action_counts) >= 50:
                rc = Counter(recent_action_counts)
                top_node, top_count = rc.most_common(1)[0]
                top_frac = top_count / len(recent_action_counts)
            else:
                top_node, top_frac = None, 0.0

            # Parent-balance: for multi-parent children, encourage intervening on under-sampled parents.
            parent_balance = {}
            try:
                for child in current_student.nodes:
                    parents = list(M_star.graph.predecessors(child))
                    if len(parents) < 2:
                        continue
                    # Only bias towards children that are currently hard.
                    child_loss = float(node_losses_start.get(child, 0.0))
                    if child_loss <= 0:
                        continue
                    desired = 1.0 / len(parents)
                    counts = Counter([a for a in recent_action_counts if a in parents])
                    denom_ct = float(sum(counts.values())) + 1e-8
                    for p in parents:
                        frac = float(counts.get(p, 0.0)) / denom_ct if denom_ct > 0 else 0.0
                        deficit = max(0.0, desired - frac)
                        parent_balance[p] = parent_balance.get(p, 0.0) + deficit * child_loss
            except Exception:
                parent_balance = {}

            step_any_valid = False
            step_invalid_samples = []  # Track failed parses for this step
            for k in range(args.candidates):
                cmd_str, plan = policy_net.generate_experiment(current_student, node_losses=node_losses_start)
                train_candidates_total += 1
                if plan is None:
                    train_candidates_invalid += 1
                    step_invalid_samples.append(cmd_str)
                    recent_parse_failures.append(cmd_str)
                    candidates.append((cmd_str, -10.0, 0.0, -10.0, None))
                else:
                    train_candidates_parsed += 1
                    step_any_valid = True
                    student_clone = copy.deepcopy(current_student)
                    # Clone learner with the same replay buffer to make candidate scoring realistic.
                    initial_buffer = []
                    for b in learner.buffer:
                        # b is {"data": ..., "intervened": ...}
                        b_data = {kk: vv.detach().clone() for kk, vv in b["data"].items()}
                        initial_buffer.append({"data": b_data, "intervened": b.get("intervened")})
                    clone_learner = SCMLearner(
                        student_clone,
                        lr=args.learner_lr,
                        buffer_steps=args.buffer_steps,
                        initial_buffer=initial_buffer,
                    )
                    batch_t = executor.run_experiment(plan)
                    clone_learner.train_step(batch_t, n_epochs=args.learner_epochs)
                    loss_end, node_losses_end = critic.evaluate_mechanisms_detailed(student_clone)
                    reward = critic.calculate_reward(loss_start, loss_end)
                    tgt = plan.get("target")
                    # Prefer interventions that help high-loss *direct children* (best for learning X1/X2 -> X3).
                    # normalize=True ensures we value "Urgency" (Avg Loss) over "Volume" (Total Loss)
                    node_weight = _direct_child_impact_weight(M_star.graph, tgt, node_losses_start, normalize=True)
                    denom = float(sum(node_losses_start.values())) + 1e-8
                    norm_weight = node_weight / denom
                    under_sample = 1.0 / np.sqrt(1.0 + episode_action_counts.get(tgt, 0))
                    cov_bonus = args.cov_bonus * norm_weight * under_sample
                    # Value novelty: prefer values that expand coverage for the same target.
                    v = float(plan.get("value", 0.0))
                    recent_vals = list(recent_values_by_target.get(tgt, []))
                    if len(recent_vals) >= 10:
                        mu = float(np.mean(recent_vals))
                        sd = float(np.std(recent_vals)) + 1e-3
                        z = abs(v - mu) / sd
                        val_bonus = args.val_bonus * float(np.clip(z, 0.0, 3.0))
                    else:
                        val_bonus = 0.0

                    # Value coverage: bonus for exploring under-sampled bins for this target.
                    n_bins = max(2, int(args.n_value_bins))
                    bidx = _bin_index(v, args.value_min, args.value_max, n_bins)
                    bin_hist = list(recent_value_bins_by_target.get(tgt, []))
                    bin_ct = bin_hist.count(bidx) if bin_hist is not None else 0
                    bin_bonus = float(args.bin_bonus) / np.sqrt(1.0 + float(bin_ct))

                    # Parent balance bonus (esp. X1 vs X2 to identify X3's multi-parent mechanism).
                    bal_bonus = float(args.parent_balance_bonus) * float(parent_balance.get(tgt, 0.0)) / (denom + 1e-8)

                    # Under-sampling bonus: strong incentive for severely neglected nodes
                    # If a node has been sampled much less than expected, boost it significantly
                    undersample_bonus = 0.0
                    if len(recent_action_counts) >= 20:
                        expected_frac = 1.0 / len(dsl.nodes)
                        actual_count = sum(1 for a in recent_action_counts if a == tgt)
                        actual_frac = actual_count / len(recent_action_counts)
                        deficit = expected_frac - actual_frac
                        if deficit > 0.05:  # More than 5% below expected
                            # Strong exponential bonus for severely under-sampled nodes
                            undersample_bonus = float(args.undersampled_bonus) * (deficit ** 1.5) * 100.0

                    # Disentanglement bonus (Triangle Breaking)
                    # This specifically addresses the X1->X2->X3 structure where we fail to learn X3
                    # because X1 and X2 are collinear in observational (and DO(X1)) data.
                    disent_bonus = _disentanglement_bonus(M_star.graph, tgt, node_losses_start)

                    # Penalize intervening on leaves (no descendants) to focus on informative actions.
                    leaf = False
                    try:
                        leaf = len(list(M_star.graph.successors(tgt))) == 0
                    except Exception:
                        leaf = False
                    leaf_pen = float(args.leaf_penalty) if leaf else 0.0

                    # Penalize collapse if we're repeating the recent-top node too much.
                    # Enhanced: exponential penalty for severe collapse
                    collapse_pen = 0.0
                    if top_node is not None and tgt == top_node and top_frac > float(args.collapse_threshold):
                        excess = float(top_frac - float(args.collapse_threshold))
                        # Quadratic penalty: becomes very severe as collapse worsens
                        collapse_pen = float(args.collapse_penalty) * (excess ** 2) * 100.0

                    score = reward + cov_bonus + val_bonus + bin_bonus + bal_bonus + disent_bonus + undersample_bonus - leaf_pen - collapse_pen
                    candidates.append((cmd_str, reward, cov_bonus, score, plan))
                    
                    # NEW: Detailed diagnostic logging every 50 steps for first 3 candidates
                    if episode % 10 == 0 and step % 50 == 0 and k < 3:
                        logging.info(
                            f"    [Bonus Detail] Candidate {k+1}: target={tgt}, "
                            f"reward={reward:.2f}, cov={cov_bonus:.2f}, val={val_bonus:.2f}, "
                            f"bin={bin_bonus:.2f}, bal={bal_bonus:.2f}, disent={disent_bonus:.2f}, "
                            f"undersample={undersample_bonus:.2f}, leaf_pen={leaf_pen:.2f}, "
                            f"collapse_pen={collapse_pen:.2f}, score={score:.2f}"
                        )

            if step_any_valid:
                train_steps_with_any_valid += 1
            
            # Log parse failures for diagnosis (every 20 steps or when all fail)
            if not step_any_valid or (episode % 10 == 0 and step % 20 == 0 and step_invalid_samples):
                parse_rate = train_candidates_parsed / max(train_candidates_total, 1)
                if step_invalid_samples:
                    sample_fail = step_invalid_samples[0][:150]
                    logging.info(
                        f"  [Parse Stats] Episode {episode}, Step {step}: "
                        f"Parsed {train_candidates_parsed}/{train_candidates_total} ({parse_rate:.1%}), "
                        f"Sample fail: '{sample_fail}'"
                    )

            valid_cmds = [c for c, r, cb, s, p in candidates if r > -9.0]
            if not valid_cmds:
                train_steps_teacher_fallback += 1
                teacher_cmd = get_teacher_command_impact(
                    dsl.nodes,
                    M_star.graph,
                    node_losses_start,
                    value_min=args.value_min,
                    value_max=args.value_max,
                )
                teacher_plan = dsl.parse_to_dict(teacher_cmd)
                tgt = teacher_plan.get("target") if teacher_plan else None
                node_weight = _direct_child_impact_weight(M_star.graph, tgt, node_losses_start, normalize=True)
                denom = float(sum(node_losses_start.values())) + 1e-8
                norm_weight = node_weight / denom
                under_sample = 1.0 / np.sqrt(1.0 + episode_action_counts.get(tgt, 0))
                cov_bonus = args.cov_bonus * norm_weight * under_sample
                score = 0.1 + cov_bonus
                candidates.append((teacher_cmd, 0.1, cov_bonus, score, teacher_plan))

            # Rank by score (reward + coverage bonus).
            sorted_cands = sorted(candidates, key=lambda x: x[3], reverse=True)
            
            # --- CRITICAL FIX: COLLAPSE BREAKER ---
            # If we are collapsed (e.g. > 50% on one node) and the agent ONLY proposes the collapsed node,
            # we must forcefully inject a random alternative. Otherwise, the agent is forced to pick
            # the collapsed node despite the massive penalty, perpetuating the loop.
            if top_node is not None and top_frac > 0.50:
                # Check if we have any valid candidate that is NOT the top_node
                has_alternative = any(c[4] is not None and c[4].get("target") != top_node for c in sorted_cands if c[1] > -9.0)
                
                if not has_alternative:
                    logging.info(f"  [Collapse Breaker] All candidates target {top_node}@{top_frac:.0%}. Injecting random alternative.")
                    
                    # 1. Find a valid node that is NOT the top_node
                    valid_others = [n for n in dsl.nodes if n != top_node]
                    if valid_others:
                        # 2. Generate a random command for it
                        breaker_cmd = get_random_valid_command_range(valid_others, args.value_min, args.value_max)
                        breaker_plan = dsl.parse_to_dict(breaker_cmd)
                        
                        # 3. Score it (It will have NO collapse penalty, so it should win easily)
                        tgt = breaker_plan.get("target")
                        node_weight = _direct_child_impact_weight(M_star.graph, tgt, node_losses_start, normalize=True)
                        denom = float(sum(node_losses_start.values())) + 1e-8
                        norm_weight = node_weight / denom
                        under_sample = 1.0 / np.sqrt(1.0 + episode_action_counts.get(tgt, 0))
                        cov_bonus = args.cov_bonus * norm_weight * under_sample
                        
                        # Give it a positive score to ensure it beats the penalized candidates (which are < 0)
                        score = 10.0 + cov_bonus 
                        
                        # Insert at the top
                        sorted_cands.insert(0, (breaker_cmd, 0.0, cov_bonus, score, breaker_plan))
                        logging.info(f"  [Collapse Breaker] Injected '{breaker_cmd}' (Score: {score:.2f})")

            # MANDATORY DIVERSITY CONSTRAINT: When severe collapse detected, filter out over-sampled node
            diversity_enforced = False
            if args.diversity_constraint and top_node is not None and top_frac > args.diversity_threshold:
                # Remove candidates targeting the over-sampled node
                diverse_cands = [c for c in sorted_cands if c[4] is not None and c[4].get("target") != top_node]
                if diverse_cands:
                    logging.info(f"  [Diversity Constraint] Collapse detected ({top_node}@{top_frac:.0%}), forcing alternative targets")
                    sorted_cands = diverse_cands
                    diversity_enforced = True
            
            # Additional forced diversity: Every 10 steps, force exploration of under-sampled nodes
            # This helps even without explicit diversity_constraint flag
            if step > 0 and step % 10 == 0 and top_node is not None and top_frac > 0.50 and not diversity_enforced:
                # Find the least sampled node that's not the top node
                node_counts = Counter(recent_action_counts)
                least_sampled = min([n for n in dsl.nodes if n != top_node], 
                                   key=lambda n: node_counts.get(n, 0), default=None)
                if least_sampled:
                    # Prefer candidates targeting the least sampled node
                    diverse_cands = [c for c in sorted_cands if c[4] is not None and c[4].get("target") == least_sampled]
                    if diverse_cands:
                        logging.info(f"  [Forced Diversity] Step {step}: Targeting under-sampled node {least_sampled} (vs {top_node}@{top_frac:.0%})")
                        sorted_cands = diverse_cands
            
            if random.random() < args.eps_explore:
                explore_pool = [c for c in candidates if c[1] > -9.0]
                explore_sorted = sorted(explore_pool, key=lambda x: x[2], reverse=True)
                winner_cmd, winner_reward, winner_cov_bonus, winner_score, winner_plan = explore_sorted[0]
            else:
                winner_cmd, winner_reward, winner_cov_bonus, winner_score, winner_plan = sorted_cands[0]
            
            # --- STRATEGY REVISION: CONTRASTIVE DPO PAIRS (EPISTEMIC CURIOSITY) ---
            # Instead of always picking the worst candidate as loser, we intelligently select
            # a loser that maximizes the "Strategy Contrast" (specifically tackling collapse).
            loser_idx = -1
            curiosity_weight = 1.0
            
            # Identify the dominant (collapsed) node
            dom_node = top_node if (top_node is not None and top_frac > 0.40) else None
            
            winner_tgt = winner_plan.get("target") if winner_plan else None
            
            if dom_node and winner_tgt and winner_tgt != dom_node:
                # Case 1: Winner is a "Novel" node (not the collapsed one).
                # We want to explicitly pair it against the Collapsed Node to teach "Novel > Collapsed".
                # We search for the *best* available candidate targeting the collapsed node to serve as the loser.
                # (Comparing against the best X1 makes the gradient signal stronger/more robust than comparing vs worst X1).
                for i in range(1, len(sorted_cands)):
                    cand = sorted_cands[i]
                    if cand[4] and cand[4].get("target") == dom_node:
                        loser_idx = i
                        # Boost the update weight because this is a high-value "Curiosity" lesson
                        curiosity_weight = 2.0 
                        break
            
            if curiosity_weight > 1.0 and step % 10 == 0:
                logging.info(f"  [Epistemic Boost] Training '{winner_cmd}' > '{sorted_cands[loser_idx][0]}' (Weight: {curiosity_weight})")

            loser_cmd, loser_reward, loser_cov_bonus, loser_score, _ = sorted_cands[loser_idx]

            # Update
            if winner_score > loser_score:
                optimizer_agent.zero_grad()
                if use_pretrained:
                    loss = dpo_loss_llm(policy_net, ref_policy, current_student, winner_cmd, loser_cmd, node_losses=node_losses_start)
                else:
                    # Tensor path for custom
                    win_seq = dsl.encode(winner_cmd).unsqueeze(0).to(device)
                    lose_seq = dsl.encode(loser_cmd).unsqueeze(0).to(device)
                    max_len = max(win_seq.shape[1], lose_seq.shape[1])
                    win_pad = F.pad(win_seq, (0, max_len - win_seq.shape[1]), value=dsl.token2id["<PAD>"])
                    lose_pad = F.pad(lose_seq, (0, max_len - lose_seq.shape[1]), value=dsl.token2id["<PAD>"])
                    loss = dpo_loss(policy_net, ref_policy, current_student, win_pad, lose_pad, node_losses=node_losses_start)
                
                # Apply Curiosity Boost (Epistemic Incentive)
                loss = loss * curiosity_weight
                    
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
                try:
                    recent_values_by_target[tgt].append(float(winner_plan.get("value")))
                except Exception:
                    pass
                try:
                    n_bins = max(2, int(args.n_value_bins))
                    bidx = _bin_index(float(winner_plan.get("value")), args.value_min, args.value_max, n_bins)
                    recent_value_bins_by_target[tgt].append(bidx)
                except Exception:
                    pass
                episode_action_counts[tgt] += 1
                recent_action_counts.append(tgt)
                
                # NEW: Track intervention coverage for collider parent analysis
                for collider in collider_nodes:
                    parents = list(M_star.graph.predecessors(collider))
                    if tgt in parents:
                        parent_intervention_counts[collider][tgt] += 1
                
                # NEW: Log intervention coverage for colliders periodically
                if step > 0 and step % 25 == 0:
                    for collider in collider_nodes:
                        parents = list(M_star.graph.predecessors(collider))
                        total_interventions = sum(parent_intervention_counts[collider].values())
                        if total_interventions > 0:
                            coverage_record = {
                                "episode": episode,
                                "step": step,
                                "collider": collider,
                                "total_parent_interventions": total_interventions,
                            }
                            for p in parents:
                                count = parent_intervention_counts[collider][p]
                                coverage_record[f"interventions_{p}"] = count
                                coverage_record[f"fraction_{p}"] = count / total_interventions if total_interventions > 0 else 0.0
                            coverage_record["balance_score"] = min(
                                [parent_intervention_counts[collider][p] for p in parents]
                            ) / max([parent_intervention_counts[collider][p] for p in parents], default=1)
                            intervention_coverage_tracking.append(coverage_record)

                if winner_score > 0.2 or (episode % 10 == 0 and step == 0):
                    logging.info(
                        f"  Step {step}: '{winner_cmd}' (Reward: {winner_reward:.2f}, "
                        f"Cov: {winner_cov_bonus:.2f}, Score: {winner_score:.2f}, "
                        f"RecentTop: {top_node}@{top_frac:.0%})"
                    )
                real_data = executor.run_experiment(winner_plan)
                learner.train_step(real_data, n_epochs=args.learner_epochs)

    # 4. Final Evaluation
    logging.info("--- Running Final Evaluation ---")
    eval_targets = []
    eval_values = []
    
    # Use the LAST student state for evaluation visualization
    visualize_contrast_save(M_star, current_student, run_dir)
    
    n_eval = 100
    n_parsed = 0
    for _ in range(n_eval):
        # Pass empty node losses for evaluation if not available, or last known
        # Ideally we should eval mechanisms first, but for speed we might skip or use last known.
        # Let's re-evaluate mechanisms to be safe and accurate.
        _, eval_losses = critic.evaluate_mechanisms_detailed(current_student)
        _, plan = policy_net.generate_experiment(current_student, node_losses=eval_losses)
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
        "run_started_at": [run_started_at.isoformat(timespec="seconds")] * len(loss_history),
        "run_dir": [run_dir] * len(loss_history),
        "train_candidates_total": [train_candidates_total] * len(loss_history),
        "train_candidates_parsed": [train_candidates_parsed] * len(loss_history),
        "train_candidates_invalid": [train_candidates_invalid] * len(loss_history),
        "train_candidate_parse_rate": [(train_candidates_parsed / max(train_candidates_total, 1))] * len(loss_history),
        "train_steps_total": [train_steps_total] * len(loss_history),
        "train_steps_with_any_valid_candidate": [train_steps_with_any_valid] * len(loss_history),
        "train_steps_teacher_fallback": [train_steps_teacher_fallback] * len(loss_history),
        "train_teacher_fallback_rate": [(train_steps_teacher_fallback / max(train_steps_total, 1))] * len(loss_history),
    })
    
    # NEW: Calculate and log collapse statistics
    if target_history:
        target_counts = Counter([t for t in target_history if t is not None])
        total_targets = sum(target_counts.values())
        if total_targets > 0:
            logging.info("--- Final Target Distribution (Collapse Analysis) ---")
            for node in dsl.nodes:
                count = target_counts.get(node, 0)
                pct = 100.0 * count / total_targets
                logging.info(f"  {node}: {count}/{total_targets} ({pct:.1f}%)")
            
            # Identify if X1 collapse still exists (bias check)
            x1_pct = 100.0 * target_counts.get("X1", 0) / total_targets
            if x1_pct > 40.0:
                logging.warning(f"  [WARNING] High X1 concentration detected ({x1_pct:.1f}%). Normalization may need tuning.")
            else:
                logging.info(f"  [SUCCESS] X1 concentration ({x1_pct:.1f}%) is balanced (<40%).")

    df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    
    # NEW: Save detailed per-node loss tracking for collider diagnostics
    if node_loss_tracking:
        node_loss_df = pd.DataFrame(node_loss_tracking)
        node_loss_df.to_csv(os.path.join(run_dir, "node_losses.csv"), index=False)
        logging.info(f"Saved per-node loss tracking: {len(node_loss_tracking)} records")
    
    # NEW: Save intervention coverage analysis for collider diagnostics
    if intervention_coverage_tracking:
        coverage_df = pd.DataFrame(intervention_coverage_tracking)
        coverage_df.to_csv(os.path.join(run_dir, "intervention_coverage.csv"), index=False)
        logging.info(f"Saved intervention coverage tracking: {len(intervention_coverage_tracking)} records")
        
        # Log final coverage statistics for colliders
        for collider in collider_nodes:
            parents = list(M_star.graph.predecessors(collider))
            total = sum(parent_intervention_counts[collider].values())
            if total > 0:
                logging.info(f"Final intervention coverage for collider {collider}:")
                for p in parents:
                    count = parent_intervention_counts[collider][p]
                    pct = 100.0 * count / total
                    logging.info(f"  {p}: {count}/{total} ({pct:.1f}%)")
    
    run_ended_at = datetime.now()
    logging.info(f"Run ended at: {run_ended_at.isoformat(timespec='seconds')}")
    logging.info(f"Run duration: {str(run_ended_at - run_started_at)}")
    logging.info(
        "Training parse stats: "
        f"candidates_parsed={train_candidates_parsed}/{train_candidates_total} "
        f"({(train_candidates_parsed / max(train_candidates_total, 1)):.1%}), "
        f"steps_with_any_valid_candidate={train_steps_with_any_valid}/{train_steps_total} "
        f"({(train_steps_with_any_valid / max(train_steps_total, 1)):.1%}), "
        f"teacher_fallback_steps={train_steps_teacher_fallback}/{train_steps_total} "
        f"({(train_steps_teacher_fallback / max(train_steps_total, 1)):.1%})"
    )
    logging.info(f"Experiment Complete. Results saved to {run_dir}")

if __name__ == "__main__":
    main()
