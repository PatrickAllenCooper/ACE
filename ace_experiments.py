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
import signal
import atexit
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

    def generate_experiment(self, scm_student, node_losses=None, intervention_history=None, max_len=5):
        """Generate intervention. intervention_history accepted for API compatibility but not used."""
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
        
        # Track generation statistics for diagnostics
        self.generation_stats = Counter()
        self.prompt_response_log = deque(maxlen=50)  # Keep last 50 for diagnosis
            
    def scm_to_prompt(self, scm, node_losses=None, intervention_history=None):
        """
        CRITICAL FIX: Completely restructured prompt to force the LLM to attend to losses.
        
        The old prompt buried the loss information and ended with "DO" which biased 
        the LLM toward always completing with "X1" (alphabetically/numerically first).
        
        New prompt:
        1. Puts the PROBLEM first (which node is failing)
        2. Explicitly states what NOT to do (avoid over-sampled nodes)
        3. Provides reasoning examples
        4. Ends with the target node decision point, not "DO"
        """
        edges = [f"{u}->{v}" for u, v in scm.graph.edges]
        edge_str = ", ".join(edges)
        
        # Identify the failing node (highest loss, excluding roots)
        failing_node = None
        max_loss = 0.0
        parent_nodes = set()
        if node_losses:
            for node in scm.nodes:
                parents = list(scm.graph.predecessors(node))
                if parents:  # Not a root
                    loss = node_losses.get(node, 0.0)
                    if loss > max_loss:
                        max_loss = loss
                        failing_node = node
                        parent_nodes = set(parents)
        
        # Build intervention history summary
        hist_str = ""
        if intervention_history:
            hist_counts = Counter(intervention_history[-100:])  # Last 100
            total = sum(hist_counts.values())
            if total > 0:
                hist_parts = [f"{n}:{hist_counts.get(n,0)}/{total}" for n in scm.nodes]
                hist_str = f"Recent interventions: [{', '.join(hist_parts)}]. "
                # Identify over-sampled node
                most_common = hist_counts.most_common(1)
                if most_common and most_common[0][1] / total > 0.3:
                    oversampled = most_common[0][0]
                    hist_str += f"WARNING: {oversampled} is over-sampled ({most_common[0][1]}/{total}). AVOID {oversampled}. "
        
        # Build loss ranking (most important information)
        loss_ranking = ""
        if node_losses:
            # Sort nodes by loss, descending
            sorted_losses = sorted(
                [(n, node_losses.get(n, 0.0)) for n in scm.nodes if list(scm.graph.predecessors(n))],
                key=lambda x: x[1],
                reverse=True
            )
            if sorted_losses:
                loss_parts = [f"{n}={v:.2f}" for n, v in sorted_losses[:3]]  # Top 3
                loss_ranking = f"PROBLEM: Node losses (high=bad): {', '.join(loss_parts)}. "
                if failing_node and parent_nodes:
                    loss_ranking += f"To fix {failing_node}, intervene on its parents: {', '.join(sorted(parent_nodes))}. "
        
        # Construct the NEW prompt - problem-first, action-oriented
        valid_nodes = ", ".join(scm.nodes)
        
        # FEW-SHOT EXAMPLES that demonstrate reasoning
        examples = """
Examples of good reasoning:
- If X3 has high loss and parents are X1,X2: "X3 is failing. To learn X3's mechanism, I should intervene on X2 (breaking X1-X2 correlation)." -> DO X2 = 1.5
- If X2 has high loss and parent is X1: "X2 is failing. To learn X2's mechanism, I should intervene on X1." -> DO X1 = -2.0
- If X5 has high loss and parent is X4: "X5 is failing. To learn X5's mechanism, I should intervene on X4." -> DO X4 = 3.0
"""

        prompt = (
            f"{loss_ranking}"
            f"{hist_str}"
            f"Graph: [{edge_str}]. "
            f"Valid targets: {valid_nodes}. Value range: [{self.dsl.value_min}, {self.dsl.value_max}].\n"
            f"{examples}\n"
            f"Based on the current losses, which node should we intervene on to learn the failing mechanism?\n"
            f"Reasoning: The highest loss node is"
        )
        
        return prompt, failing_node, parent_nodes

    def forward(self, scm_student, target_text_list=None, node_losses=None, intervention_history=None):
        prompt_text, _, _ = self.scm_to_prompt(scm_student, node_losses=node_losses, intervention_history=intervention_history)
        if target_text_list is None: target_text_list = [""]
        full_texts = [prompt_text + " " + t for t in target_text_list]
        inputs = self.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model(**inputs)
        return outputs.logits, inputs.input_ids

    def generate_experiment(self, scm_student, node_losses=None, intervention_history=None, max_new_tokens=64):
        prompt_text, failing_node, parent_nodes = self.scm_to_prompt(
            scm_student, node_losses=node_losses, intervention_history=intervention_history
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=self.tokenizer.eos_token_id, 
                do_sample=True, 
                temperature=0.8,  # Slightly higher for more diversity
                top_p=0.9,
                repetition_penalty=1.3
            )
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated portion (after the prompt)
        if full_text.startswith(prompt_text):
            generated_text = full_text[len(prompt_text):].strip()
        else:
            # Try to find where generation started
            if "Reasoning:" in full_text:
                generated_text = full_text.split("Reasoning:")[-1].strip()
            else:
                generated_text = full_text
        
        try:
            # Try parsing the generated text for DO command
            plan = self.dsl.parse_to_dict_lenient(generated_text, clip_out_of_range=True)
            
            # Fallback: try parsing the full text if generated_text didn't work
            if plan is None and generated_text != full_text:
                plan = self.dsl.parse_to_dict_lenient(full_text, clip_out_of_range=True)
            
            if plan is None:
                # CRITICAL FIX: If LLM fails to produce valid command, use informed fallback
                # Instead of returning None, generate a smart intervention based on losses
                if failing_node and parent_nodes:
                    # Target a parent of the failing node
                    target = random.choice(list(parent_nodes))
                    value = random.uniform(float(self.dsl.value_min), float(self.dsl.value_max))
                    plan = {"target": target, "value": value, "samples": 200}
                    cmd_str = f"DO {target} = {value:.4f}"
                    logging.debug(f"LLM_FALLBACK: Generated '{cmd_str}' (parent of failing {failing_node})")
                    # Track the fallback
                    self.generation_stats["fallback_parent"] += 1
                    return cmd_str, plan
                else:
                    logging.debug(f"PARSE_FAIL: generated='{generated_text[:150]}', full='{full_text[:150]}'")
                    self.generation_stats["parse_fail"] += 1
                    return generated_text if generated_text else full_text, None

            # Track successful generation
            target = plan.get("target")
            self.generation_stats[f"generated_{target}"] += 1
            
            # Log prompt-response for diagnosis (periodically)
            if len(self.prompt_response_log) < 50 or random.random() < 0.01:
                self.prompt_response_log.append({
                    "prompt_snippet": prompt_text[-200:],
                    "generated": generated_text[:100],
                    "target": target,
                    "failing_node": failing_node,
                    "parent_nodes": list(parent_nodes) if parent_nodes else []
                })

            # Canonicalize the command string
            cmd_str = f"DO {plan['target']} = {float(plan['value']):.4f}"
            return cmd_str, plan
            
        except Exception as e:
            logging.debug(f"PARSE_EXCEPTION: {e}, generated='{generated_text[:150] if generated_text else None}'")
            self.generation_stats["exception"] += 1
            return generated_text if generated_text else full_text, None
    
    def log_generation_diagnostics(self):
        """Log accumulated generation statistics for diagnosis."""
        if self.generation_stats:
            total = sum(self.generation_stats.values())
            logging.info(f"LLM Generation Stats (n={total}):")
            for key, count in self.generation_stats.most_common():
                pct = 100.0 * count / total if total > 0 else 0
                logging.info(f"  {key}: {count} ({pct:.1f}%)")
        
        # Log sample prompt-response pairs
        if self.prompt_response_log:
            logging.info(f"Sample prompt-response pairs (last {len(self.prompt_response_log)}):")
            for i, entry in enumerate(list(self.prompt_response_log)[-3:]):
                logging.info(f"  [{i}] failing={entry['failing_node']}, parents={entry['parent_nodes']}")
                logging.info(f"      generated: '{entry['generated'][:80]}'")
                logging.info(f"      target: {entry['target']}")

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

class DPOLogger:
    """
    Comprehensive logging for DPO training diagnostics.
    
    Tracks:
    - Loss values and components
    - Preference margins (how much policy prefers winner over loser)
    - Log probability differences
    - Training stability indicators
    """
    def __init__(self):
        self.history = {
            "loss": [],
            "preference_margin": [],  # policy_win_lp - policy_lose_lp
            "ref_margin": [],          # ref_win_lp - ref_lose_lp
            "kl_from_ref": [],         # How far policy has drifted from ref
            "winner_target": [],
            "loser_target": [],
            "sigmoid_input": [],       # The logit input to sigmoid (should be positive for learning)
        }
        self.step_count = 0
        
    def log(self, loss, policy_win_lp, policy_lose_lp, ref_win_lp, ref_lose_lp, 
            winner_target, loser_target, logit):
        self.history["loss"].append(loss)
        self.history["preference_margin"].append((policy_win_lp - policy_lose_lp).item())
        self.history["ref_margin"].append((ref_win_lp - ref_lose_lp).item())
        self.history["kl_from_ref"].append(((policy_win_lp - ref_win_lp) + (policy_lose_lp - ref_lose_lp)).item() / 2)
        self.history["winner_target"].append(winner_target)
        self.history["loser_target"].append(loser_target)
        self.history["sigmoid_input"].append(logit.item())
        self.step_count += 1
        
    def report(self, every_n=100):
        """Generate periodic report on DPO training health."""
        if self.step_count == 0 or self.step_count % every_n != 0:
            return
            
        recent = min(100, len(self.history["loss"]))
        
        avg_loss = np.mean(self.history["loss"][-recent:])
        avg_pref_margin = np.mean(self.history["preference_margin"][-recent:])
        avg_sigmoid_input = np.mean(self.history["sigmoid_input"][-recent:])
        avg_kl = np.mean(self.history["kl_from_ref"][-recent:])
        
        # Count how often policy prefers winner (healthy = positive preference margin)
        positive_prefs = sum(1 for m in self.history["preference_margin"][-recent:] if m > 0)
        pref_rate = positive_prefs / recent if recent > 0 else 0
        
        # Count winner/loser target distribution
        winner_counts = Counter(self.history["winner_target"][-recent:])
        loser_counts = Counter(self.history["loser_target"][-recent:])
        
        logging.info(f"--- DPO Training Report (step {self.step_count}) ---")
        logging.info(f"  Avg Loss: {avg_loss:.4f} (optimal ~0, stuck at 0.693 = random)")
        logging.info(f"  Avg Preference Margin: {avg_pref_margin:.4f} (should be positive)")
        logging.info(f"  Policy Prefers Winner: {pref_rate:.1%} of time (should be >50%)")
        logging.info(f"  Avg Sigmoid Input: {avg_sigmoid_input:.4f} (positive = learning)")
        logging.info(f"  Avg KL from Reference: {avg_kl:.4f}")
        logging.info(f"  Recent Winners: {dict(winner_counts)}")
        logging.info(f"  Recent Losers: {dict(loser_counts)}")
        
        # Health check warnings
        if avg_loss > 0.68:
            logging.warning(f"  ⚠️ DPO loss near random chance (0.693) - model may not be learning!")
        if pref_rate < 0.4:
            logging.warning(f"  ⚠️ Policy rarely prefers winner - preference signal may be inverted!")
        if abs(avg_kl) > 5.0:
            logging.warning(f"  ⚠️ Large KL divergence from reference - consider updating reference policy")
            
    def save(self, results_dir):
        """Save DPO training history to CSV for analysis."""
        if not self.history["loss"]:
            return
        df = pd.DataFrame({
            "loss": self.history["loss"],
            "preference_margin": self.history["preference_margin"],
            "ref_margin": self.history["ref_margin"],
            "kl_from_ref": self.history["kl_from_ref"],
            "sigmoid_input": self.history["sigmoid_input"],
            "winner_target": self.history["winner_target"],
            "loser_target": self.history["loser_target"],
        })
        path = os.path.join(results_dir, "dpo_training.csv")
        df.to_csv(path, index=False)
        logging.info(f"Saved DPO training history to {path}")


# Global DPO logger instance
_dpo_logger = None

def get_dpo_logger():
    global _dpo_logger
    if _dpo_logger is None:
        _dpo_logger = DPOLogger()
    return _dpo_logger


def dpo_loss_llm(policy_model, ref_model, scm_state, win_text, lose_text, node_losses=None, intervention_history=None, beta=0.1):
    """
    LLM DPO Loss with comprehensive logging.
    
    The loss encourages the policy to prefer win_text over lose_text:
    L = -log(sigmoid(beta * [(policy_win - ref_win) - (policy_lose - ref_lose)]))
    
    Key insight: If the sigmoid input is negative, the policy prefers the loser.
    We need positive sigmoid inputs for the model to learn the preference.
    """
    def get_log_probs(model, text, losses, hist):
        logits, input_ids = model(scm_state, [text], node_losses=losses, intervention_history=hist)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum(dim=-1)

    policy_win_lp = get_log_probs(policy_model, win_text, node_losses, intervention_history)
    policy_lose_lp = get_log_probs(policy_model, lose_text, node_losses, intervention_history)
    with torch.no_grad():
        ref_win_lp = get_log_probs(ref_model, win_text, node_losses, intervention_history)
        ref_lose_lp = get_log_probs(ref_model, lose_text, node_losses, intervention_history)

    # The DPO objective
    logits = (policy_win_lp - ref_win_lp) - (policy_lose_lp - ref_lose_lp)
    loss = -F.logsigmoid(beta * logits).mean()
    
    # Comprehensive logging
    logger = get_dpo_logger()
    
    # Extract targets from command strings for logging
    win_target = None
    lose_target = None
    try:
        import re
        win_match = re.search(r'DO\s+(X\d+)', win_text)
        lose_match = re.search(r'DO\s+(X\d+)', lose_text)
        win_target = win_match.group(1) if win_match else "?"
        lose_target = lose_match.group(1) if lose_match else "?"
    except:
        pass
    
    logger.log(
        loss=loss.item(),
        policy_win_lp=policy_win_lp,
        policy_lose_lp=policy_lose_lp,
        ref_win_lp=ref_win_lp,
        ref_lose_lp=ref_lose_lp,
        winner_target=win_target,
        loser_target=lose_target,
        logit=logits
    )
    
    # Periodic reporting
    logger.report(every_n=100)
    
    return loss


def supervised_pretrain_llm(policy_model, scm, graph, nodes, node_losses, optimizer, n_steps=50, value_min=-5.0, value_max=5.0):
    """
    CRITICAL FIX: Pre-train the LLM policy on teacher-generated interventions.
    
    This addresses the core problem: the LLM ignores the prompt and always outputs X1.
    By doing supervised fine-tuning on balanced, loss-aware interventions BEFORE DPO,
    we give the model a better starting point.
    """
    logging.info(f"Starting supervised pre-training ({n_steps} steps)...")
    policy_model.train()
    
    total_loss = 0.0
    target_counts = Counter()
    
    for step in range(n_steps):
        # Generate a teacher intervention targeting parents of high-loss nodes
        teacher_cmd = get_teacher_command_impact(
            nodes, graph, node_losses, 
            value_min=value_min, value_max=value_max
        )
        plan = policy_model.dsl.parse_to_dict(teacher_cmd)
        if plan is None:
            continue
            
        target = plan.get("target")
        target_counts[target] += 1
        
        # Create the target text that should follow the prompt
        # The prompt ends with "Reasoning: The highest loss node is"
        # We want the model to continue with the correct reasoning
        
        # Find the failing node for this state
        failing_node = None
        max_loss = 0.0
        for node in nodes:
            parents = list(graph.predecessors(node))
            if parents and node_losses.get(node, 0.0) > max_loss:
                max_loss = node_losses.get(node, 0.0)
                failing_node = node
        
        # Construct the target completion
        if failing_node:
            parents = list(graph.predecessors(failing_node))
            target_text = (
                f" {failing_node} with loss {max_loss:.2f}. "
                f"Its parents are {', '.join(parents)}. "
                f"I should intervene on {target} to help learn {failing_node}'s mechanism. "
                f"DO {target} = {plan['value']:.2f}"
            )
        else:
            target_text = f" unknown. DO {target} = {plan['value']:.2f}"
        
        # Get prompt
        prompt_text, _, _ = policy_model.scm_to_prompt(scm, node_losses=node_losses)
        full_text = prompt_text + target_text
        
        # Compute supervised loss (cross-entropy on target tokens)
        inputs = policy_model.tokenizer(full_text, return_tensors="pt", truncation=True).to(policy_model.device)
        prompt_inputs = policy_model.tokenizer(prompt_text, return_tensors="pt", truncation=True).to(policy_model.device)
        prompt_len = prompt_inputs.input_ids.shape[1]
        
        outputs = policy_model.model(**inputs, labels=inputs.input_ids)
        
        # Mask loss for prompt tokens (only train on completion)
        # This is approximate - ideally we'd use a proper masked loss
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(policy_model.model.parameters(), 1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
        if step % 10 == 0:
            logging.info(f"  Pretrain step {step}: loss={loss.item():.4f}")
    
    avg_loss = total_loss / max(n_steps, 1)
    logging.info(f"Supervised pre-training complete. Avg loss: {avg_loss:.4f}")
    logging.info(f"  Target distribution: {dict(target_counts)}")
    
    return avg_loss

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

def visualize_scm_graph(scm, results_dir, node_losses=None):
    """
    Visualize the SCM graph structure with node types and mechanism equations.
    
    This provides a clear visual reference for understanding the causal structure
    being learned, including:
    - Node positions in causal hierarchy
    - Edge directions (causal relationships)
    - Node types (root, intermediate, collider, leaf)
    - Current loss values (if provided)
    """
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        G = scm.graph
        
        # Compute node properties
        node_types = {}
        for node in G.nodes():
            parents = list(G.predecessors(node))
            children = list(G.successors(node))
            
            if len(parents) == 0:
                node_types[node] = "root"
            elif len(parents) >= 2:
                node_types[node] = "collider"
            elif len(children) == 0:
                node_types[node] = "leaf"
            else:
                node_types[node] = "intermediate"
        
        # Color mapping by node type
        color_map = {
            "root": "#4CAF50",       # Green - exogenous
            "intermediate": "#2196F3", # Blue - single parent
            "collider": "#FF5722",    # Red/Orange - multiple parents (CRITICAL)
            "leaf": "#9C27B0"         # Purple - no children
        }
        node_colors = [color_map[node_types[node]] for node in G.nodes()]
        
        # Use hierarchical layout (topological)
        try:
            # Try graphviz layout if available
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            # Fallback to spring layout with topological hints
            pos = {}
            topo_order = list(nx.topological_sort(G))
            for i, node in enumerate(topo_order):
                # Spread nodes horizontally by topological order
                depth = len(list(nx.ancestors(G, node)))
                pos[node] = (depth * 2, -i * 1.5)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                               arrowsize=25, arrowstyle='->', ax=ax,
                               connectionstyle="arc3,rad=0.1")
        
        # Node labels with loss values if available
        if node_losses:
            labels = {node: f"{node}\nL={node_losses.get(node, 0):.2f}" for node in G.nodes()}
        else:
            labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', ax=ax)
        
        # Add edge labels showing causal direction
        edge_labels = {(u, v): f"{u}→{v}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map["root"], 
                      markersize=15, label='Root (exogenous)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map["intermediate"], 
                      markersize=15, label='Intermediate'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map["collider"], 
                      markersize=15, label='Collider (multi-parent) ⚠️'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map["leaf"], 
                      markersize=15, label='Leaf (no children)'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        # Add title with mechanism equations
        mechanism_text = "Ground Truth Mechanisms:\n"
        mechanism_text += "X1 ~ N(0,1)  [root]\n"
        mechanism_text += "X4 ~ N(2,1)  [root]\n"
        mechanism_text += "X2 = 2.0*X1 + 1.0 + ε\n"
        mechanism_text += "X3 = 0.5*X1 - X2 + sin(X2) + ε  [COLLIDER]\n"
        mechanism_text += "X5 = 0.2*X4² + ε"
        
        ax.text(0.02, 0.02, mechanism_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title("Structural Causal Model (SCM) Graph\n" + 
                    "Edges show causal direction: Parent → Child", fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "scm_graph.png"), dpi=150)
        plt.close()
        logging.info(f"Saved SCM graph visualization to {results_dir}/scm_graph.png")
        
    except Exception as e:
        logging.warning(f"Failed to save SCM graph visualization: {e}")


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
    """
    Improved mechanism contrast visualization with:
    - Proper legends on all plots
    - Clear axis labels
    - MSE annotations
    - Titles explaining what each plot shows
    """
    try:
        relationships = []
        for child in oracle.nodes:
            parents = oracle.get_parents(child)
            if not parents: 
                relationships.append(('Root', child))
            else:
                for parent in parents: 
                    relationships.append((parent, child))
        
        n_plots = len(relationships)
        cols = 3
        rows = (n_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        x_range = torch.linspace(-4, 4, 100)
        
        for i, rel in enumerate(relationships):
            ax = axes[i]
            if rel[0] == 'Root':
                node = rel[1]
                true_data = oracle.generate(1000)[node].detach().numpy()
                with torch.no_grad():
                    pred_data = student.forward(1000)[node].detach().numpy()
                ax.hist(true_data, bins=30, density=True, alpha=0.5, color='black', label='Ground Truth')
                ax.hist(pred_data, bins=30, density=True, alpha=0.5, color='red', label='Student')
                ax.set_xlabel(f'{node} Value')
                ax.set_ylabel('Density')
                ax.set_title(f"Root: {node}")
                ax.legend(loc='upper right', fontsize=8)
            else:
                parent, child = rel
                parents_list = oracle.get_parents(child)
                data_context = {}
                other_parents = []
                for p in parents_list:
                    if p == parent:
                        data_context[p] = x_range
                    else:
                        data_context[p] = torch.zeros(100)
                        other_parents.append(p)
                
                y_true = oracle.mechanisms(data_context, child).detach().numpy()
                with torch.no_grad():
                    if len(parents_list) > 0:
                        p_tensor = torch.stack([data_context[p] for p in parents_list], dim=1)
                        y_pred = student.mechanisms[child](p_tensor).squeeze().detach().numpy()
                    else:
                        y_pred = student.mechanisms[child]['mu'].expand(100).detach().numpy()

                ax.plot(x_range.numpy(), y_true, 'k--', lw=3, label='Ground Truth')
                ax.plot(x_range.numpy(), y_pred, 'r-', lw=2, alpha=0.8, label='Student')
                ax.set_xlabel(parent)
                ax.set_ylabel(child)
                
                # Clear title showing what's held constant
                if other_parents:
                    title = f"{parent} → {child}\n(with {', '.join(other_parents)}=0)"
                else:
                    title = f"{parent} → {child}"
                ax.set_title(title)
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # Add MSE annotation
                mse = np.mean((y_true - y_pred)**2)
                color = '#27ae60' if mse < 0.5 else '#e74c3c'
                ax.annotate(f'MSE: {mse:.3f}', xy=(0.95, 0.05), xycoords='axes fraction',
                           ha='right', va='bottom', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
        
        for j in range(i + 1, len(axes)): 
            axes[j].axis('off')
        
        fig.suptitle('Mechanism Comparison: Ground Truth (black dashed) vs Student (red solid)', 
                     fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "mechanism_contrast.png"), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved mechanism contrast plot to {results_dir}/mechanism_contrast.png")
    except Exception as e:
        logging.error(f"Failed to save contrast plot: {e}")

# ----------------------------------------------------------------
# CHECKPOINT AND SAVE UTILITIES
# ----------------------------------------------------------------
def save_checkpoint(run_dir, episode, policy_net, optimizer, loss_history, 
                   reward_history, recent_actions):
    """Save training checkpoint for recovery.
    
    Checkpoints saved to checkpoints/ directory (separate from results/)
    for easy results copying without large checkpoint files.
    """
    # Save to checkpoints/ directory, not results/
    checkpoint_dir = os.path.join("checkpoints", os.path.basename(run_dir))
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode}.pt")
    
    try:
        torch.save({
            'episode': episode,
            'policy_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history,
            'reward_history': reward_history,
            'recent_actions': list(recent_actions),  # Convert deque to list for saving
        }, checkpoint_path)
        
        logging.info(f"✓ Saved checkpoint to {checkpoint_path}")
        
        # Cleanup old checkpoints (keep only last 3)
        import glob
        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_ep*.pt")))
        if len(checkpoints) > 3:
            for old_checkpoint in checkpoints[:-3]:
                os.remove(old_checkpoint)
                logging.debug(f"  Removed old checkpoint: {os.path.basename(old_checkpoint)}")
                
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")

# ----------------------------------------------------------------
# EMERGENCY SAVE HANDLER
# ----------------------------------------------------------------
def create_emergency_save_handler(run_dir, oracle, history_data):
    """
    Create handler that saves outputs on unexpected termination (SIGTERM/timeout).
    
    SLURM sends SIGTERM ~30 seconds before killing the job, giving us time to save.
    """
    def save_on_exit(signum=None, frame=None):
        try:
            signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT" if signum == signal.SIGINT else "EXIT"
            logging.info("=" * 70)
            logging.info(f"EMERGENCY SAVE: Received {signal_name} signal")
            logging.info("=" * 70)
            
            # Get latest student from reference
            student = history_data.get("student_ref", {}).get("student")
            
            # Save mechanism contrast (most important visualization)
            try:
                if student is not None:
                    visualize_contrast_save(oracle, student, run_dir)
                    logging.info("✓ Saved mechanism_contrast.png")
                else:
                    logging.warning("✗ No student available for mechanism contrast")
            except Exception as e:
                logging.error(f"✗ Failed to save mechanism_contrast: {e}")
            
            # Save partial metrics
            try:
                # Build metrics from history lists
                if history_data.get("loss_history"):
                    df = pd.DataFrame({
                        "dpo_loss": history_data["loss_history"],
                        "reward": history_data["reward_history"],
                        "cov_bonus": history_data.get("cov_bonus_history", []),
                        "score": history_data.get("score_history", []),
                        "target": history_data["target_history"],
                        "value": history_data["value_history"],
                        "episode": history_data.get("episode_history", []),
                        "step": history_data.get("step_history", []),
                    })
                    df.to_csv(os.path.join(run_dir, "metrics_interrupted.csv"), index=False)
                    logging.info(f"✓ Saved {len(df)} records to metrics_interrupted.csv")
            except Exception as e:
                logging.error(f"✗ Failed to save metrics: {e}")
            
            # Save training curves
            try:
                if history_data.get("loss_history") and history_data.get("reward_history"):
                    save_plots(run_dir, 
                              history_data["loss_history"], 
                              history_data["reward_history"],
                              history_data.get("target_history", []),
                              history_data.get("value_history", []),
                              history_data.get("nodes", []))
                    logging.info("✓ Saved training_curves.png and strategy_analysis.png")
            except Exception as e:
                logging.error(f"✗ Failed to save plots: {e}")
            
            logging.info("=" * 70)
            logging.info("Emergency save complete - exiting")
            logging.info("=" * 70)
            
        except Exception as e:
            logging.error(f"Emergency save handler failed: {e}")
        
        # Exit gracefully
        if signum is not None:
            sys.exit(0)
    
    return save_on_exit

# ----------------------------------------------------------------
# EARLY STOPPING AND ROOT FITTING UTILITIES
# ----------------------------------------------------------------
class EarlyStopping:
    """
    Detect training saturation and stop when no improvement is observed.
    Based on analysis showing 89.3% of steps produced zero reward in recent runs.
    
    UPDATE Jan 20: Added min_episodes to prevent stopping too early.
    Initial testing showed episode 8 was too early (X5 hadn't converged).
    
    UPDATE Jan 20 v2: Added per-node convergence checking.
    Test showed global zero-reward check stops before slow learners (X5) converge.
    """
    def __init__(self, patience=20, min_delta=0.01, min_episodes=30, 
                 node_targets=None):
        self.patience = patience
        self.min_delta = min_delta
        self.min_episodes = min_episodes
        self.counter = 0
        self.best_loss = float('inf')
        self.episodes_no_improvement = 0
        
        # Per-node convergence tracking
        self.node_targets = node_targets or {
            'X1': 1.0,   # Roots are hard
            'X2': 0.5,   # Linear should be easy
            'X3': 0.5,   # Collider medium
            'X4': 1.0,   # Root hard
            'X5': 0.5    # Quadratic medium
        }
        self.node_converged_count = {node: 0 for node in self.node_targets}
        
    def check_loss(self, current_loss):
        """Check if we should stop based on loss improvement."""
        if current_loss < self.best_loss - self.min_delta:
            # Improvement detected
            self.best_loss = current_loss
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                logging.info(f"⚠️  Early stopping: No loss improvement for {self.patience} episodes")
                logging.info(f"   Best loss: {self.best_loss:.4f}, Current loss: {current_loss:.4f}")
                return True
        return False
    
    def check_per_node_convergence(self, node_losses, patience=10):
        """
        Check if ALL nodes have converged to their targets.
        More intelligent than zero-reward check - accounts for different node timescales.
        """
        if len(node_losses) == 0:
            return False
            
        # Check each node against its target
        all_converged = True
        converged_nodes = []
        unconverged_nodes = []
        
        for node, loss in node_losses.items():
            target = self.node_targets.get(node, 1.0)
            if loss < target:
                self.node_converged_count[node] += 1
                if self.node_converged_count[node] >= patience:
                    converged_nodes.append(f"{node}:{loss:.3f}")
            else:
                self.node_converged_count[node] = 0
                all_converged = False
                unconverged_nodes.append(f"{node}:{loss:.3f}(target<{target})")
        
        if all_converged and all(count >= patience for count in self.node_converged_count.values()):
            logging.info(f"⚠️  Early stopping: ALL nodes converged for {patience} episodes")
            logging.info(f"   Converged: {', '.join(converged_nodes)}")
            return True
        
        # Log progress every check
        if len(unconverged_nodes) > 0:
            logging.debug(f"  [Convergence] Still training: {', '.join(unconverged_nodes)}")
            
        return False
    
    def check_zero_rewards(self, recent_rewards, threshold=0.85):
        """
        Check if too many recent steps have zero reward (training saturation).
        Threshold: fraction of steps that must have zero reward to trigger stopping.
        
        NOTE: This is a fallback. Per-node convergence is preferred.
        """
        if len(recent_rewards) < 50:  # Need enough samples
            return False
            
        zero_count = sum(1 for r in recent_rewards if abs(r) < 0.01)
        zero_fraction = zero_count / len(recent_rewards)
        
        if zero_fraction > threshold:
            logging.info(f"⚠️  Early stopping: {zero_fraction*100:.1f}% of last {len(recent_rewards)} steps had zero reward")
            logging.info(f"   This indicates training saturation - learner has converged")
            return True
        return False


class DedicatedRootLearner:
    """
    Dedicated learner for root node distributions that ONLY trains on observational data.
    
    Problem: Root nodes don't learn from interventional data because DO(X=v) overrides
    the natural distribution. Mixing interventional and observational training is suboptimal.
    
    Solution: Separate model that only sees observational data, never interventional.
    This model learns ONLY the root distributions X1~N(0,1), X4~N(2,1).
    """
    def __init__(self, root_nodes):
        self.root_nodes = root_nodes
        self.distributions = {}
        for node in root_nodes:
            self.distributions[node] = {
                'mu': nn.Parameter(torch.zeros(1)),
                'log_sigma': nn.Parameter(torch.zeros(1))  # Log for unconstrained optimization
            }
        self.optimizer = optim.Adam([p for d in self.distributions.values() for p in d.values()], lr=0.01)
        
    def fit(self, observational_data, epochs=200):
        """Fit root distributions using MLE on observational data."""
        losses_before = {}
        losses_after = {}
        
        for node in self.root_nodes:
            if node in observational_data:
                samples = observational_data[node]
                true_mu = samples.mean()
                true_sigma = samples.std()
                
                with torch.no_grad():
                    pred_mu = self.distributions[node]['mu'].item()
                    pred_sigma = torch.exp(self.distributions[node]['log_sigma']).item()
                    losses_before[node] = abs(pred_mu - true_mu.item()) + abs(pred_sigma - true_sigma.item())
        
        # Train
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            total_loss = 0.0
            
            for node in self.root_nodes:
                if node in observational_data:
                    samples = observational_data[node]
                    pred_mu = self.distributions[node]['mu']
                    pred_sigma = torch.exp(self.distributions[node]['log_sigma'])
                    
                    # Negative log-likelihood loss
                    loss = 0.5 * torch.mean(((samples - pred_mu) / pred_sigma) ** 2) + torch.log(pred_sigma)
                    total_loss += loss
            
            if total_loss > 0:
                total_loss.backward()
                self.optimizer.step()
        
        # Log results
        for node in self.root_nodes:
            if node in observational_data:
                samples = observational_data[node]
                true_mu = samples.mean()
                true_sigma = samples.std()
                
                with torch.no_grad():
                    pred_mu = self.distributions[node]['mu'].item()
                    pred_sigma = torch.exp(self.distributions[node]['log_sigma']).item()
                    losses_after[node] = abs(pred_mu - true_mu.item()) + abs(pred_sigma - true_sigma.item())
        
        logging.info(f"[Dedicated Root Learner] Before: {losses_before}, After: {losses_after}")
        return losses_after
    
    def apply_to_student(self, student_scm):
        """Copy learned root distributions to student SCM."""
        for node in self.root_nodes:
            if node in student_scm.mechanisms:
                mech = student_scm.mechanisms[node]
                if isinstance(mech, nn.ParameterDict):
                    with torch.no_grad():
                        mech['mu'].copy_(self.distributions[node]['mu'])
                        # Convert log_sigma to sigma for student
                        sigma_val = torch.exp(self.distributions[node]['log_sigma'])
                        mech['sigma'].copy_(sigma_val)


def fit_root_distributions(student_scm, ground_truth_scm, critic, root_nodes, n_samples=500, epochs=100, 
                          dedicated_learner=None):
    """
    Explicitly fit root node distributions using observational data.
    
    UPDATE Jan 20: Now supports optional dedicated root learner for better isolation.
    
    Problem: Root nodes (X1~N(0,1), X4~N(2,1)) don't learn from interventional data
             because interventions DO(X1=v) override the natural distribution.
             
    Solution: Train on pure observational data where roots are sampled from natural distributions.
              If dedicated_learner provided, use that; otherwise train student directly.
              
    Args:
        student_scm: StudentSCM to train
        ground_truth_scm: GroundTruthSCM to sample from
        critic: ScientificCritic for evaluation
        root_nodes: List of root node names (nodes with no parents)
        n_samples: Number of observational samples
        epochs: Training epochs for root fitting
        dedicated_learner: Optional DedicatedRootLearner for isolated training
    """
    logging.info(f"[Root Fitting] Fitting {len(root_nodes)} root nodes: {root_nodes}")
    
    # Generate pure observational data (no interventions)
    obs_data = ground_truth_scm.generate(n_samples=n_samples, interventions=None)
    
    if dedicated_learner is not None:
        # Use dedicated learner (preferred - better isolation)
        logging.info(f"[Root Fitting] Using dedicated root learner")
        losses = dedicated_learner.fit(obs_data, epochs=epochs)
        dedicated_learner.apply_to_student(student_scm)
        return losses
    
    # Fallback: Train student directly (original method)
    optimizer = optim.Adam(student_scm.parameters(), lr=0.001)
    
    initial_losses = {}
    for node in root_nodes:
        if node in student_scm.mechanisms:
            mech = student_scm.mechanisms[node]
            if isinstance(mech, nn.ParameterDict):
                with torch.no_grad():
                    pred_mu = mech['mu'].item()
                    pred_sigma = mech['sigma'].item()
                    true_samples = obs_data[node]
                    true_mu = true_samples.mean().item()
                    true_sigma = true_samples.std().item()
                    initial_losses[node] = abs(pred_mu - true_mu) + abs(pred_sigma - true_sigma)
    
    # Train specifically on root nodes
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        total_loss = 0.0
        for node in root_nodes:
            if node in student_scm.mechanisms:
                mech = student_scm.mechanisms[node]
                if isinstance(mech, nn.ParameterDict):
                    pred_mu = mech['mu']
                    pred_sigma = F.softplus(mech['sigma'])
                    
                    true_samples = obs_data[node]
                    true_mu = true_samples.mean()
                    true_sigma = true_samples.std()
                    
                    # Loss: match both mean and std
                    loss = (pred_mu - true_mu) ** 2 + (pred_sigma - true_sigma) ** 2
                    total_loss += loss
        
        if total_loss > 0:
            total_loss.backward()
            optimizer.step()
    
    # Log improvement
    final_losses = {}
    for node in root_nodes:
        if node in student_scm.mechanisms:
            mech = student_scm.mechanisms[node]
            if isinstance(mech, nn.ParameterDict):
                with torch.no_grad():
                    pred_mu = mech['mu'].item()
                    pred_sigma = mech['sigma'].item()
                    true_samples = obs_data[node]
                    true_mu = true_samples.mean().item()
                    true_sigma = true_samples.std().item()
                    final_losses[node] = abs(pred_mu - true_mu) + abs(pred_sigma - true_sigma)
    
    logging.info(f"[Root Fitting] Complete - Initial: {initial_losses}, Final: {final_losses}")
    return final_losses


def calculate_value_novelty_bonus(value, target, value_history, window=100):
    """
    Reward exploring novel intervention values.
    
    Even if the SCM doesn't improve, exploring new regions of value space
    has inherent value for future learning.
    
    Args:
        value: Current intervention value
        target: Current intervention target
        value_history: List of (target, value) tuples
        window: Look-back window
        
    Returns:
        bonus: Novelty bonus (0-5)
    """
    recent_values = [v for t, v in value_history[-window:] if t == target]
    
    if len(recent_values) < 5:
        return 5.0  # Early exploration bonus
    
    # Compute distance to nearest previous value
    distances = [abs(value - prev_val) for prev_val in recent_values]
    min_distance = min(distances) if distances else 10.0
    
    # Reward novelty (values far from previous ones)
    # Scale: 0-2 distance → 0-5 bonus
    novelty_bonus = min(5.0, min_distance * 2.5)
    
    return novelty_bonus


def compute_unified_diversity_score(target, recent_targets, all_nodes, max_concentration=0.4, recent_window=100,
                                   collider_parents=None, node_losses=None):
    """
    Unified diversity score - consolidates all diversity concerns into one function.
    
    UPDATED Jan 21: Added adaptive concentration threshold for collider parent learning.
    
    CRITICAL FIX: The concentration penalty was fighting against necessary X2 exploration
    for learning the X3 collider. Now uses adaptive threshold that relaxes when learning
    is still in progress.
    
    Computes:
    1. Entropy bonus (encourages balanced distribution across all nodes)
    2. Undersampling bonus (rewards intervening on neglected nodes)
    3. ADAPTIVE Concentration penalty (penalizes oversampling, but relaxes for active learning)
    
    Args:
        target: Current intervention target
        recent_targets: List of recent intervention targets
        all_nodes: List of all possible nodes
        max_concentration: Base maximum allowed concentration (e.g., 0.4 = 40%)
        recent_window: Window size for computing statistics
        collider_parents: Optional list of nodes that are parents of colliders
        node_losses: Optional dict of current per-node losses
        
    Returns:
        score: Unified diversity score (positive = good diversity)
    """
    if len(recent_targets) < 20:
        return 0.0
    
    recent = recent_targets[-recent_window:]
    counts = Counter(recent)
    total = len(recent)
    
    # 1. ENTROPY: Reward balanced distribution
    probs = [counts[n] / total for n in all_nodes]
    entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
    entropy_bonus = 50.0 * entropy  # Scale to be significant
    
    # 2. UNDERSAMPLING: Reward intervening on neglected nodes
    expected_freq = 1.0 / len(all_nodes)
    actual_freq = counts[target] / total
    deficit = expected_freq - actual_freq
    undersample_bonus = 200.0 * max(0, deficit)  # Only reward if undersampled
    
    # 3. ADAPTIVE CONCENTRATION PENALTY
    # CRITICAL FIX: Don't penalize concentration on collider parents if learning is active
    adaptive_threshold = max_concentration
    
    if collider_parents and target in collider_parents and node_losses:
        # Check if any collider children of this target still have high loss
        # If so, this parent needs continued sampling - relax threshold
        for node, loss in node_losses.items():
            if loss > 0.3:  # Node still learning
                # Relax threshold to allow strategic concentration
                adaptive_threshold = max(0.75, max_concentration + 0.15)
                break
    
    max_count = max(counts.values())
    concentration = max_count / total
    if concentration > adaptive_threshold:
        excess = concentration - adaptive_threshold
        # REDUCED penalty strength from 300.0 to 150.0
        concentration_penalty = -150.0 * excess
    else:
        concentration_penalty = 0.0
    
    # Combine all diversity concerns
    total_score = entropy_bonus + undersample_bonus + concentration_penalty
    
    return total_score


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
    parser.add_argument("--undersampled_bonus", type=float, default=200.0, help="Strong bonus for severely under-sampled nodes (INCREASED from 100.0 to address policy collapse)")
    parser.add_argument("--diversity_constraint", action="store_true", help="Enforce mandatory diversity: reject candidates targeting over-sampled nodes when collapse detected")
    parser.add_argument("--diversity_threshold", type=float, default=0.60, help="Threshold for mandatory diversity enforcement (e.g., 60 percent triggers constraint)")
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
    parser.add_argument("--pretrain_steps", type=int, default=100, help="Supervised pre-training steps before DPO")
    parser.add_argument("--pretrain_interval", type=int, default=50, help="Re-run pre-training every N episodes (0=disabled)")
    parser.add_argument("--smart_breaker", action="store_true", default=True, help="Use smart collapse breaker that prioritizes collider parents")
    # CRITICAL FIX: Periodic Observational Training to prevent mechanism forgetting
    parser.add_argument("--obs_train_interval", type=int, default=3, help="Train on observational data every N steps (0=disabled) - INCREASED from 5 for better root learning")
    parser.add_argument("--obs_train_samples", type=int, default=200, help="Number of observational samples per training injection - INCREASED from 100")
    parser.add_argument("--obs_train_epochs", type=int, default=100, help="Training epochs for observational data - INCREASED from 50")
    
    # NEW: Early stopping parameters to detect training saturation
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping when training plateaus")
    parser.add_argument("--early_stop_patience", type=int, default=20, help="Episodes to wait before early stopping")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.01, help="Minimum improvement to reset patience")
    parser.add_argument("--early_stop_min_episodes", type=int, default=40, help="Minimum episodes before allowing early stop (prevents stopping too early)")
    parser.add_argument("--zero_reward_threshold", type=float, default=0.92, help="Stop if this fraction of recent steps have zero reward (increased from 0.85)")
    parser.add_argument("--use_per_node_convergence", action="store_true", help="Use per-node convergence check (recommended - more intelligent than zero-reward)")
    parser.add_argument("--node_convergence_patience", type=int, default=10, help="Episodes each node must stay converged before stopping")
    
    # NEW: Root-specific training
    parser.add_argument("--root_fitting", action="store_true", help="Enable root-specific distribution fitting")
    parser.add_argument("--root_fit_interval", type=int, default=5, help="Fit root distributions every N episodes")
    parser.add_argument("--root_fit_samples", type=int, default=500, help="Samples for root fitting")
    parser.add_argument("--root_fit_epochs", type=int, default=100, help="Epochs for root fitting")
    parser.add_argument("--use_dedicated_root_learner", action="store_true", help="Use dedicated root learner (recommended - better isolation from interventional data)")
    parser.add_argument("--dedicated_root_interval", type=int, default=3, help="Train dedicated root learner every N episodes")
    
    # NEW: Improved diversity penalties
    parser.add_argument("--diversity_reward_weight", type=float, default=0.3, help="Weight for diversity reward (0.0-1.0)")
    parser.add_argument("--max_concentration", type=float, default=0.4, help="Maximum allowed concentration on any single node (reduced to 40 percent from 50 percent based on test results)")
    parser.add_argument("--concentration_penalty", type=float, default=200.0, help="Penalty for exceeding max_concentration")
    
    # NEW: Reference policy updates
    parser.add_argument("--update_reference_interval", type=int, default=25, help="Update reference policy every N episodes (0=never)")
    
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

    # --- NEW: DETAILED VALUE LOGGING ---
    # Create a separate CSV for tracking value distributions of collider parents
    value_log_path = os.path.join(run_dir, "value_diversity.csv")
    with open(value_log_path, 'w') as f:
        f.write("episode,step,node,value,is_breaker\n")

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

    # 2.5. Supervised Pre-training Phase (CRITICAL FIX for LLM ignoring prompt)
    if use_pretrained and args.pretrain_steps > 0:
        # Generate initial node losses for pre-training context
        temp_student = StudentSCM(M_star)
        _, pretrain_losses = critic.evaluate_mechanisms_detailed(temp_student)
        
        supervised_pretrain_llm(
            policy_net, temp_student, M_star.graph, dsl.nodes, pretrain_losses,
            optimizer_agent, n_steps=args.pretrain_steps,
            value_min=args.value_min, value_max=args.value_max
        )
        
        # Update reference policy after pre-training
        ref_policy = copy.deepcopy(policy_net)
        ref_policy.eval()
        logging.info("Reference policy updated after pre-training.")

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
    
    # Create student reference container for emergency save handler
    # Using dict to allow mutation (reference updates during training)
    student_ref = {"student": StudentSCM(M_star)}
    
    # --- CRITICAL: EMERGENCY SAVE HANDLER ---
    # Register signal handlers for graceful shutdown on timeout/interruption
    # Note: history_data holds references to the lists above, which get updated during training
    # student_ref["student"] gets updated each episode
    history_data = {
        "loss_history": loss_history,
        "reward_history": reward_history,
        "target_history": target_history,
        "value_history": value_history,
        "episode_history": episode_history,
        "step_history": step_history,
        "cov_bonus_history": cov_bonus_history,
        "score_history": score_history,
        "nodes": dsl.nodes,
        "metrics": None,  # Will build from history lists in handler
        "student_ref": student_ref  # Mutable reference
    }
    
    save_handler = create_emergency_save_handler(run_dir, M_star, history_data)
    signal.signal(signal.SIGTERM, save_handler)  # SLURM timeout
    signal.signal(signal.SIGINT, save_handler)   # Ctrl+C
    atexit.register(lambda: save_handler())      # Normal exit
    
    logging.info("✓ Registered emergency save handlers (SIGTERM, SIGINT, atexit)")
    
    # NEW: Log enabled improvements
    logging.info("\n" + "="*70)
    logging.info("TRAINING IMPROVEMENTS ENABLED:")
    logging.info("="*70)
    if args.early_stopping:
        logging.info(f"✓ Early Stopping: patience={args.early_stop_patience}, min_episodes={args.early_stop_min_episodes}, min_delta={args.early_stop_min_delta}")
        if args.use_per_node_convergence:
            logging.info(f"  Method: Per-node convergence (patience={args.node_convergence_patience}) - RECOMMENDED")
        else:
            logging.info(f"  Method: Zero-reward threshold ({args.zero_reward_threshold*100:.0f}%) - FALLBACK")
    else:
        logging.info("✗ Early stopping disabled (use --early_stopping)")
    
    if args.use_dedicated_root_learner:
        logging.info(f"✓ Dedicated Root Learner: interval={args.dedicated_root_interval} - PRIMARY root learning method")
    else:
        logging.info(f"✗ Dedicated root learner disabled - will use observational training fallback")
        logging.info(f"  Obs Training: interval={args.obs_train_interval}, samples={args.obs_train_samples}, epochs={args.obs_train_epochs}")
    
    logging.info(f"✓ SIMPLIFIED REWARD SYSTEM: 3 components (was 11)")
    logging.info(f"  - Information gain (primary objective)")
    logging.info(f"  - Node importance (parent of high-loss nodes)")
    logging.info(f"  - Unified diversity (entropy + undersampling + concentration)")
    logging.info(f"✓ Diversity weight: {args.diversity_reward_weight}, max_concentration: {args.max_concentration*100:.0f}%")
    logging.info(f"✓ Hard Cap Threshold: 60% (backup safety mechanism)")
    
    if args.update_reference_interval > 0:
        logging.info(f"✓ Reference Policy Updates: every {args.update_reference_interval} episodes")
    else:
        logging.info("✗ Reference policy updates disabled")
    
    logging.info("="*70 + "\n")
    
    logging.info(f"--- Starting Discovery Loop ({args.episodes} Episodes) ---")
    
    # Identify collider nodes for special tracking
    collider_nodes = [n for n in M_star.nodes if len(list(M_star.graph.predecessors(n))) >= 2]
    if collider_nodes:
        logging.info(f"Collider nodes identified (multi-parent): {collider_nodes}")
    
    # Identify root nodes (no parents) for special fitting
    root_nodes = [n for n in M_star.nodes if len(list(M_star.graph.predecessors(n))) == 0]
    if root_nodes:
        logging.info(f"Root nodes identified (no parents): {root_nodes}")
    
    # NEW: Dedicated root learner initialization
    dedicated_root_learner = None
    if args.use_dedicated_root_learner and root_nodes:
        dedicated_root_learner = DedicatedRootLearner(root_nodes)
        logging.info(f"✓ Dedicated root learner initialized for {root_nodes}")
    
    # NEW: Early stopping initialization
    early_stopper = None
    recent_rewards_for_stopping = deque(maxlen=100)  # Track last 100 step rewards
    if args.early_stopping:
        early_stopper = EarlyStopping(
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
            min_episodes=args.early_stop_min_episodes
        )
        logging.info(f"✓ Early stopping enabled (patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta}, min_episodes={args.early_stop_min_episodes})")
        if args.use_per_node_convergence:
            logging.info(f"  Using per-node convergence (patience={args.node_convergence_patience})")
        else:
            logging.info(f"  Using zero-reward threshold ({args.zero_reward_threshold*100:.0f}%)")

    recent_action_counts = deque(maxlen=500)
    recent_values_by_target = {n: deque(maxlen=200) for n in dsl.nodes}
    recent_value_bins_by_target = {n: deque(maxlen=200) for n in dsl.nodes}

    for episode in range(args.episodes):
        current_student = StudentSCM(M_star)
        student_ref["student"] = current_student  # Update for emergency handler
        learner = SCMLearner(current_student, lr=args.learner_lr, buffer_steps=args.buffer_steps)
        episode_action_counts = Counter()
        best_mech_loss = float("inf")
        no_improve_steps = 0

        # Periodic re-training to combat policy drift (CRITICAL for LLM policies)
        # UPDATED Jan 21: Also retrain if gradients are near-zero (DPO not learning)
        recent_grad_norm = getattr(main, '_last_grad_norm', 1.0)  # Access last gradient norm
        should_retrain = (
            (args.pretrain_interval > 0 and episode > 0 and episode % args.pretrain_interval == 0) or
            (episode > 0 and episode % 10 == 0 and recent_grad_norm < 0.001)  # Emergency retrain
        )
        
        if use_pretrained and should_retrain:
            if recent_grad_norm < 0.001:
                logging.info(f"--- Emergency Re-training at Episode {episode} (grad_norm={recent_grad_norm:.6f}) ---")
            else:
                logging.info(f"--- Periodic Re-training at Episode {episode} ---")
            
            _, retrain_losses = critic.evaluate_mechanisms_detailed(current_student)
            supervised_pretrain_llm(
                policy_net, current_student, M_star.graph, dsl.nodes, retrain_losses,
                optimizer_agent, n_steps=args.pretrain_steps // 2,  # Shorter re-training
                value_min=args.value_min, value_max=args.value_max
            )
            # Update reference policy
            ref_policy = copy.deepcopy(policy_net)
            ref_policy.eval()
            logging.info("Reference policy updated after re-training.")
        
        # NEW: Periodic reference policy update (separate from re-training)
        # This addresses the KL divergence explosion (0 → -2,300 observed in recent runs)
        if args.update_reference_interval > 0 and episode > 0 and episode % args.update_reference_interval == 0:
            logging.info(f"[Ref Update] Updating reference policy at episode {episode}")
            ref_policy = copy.deepcopy(policy_net)
            ref_policy.eval()
            
            # Log current generation distribution for monitoring
            if hasattr(policy_net, 'generation_stats'):
                logging.info(f"  Current generation: {policy_net.generation_stats}")
        
        # NEW: Dedicated root learner training (more frequent, better isolation)
        if args.use_dedicated_root_learner and dedicated_root_learner and episode > 0 and episode % args.dedicated_root_interval == 0:
            obs_data = M_star.generate(n_samples=1000, interventions=None)
            losses = dedicated_root_learner.fit(obs_data, epochs=200)
            dedicated_root_learner.apply_to_student(current_student)
            logging.info(f"[Dedicated Root Learner] Trained and applied at episode {episode}: {losses}")
        
        # SIMPLIFIED: Use only dedicated root learner (if enabled), skip redundant root_fitting
        # Dedicated learner is more isolated and effective than mixed approach

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
            
            # Build intervention history for prompt context
            intervention_history = list(recent_action_counts)
            
            # SPEED IMPROVEMENT: Reduce candidates after warmup
            if episode < 20:
                num_candidates = args.candidates  # Full exploration early
            elif episode < 50:
                num_candidates = max(3, args.candidates // 2)  # Half after warmup
            else:
                num_candidates = 3  # Minimal late in training
            
            for k in range(num_candidates):
                cmd_str, plan = policy_net.generate_experiment(
                    current_student, 
                    node_losses=node_losses_start,
                    intervention_history=intervention_history
                )
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
                    
                    # NEW: Add value novelty bonus to combat zero-reward saturation
                    tgt = plan.get("target")
                    val = plan.get("value")
                    novelty_bonus = calculate_value_novelty_bonus(
                        val, tgt, list(zip(target_history, value_history))
                    )
                    reward += novelty_bonus * 0.1  # Small weight to avoid dominating main reward
                    # Prefer interventions that help high-loss *direct children* (best for learning X1/X2 -> X3).
                    # normalize=True ensures we value "Urgency" (Avg Loss) over "Volume" (Total Loss)
                    node_weight = _direct_child_impact_weight(M_star.graph, tgt, node_losses_start, normalize=True)
                    denom = float(sum(node_losses_start.values())) + 1e-8
                    norm_weight = node_weight / denom
                    under_sample = 1.0 / np.sqrt(1.0 + episode_action_counts.get(tgt, 0))
                    cov_bonus = args.cov_bonus * norm_weight * under_sample
                    # SIMPLIFIED REWARD SYSTEM (Jan 20 Simplification)
                    # Removed 6 redundant components: val_bonus, bin_bonus, bal_bonus, 
                    # disent_bonus, leaf_pen, collapse_pen
                    # Consolidated diversity: diversity_penalty, coverage_bonus, undersample_bonus → unified
                    
                    # 1. NODE IMPORTANCE: Intervene on parents of high-loss nodes
                    node_importance = cov_bonus  # Already computed (node_weight * under_sample)
                    
                    # 2. UNIFIED DIVERSITY SCORE: All diversity concerns in one function
                    # UPDATED: Pass collider parents and node losses for adaptive threshold
                    collider_parents = [p for node in dsl.nodes 
                                      for p in M_star.get_parents(node) 
                                      if len(M_star.get_parents(node)) > 1]
                    
                    unified_diversity = compute_unified_diversity_score(
                        target=tgt,
                        recent_targets=list(recent_action_counts),
                        all_nodes=dsl.nodes,
                        max_concentration=args.max_concentration,
                        recent_window=100,
                        collider_parents=collider_parents,
                        node_losses=node_losses_start
                    )
                    
                    # 3. FINAL SCORE: Information gain + node importance + diversity
                    # Simple, interpretable, no redundancy
                    score = (
                        reward +                                    # Information gain (primary)
                        node_importance +                           # Parent of high-loss nodes
                        args.diversity_reward_weight * unified_diversity  # Balanced exploration
                    )
                    
                    candidates.append((cmd_str, reward, cov_bonus, score, plan))
                    
                    # SIMPLIFIED: Diagnostic logging for new reward system
                    if episode % 10 == 0 and step % 50 == 0 and k < 3:
                        logging.info(
                            f"    [Simplified] Candidate {k+1}: target={tgt}, "
                            f"reward={reward:.2f}, node_importance={cov_bonus:.2f}, "
                            f"diversity={unified_diversity:.2f}, score={score:.2f}"
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
            
            # --- CRITICAL FIX: SMART COLLAPSE BREAKER ---
            # If we are collapsed (e.g. > 50% on one node) and the agent ONLY proposes the collapsed node,
            # we must forcefully inject an alternative. 
            # 
            # KEY INSIGHT: Instead of random injection, PRIORITIZE PARENTS OF HIGH-LOSS COLLIDERS.
            # This directly addresses the X1→X3, X2→X3 learning failure.
            if top_node is not None and top_frac > 0.50:
                # Check if we have any valid candidate that is NOT the top_node
                has_alternative = any(c[4] is not None and c[4].get("target") != top_node for c in sorted_cands if c[1] > -9.0)
                
                if not has_alternative:
                    # SMART BREAKER: Find the highest-loss multi-parent node and target its under-sampled parent
                    breaker_cmd = None
                    breaker_target = None
                    breaker_reason = "random"
                    
                    if args.smart_breaker:
                        # 1. Check for COLLIDER PARENT COLLAPSE (Single-Value Trap)
                        # If collapsed on a parent of a collider (like X2), we might be trapped in a single value.
                        # We must inject the SAME node but with a RADICALLY different value.
                        is_collider_parent = False
                        for child in current_student.nodes:
                             parents = list(M_star.graph.predecessors(child))
                             if len(parents) >= 2 and top_node in parents:
                                 is_collider_parent = True
                                 break
                        
                        if is_collider_parent:
                            recent_vals = list(recent_values_by_target.get(top_node, []))
                            if recent_vals:
                                mean_val = np.mean(recent_vals)
                                # Pick a value far from the mean (e.g. flip sign or go to bounds)
                                if mean_val > 0:
                                    new_val = random.uniform(args.value_min, 0.0)
                                else:
                                    new_val = random.uniform(0.0, args.value_max)
                                
                                breaker_cmd = f"DO {top_node} = {new_val:.4f}"
                                breaker_reason = f"value_diversity_for_collider_parent (mean={mean_val:.2f})"

                        # 2. If not trapped in value collapse, try to target neglected parents
                        if breaker_cmd is None:
                            # Find colliders (multi-parent nodes) sorted by loss
                            colliders_by_loss = []
                            for node in current_student.nodes:
                                parents = list(M_star.graph.predecessors(node))
                                if len(parents) >= 2:
                                    loss = node_losses_start.get(node, 0.0)
                                    colliders_by_loss.append((node, parents, loss))
                            colliders_by_loss.sort(key=lambda x: x[2], reverse=True)
                            
                            # For the highest-loss collider, find its least-sampled parent (excluding top_node)
                            for collider, parents, loss in colliders_by_loss:
                                if loss < 0.5:  # Collider is already learned
                                    continue
                                
                                # Count recent interventions on each parent
                                parent_counts = {p: sum(1 for a in recent_action_counts if a == p) for p in parents}
                                
                                # Find least-sampled parent that isn't the collapsed node
                                valid_parents = [p for p in parents if p != top_node]
                                if valid_parents:
                                    least_sampled = min(valid_parents, key=lambda p: parent_counts.get(p, 0))
                                    breaker_target = least_sampled
                                    breaker_reason = f"parent of failing {collider} (loss={loss:.2f})"
                                    break
                    
                    # Fallback to random if smart breaker didn't find a target
                    if breaker_cmd is None and breaker_target is None:
                        valid_others = [n for n in dsl.nodes if n != top_node]
                        breaker_target = random.choice(valid_others) if valid_others else None
                        breaker_reason = "random fallback"
                    
                    if breaker_cmd is None and breaker_target:
                        value = random.uniform(float(args.value_min), float(args.value_max))
                        breaker_cmd = f"DO {breaker_target} = {value:.4f}"

                    if breaker_cmd:
                        breaker_plan = dsl.parse_to_dict(breaker_cmd)
                        
                        # Score it high enough to win
                        tgt = breaker_plan.get("target")
                        node_weight = _direct_child_impact_weight(M_star.graph, tgt, node_losses_start, normalize=True)
                        denom = float(sum(node_losses_start.values())) + 1e-8
                        norm_weight = node_weight / denom
                        under_sample = 1.0 / np.sqrt(1.0 + episode_action_counts.get(tgt, 0))
                        cov_bonus = args.cov_bonus * norm_weight * under_sample
                        
                        # Add disentanglement bonus for collider parent targeting
                        disent_bonus = _disentanglement_bonus(M_star.graph, tgt, node_losses_start)
                        
                        # Give it a high score to ensure it wins
                        score = 10.0 + cov_bonus + disent_bonus * 0.1
                        
                        # Insert at the top
                        sorted_cands.insert(0, (breaker_cmd, 0.0, cov_bonus, score, breaker_plan))
                        logging.info(f"  [Smart Breaker] Injected '{breaker_cmd}' ({breaker_reason}, Score: {score:.2f})")

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
            
            # --- CRITICAL FIX: HARD INTERVENTION CAP ---
            # Prevent catastrophic over-concentration on single node (e.g., 99% X2)
            MAX_NODE_FRACTION = 0.70
            
            if len(recent_action_counts) > 10 and winner_plan is not None:
                node_counts = Counter(recent_action_counts)
                total_interventions = len(recent_action_counts)
                winner_target = winner_plan.get("target")
                
                if winner_target:
                    winner_count = node_counts.get(winner_target, 0)
                    winner_fraction = winner_count / total_interventions
                    
                    if winner_fraction > MAX_NODE_FRACTION:
                        logging.info(f"  [Hard Cap] {winner_target} at {winner_fraction:.1%} > {MAX_NODE_FRACTION:.0%}, forcing alternative")
                        
                        # Find undersampled collider parent
                        collider_nodes_local = [n for n in M_star.nodes if len(M_star.get_parents(n)) > 1]
                        
                        if collider_nodes_local:
                            collider = collider_nodes_local[0]
                            parents = M_star.get_parents(collider)
                            
                            # Pick least-sampled parent
                            undersampled = min(parents, key=lambda p: node_counts.get(p, 0))
                            
                            # Generate new plan
                            forced_value = random.uniform(args.value_min, args.value_max)
                            winner_plan = {
                                "target": undersampled,
                                "value": forced_value,
                                "command": f"DO {undersampled} = {forced_value:.4f}"
                            }
                            winner_cmd = winner_plan["command"]
                            winner_score = 100.0  # High score to indicate forced action
                            
                            logging.info(f"  [Hard Cap] Forced intervention on {undersampled}")
            
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
                    loss = dpo_loss_llm(
                        policy_net, ref_policy, current_student, 
                        winner_cmd, loser_cmd, 
                        node_losses=node_losses_start,
                        intervention_history=intervention_history
                    )
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
                
                # Gradient monitoring (periodic) - CRITICAL for diagnosing DPO failure
                if episode % 20 == 0 and step == 0:
                    total_grad_norm = 0.0
                    num_params = 0
                    for name, param in policy_net.named_parameters():
                        if param.grad is not None:
                            total_grad_norm += param.grad.norm().item() ** 2
                            num_params += 1
                    total_grad_norm = np.sqrt(total_grad_norm)
                    # Store for use in retraining decision
                    main._last_grad_norm = total_grad_norm
                    logging.info(f"  [Gradient Check] Episode {episode}: grad_norm={total_grad_norm:.6f}, num_params_with_grad={num_params}")
                    if total_grad_norm < 1e-6:
                        logging.warning(f"  [WARNING] Gradients are near-zero! DPO may not be training the model.")
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                
                optimizer_agent.step()
                loss_history.append(loss.item())
                reward_history.append(winner_reward)
                score_history.append(winner_score)
                cov_bonus_history.append(winner_cov_bonus)
                
                # NEW: Track rewards for early stopping detection
                if args.early_stopping:
                    recent_rewards_for_stopping.append(winner_reward)
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
                val = float(winner_plan.get("value"))
                try:
                    recent_values_by_target[tgt].append(val)
                except Exception:
                    pass
                
                # --- NEW: LOG VALUES FOR COLLIDER PARENTS ---
                # Check if this node is a collider parent
                is_cp = False
                for child in current_student.nodes:
                    pars = list(M_star.graph.predecessors(child))
                    if len(pars) >= 2 and tgt in pars:
                        is_cp = True
                        break
                
                if is_cp:
                    is_breaker_flag = 1 if "Breaker" in winner_cmd else 0 # Approximate check
                    # Better check: compare cmd string to known breaker injections
                    # or just assume high scores early in episode might be breaker
                    
                    with open(value_log_path, 'a') as f:
                        f.write(f"{episode},{step},{tgt},{val:.4f},{is_breaker_flag}\n")

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
                
                # OBSERVATIONAL TRAINING: Preserve mechanisms by periodic observational data injection
                # This prevents catastrophic forgetting of mechanisms (especially X2)
                # when interventional training dominates
                if args.obs_train_interval > 0 and step > 0 and step % args.obs_train_interval == 0:
                    obs_data = M_star.generate(n_samples=args.obs_train_samples, interventions=None)
                    learner.train_step(obs_data, n_epochs=args.obs_train_epochs)
                    if episode % 10 == 0 and step % (args.obs_train_interval * 5) == 0:
                        logging.info(f"  [Obs Training] Step {step}: Injected {args.obs_train_samples} samples")
        
        # --- EARLY STOPPING CHECKS ---
        # Check if training has saturated (most steps producing zero reward)
        if args.early_stopping and early_stopper is not None:
            # NEW: Don't allow early stopping before minimum episodes
            if episode < early_stopper.min_episodes:
                if episode % 10 == 0:
                    logging.info(f"  [Early Stop] Skipping checks (episode {episode} < min {early_stopper.min_episodes})")
            else:
                # Get final loss for this episode
                final_loss, node_losses_final = critic.evaluate_mechanisms_detailed(current_student)
                
                # PRIORITY 1: Per-node convergence check (more intelligent)
                if args.use_per_node_convergence:
                    if early_stopper.check_per_node_convergence(
                        node_losses_final,
                        patience=args.node_convergence_patience
                    ):
                        logging.info(f"✓ Per-node convergence detected at episode {episode}/{args.episodes}")
                        logging.info(f"   Final loss: {final_loss:.4f}, episodes trained: {episode+1}")
                        break
                else:
                    # Fallback: Check loss-based stopping
                    if early_stopper.check_loss(final_loss):
                        logging.info(f"✓ Early stopping triggered at episode {episode}/{args.episodes}")
                        logging.info(f"   Final loss: {final_loss:.4f}, episodes trained: {episode+1}")
                        break
                    
                    # Check zero-reward-based stopping (training saturation)
                    # IMPROVED: Check with smaller window (50 instead of 100) and be more aggressive
                    if len(recent_rewards_for_stopping) >= 50:
                        recent_window = list(recent_rewards_for_stopping)[-100:] if len(recent_rewards_for_stopping) >= 100 else list(recent_rewards_for_stopping)
                        zero_count = sum(1 for r in recent_window if abs(r) < 0.01)
                        zero_fraction = zero_count / len(recent_window)
                        
                        if zero_fraction >= args.zero_reward_threshold:
                            logging.info(f"✓ Training saturation detected at episode {episode}/{args.episodes}")
                            logging.info(f"   Zero-reward fraction: {zero_fraction:.1%} (threshold: {args.zero_reward_threshold:.1%})")
                            logging.info(f"   Episodes trained: {episode+1}")
                            break
            
            # Log progress every 10 episodes
            if episode % 10 == 0 and len(recent_rewards_for_stopping) >= 50:
                zero_count = sum(1 for r in recent_rewards_for_stopping if abs(r) < 0.01)
                zero_pct = zero_count / len(recent_rewards_for_stopping) * 100
                logging.info(f"  [Early Stop Monitor] Zero-reward steps: {zero_pct:.1f}% (threshold: {args.zero_reward_threshold*100:.0f}%)")
        
        # --- INCREMENTAL CHECKPOINT SAVES ---
        # Save checkpoint every 50 episodes for recovery
        if episode > 0 and episode % 50 == 0:
            save_checkpoint(run_dir, episode, policy_net, optimizer_agent,
                          loss_history, reward_history, recent_action_counts)
        
        # Save intermediate visualizations every 100 episodes
        if episode > 0 and episode % 100 == 0:
            try:
                visualize_contrast_save(M_star, current_student, run_dir)
                save_plots(run_dir, loss_history, reward_history, 
                          target_history, value_history, dsl.nodes)
                logging.info(f"✓ Saved intermediate visualizations at episode {episode}")
            except Exception as e:
                logging.error(f"Failed to save intermediate visualizations: {e}")

    # 4. Final Evaluation
    logging.info("--- Running Final Evaluation ---")
    
    # Log LLM generation diagnostics if using HuggingFace policy
    if use_pretrained and hasattr(policy_net, 'log_generation_diagnostics'):
        policy_net.log_generation_diagnostics()
    eval_targets = []
    eval_values = []
    
    # Use the LAST student state for evaluation visualization
    visualize_contrast_save(M_star, current_student, run_dir)
    
    # Visualize the SCM graph structure with final losses
    final_total_loss, final_node_losses = critic.evaluate_mechanisms_detailed(current_student)
    visualize_scm_graph(M_star, run_dir, node_losses=final_node_losses)
    logging.info(f"Final mechanism losses: {final_node_losses}")
    
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

    # Save DPO training diagnostics
    dpo_logger = get_dpo_logger()
    dpo_logger.save(run_dir)

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
