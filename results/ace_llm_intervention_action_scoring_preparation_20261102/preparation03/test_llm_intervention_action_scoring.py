"""Artificial full-versus-selected score/gradient oracles, no pretrained model."""
import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from llm_intervention_action_scoring import (action_positions, projection_profile,
    sequence_log_probs_selected, supervised_loss_selected)
from llm_intervention_learning import (ActionBatch, pack_actions, sequence_log_probs, supervised_loss,
    install_adapters, preference_loss, reference_policy, snapshot_reference)


class CausalToy(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(341).get_state())
            self.embedding = nn.Embedding(17, 6)
            self.q_proj = nn.Linear(6, 6)
            self.v_proj = nn.Linear(6, 6)
            self.lm_head = nn.Linear(6, 17)
        self.calls = 0
    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep=0):
        self.calls += 1
        assert use_cache is False
        h = self.embedding(input_ids)*attention_mask[..., None]
        h = torch.tanh(self.q_proj(h.cumsum(dim=1)))+self.v_proj(h)
        self.last_full_context = h.shape[1]
        if isinstance(logits_to_keep, torch.Tensor):
            h = h[:, logits_to_keep]
        self.last_projected_positions = h.shape[1]
        return SimpleNamespace(logits=self.lm_head(h))


def fixture():
    return pack_actions([2,3,4,5,6,7], [[8,9,1],[10,1]], 0, 32)


def loss_and_gradients(model, batch, loss):
    model.zero_grad(set_to_none=True)
    value = loss(model,batch)
    value.backward()
    return value.detach(), {n:p.grad.detach().clone() for n,p in model.named_parameters() if p.requires_grad}


class ScoringTests(unittest.TestCase):
    def test_full_projection_loss_and_all_parameter_gradients_equal(self):
        a = CausalToy(); b = copy.deepcopy(a); batch = fixture()
        full, grad_full = loss_and_gradients(a,batch,supervised_loss)
        short, grad_short = loss_and_gradients(b,batch,supervised_loss_selected)
        torch.testing.assert_close(full,short,rtol=1e-6,atol=1e-7)
        for n in grad_full:
            torch.testing.assert_close(grad_full[n],grad_short[n],rtol=2e-5,atol=2e-7)
        self.assertGreater(grad_short['embedding.weight'][2].abs().sum().item(),0)
        self.assertEqual(b.last_full_context,9)
        self.assertEqual(b.last_projected_positions,3)
        profile=projection_profile(batch,17)
        self.assertEqual(profile['full_logit_elements'],306)
        self.assertEqual(profile['selected_logit_elements'],102)

    def test_unequal_prompt_masks_shift_and_union_exact(self):
        batch=fixture()
        labels=batch.labels.clone();labels[0,3]=batch.input_ids[0,3]
        varied=ActionBatch(batch.input_ids,batch.attention_mask,labels)
        positions,_,_=action_positions(varied)
        self.assertEqual(positions.tolist(),[2,5,6,7])
        model=CausalToy()
        torch.testing.assert_close(sequence_log_probs(model,varied),sequence_log_probs_selected(model,varied),rtol=1e-6,atol=1e-6)

    def test_preference_adapter_gradients_and_fixed_reference_equal(self):
        a=CausalToy()
        with torch.random.fork_rng(devices=[]):
            install_adapters(a,rank=2,alpha=4.)
        with torch.no_grad():
            a.q_proj.B.fill_(.02);a.v_proj.B.fill_(.03)
        snapshot_reference(a)
        with torch.no_grad():
            a.q_proj.B.add_(.01)
        b=copy.deepcopy(a);batch=fixture()
        references={n:x.clone() for n,x in b.named_buffers()}
        def loss(scorer):
            def compute(m,batch):
                policy=scorer(m,batch).reshape(1,2)
                with reference_policy(m):reference=scorer(m,batch).reshape(1,2)
                return preference_loss(policy,reference)
            return compute
        va,ga=loss_and_gradients(a,batch,loss(sequence_log_probs))
        vb,gb=loss_and_gradients(b,batch,loss(sequence_log_probs_selected))
        torch.testing.assert_close(va,vb,rtol=1e-6,atol=1e-7)
        for n in ga:torch.testing.assert_close(ga[n],gb[n],rtol=2e-5,atol=2e-7)
        self.assertTrue(any(x.abs().sum()>0 for x in gb.values()))
        for n,x in b.named_buffers():torch.testing.assert_close(x,references[n],rtol=0,atol=0)

    def test_invalid_batches_rejected_before_forward(self):
        batch=fixture()
        variants=[]
        for kind in ('label_identity','first_token','padding','empty','attention_hole'):
            ids,att,lab=[x.clone() for x in (batch.input_ids,batch.attention_mask,batch.labels)]
            if kind=='label_identity':lab[0,-1]=2
            if kind=='first_token':lab[0,0]=ids[0,0]
            if kind=='padding':lab[1,-1]=ids[1,-1]
            if kind=='empty':lab[0]=-100
            if kind=='attention_hole':att[0,2]=0
            variants.append(ActionBatch(ids,att,lab))
        for bad in variants:
            m=CausalToy()
            with self.assertRaises(ValueError):sequence_log_probs_selected(m,bad)
            self.assertEqual(m.calls,0)

    def test_ignored_selection_and_nonfinite_outputs_rejected(self):
        class Ignores(CausalToy):
            def forward(self,**kwargs):
                kwargs.pop('logits_to_keep');return super().forward(**kwargs)
        with self.assertRaises(ValueError):sequence_log_probs_selected(Ignores(),fixture())
        class Nonfinite(CausalToy):
            def forward(self,**kwargs):
                out=super().forward(**kwargs);out.logits[0,0,0]=float('nan');return out
        with self.assertRaises(ValueError):sequence_log_probs_selected(Nonfinite(),fixture())


if __name__=='__main__':unittest.main()
