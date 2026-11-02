"""New artificial arithmetic checks, no pretrained model or GPU."""
import copy
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.nn import functional as F
from llm_intervention_canonical_scoring import (
    FixedRowFP32Projection, install_canonical_projection,
    sequence_log_probs_canonical, supervised_loss_canonical)
from llm_intervention_learning import install_adapters, preference_loss
from llm_intervention_learning import ActionBatch
from test_llm_intervention_action_scoring import CausalToy, fixture


def toy():
    m=CausalToy().to(torch.bfloat16)
    m.lm_head.bias = None  # Qwen fixture contract uses a bias-free output head.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(390).get_state())
        install_adapters(m,rank=2,alpha=4.)
    with torch.no_grad():m.q_proj.B.fill_(.02);m.v_proj.B.fill_(.03)
    install_canonical_projection(m)
    return m


class CanonicalChecks(unittest.TestCase):
    def test_independent_masked_objective_and_asymmetric_dpo_oracle(self):
        class DirectHidden(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden=torch.nn.Parameter((torch.arange(36).reshape(2,9,2)%7/16).to(torch.bfloat16))
                self.lm_head=torch.nn.Linear(2,17,bias=False,dtype=torch.bfloat16).requires_grad_(False)
                with torch.no_grad():self.lm_head.weight.copy_((torch.arange(34).reshape(17,2)%9-4)/16)
                install_canonical_projection(self)
            def forward(self,input_ids,attention_mask,use_cache,logits_to_keep=0):
                h=self.hidden
                if isinstance(logits_to_keep,torch.Tensor):h=h[:,logits_to_keep]
                return SimpleNamespace(logits=self.lm_head(h))
        m=DirectHidden();batch=fixture();w=m.lm_head.weight.detach().double().tolist()
        h=m.hidden.detach().double().tolist();labels=batch.labels.tolist()
        scores=[0.,0.];probabilities={};counts=[0,0]
        for b in range(2):
            for j in range(1,9):
                if labels[b][j]==-100:continue
                logits=[sum(a*z for a,z in zip(row,h[b][j-1])) for row in w]
                top=max(logits);logz=top+math.log(sum(math.exp(v-top) for v in logits))
                scores[b]+=logits[labels[b][j]]-logz;counts[b]+=1
                probabilities[b,j]=[math.exp(v-logz) for v in logits]
        reference=torch.tensor([[-2.3,-7.1]],requires_grad=True);beta=.1
        margin=beta*((scores[0]-float(reference[0,0].detach()))-(scores[1]-float(reference[0,1].detach())))
        for objective in ('sft','dpo'):
            expected_loss=(-sum(scores[b]/counts[b] for b in range(2))/2 if objective=='sft'
                           else max(0.,-margin)+math.log1p(math.exp(-abs(margin))))
            coeff=([-1/(2*n) for n in counts] if objective=='sft'
                   else [-beta/(1+math.exp(margin)),beta/(1+math.exp(margin))])
            grad=torch.zeros(2,9,2,dtype=torch.float64)
            for (b,j),probs in probabilities.items():
                for k in range(2):
                    grad[b,j-1,k]=coeff[b]*sum(((c==labels[b][j])-probs[c])*w[c][k] for c in range(17))
            for selected in (False,True):
                m.zero_grad(set_to_none=True)
                actual=sequence_log_probs_canonical(m,batch,selected=selected)
                torch.testing.assert_close(actual,torch.tensor(scores),atol=2e-6,rtol=1e-6)
                loss=(supervised_loss_canonical(m,batch,selected=selected) if objective=='sft'
                      else preference_loss(actual.reshape(1,2),reference,beta=beta))
                loss.backward()
                self.assertAlmostEqual(float(loss.detach()),expected_loss,places=6)
                torch.testing.assert_close(m.hidden.grad,grad.to(torch.bfloat16),atol=1e-5,rtol=1e-2)
                self.assertIsNone(reference.grad)

    def test_padding_row_permutation_and_noncontiguous_inputs(self):
        batch=fixture();m=toy();before=sequence_log_probs_canonical(m,batch,selected=True)
        pad=ActionBatch(torch.cat([batch.input_ids,torch.zeros(2,3,dtype=torch.long)],1),
            torch.cat([batch.attention_mask,torch.zeros(2,3,dtype=torch.long)],1),
            torch.cat([batch.labels,torch.full((2,3),-100,dtype=torch.long)],1))
        for selected in (False,True):
            torch.testing.assert_close(sequence_log_probs_canonical(m,pad,selected=selected),before,atol=0,rtol=0)
            perm=ActionBatch(pad.input_ids.flip(0),pad.attention_mask.flip(0),pad.labels.flip(0))
            torch.testing.assert_close(sequence_log_probs_canonical(m,perm,selected=selected),before.flip(0),atol=0,rtol=0)
        x=torch.arange(36,dtype=torch.bfloat16).reshape(2,3,6)/32
        y=m.lm_head(x);z=x.transpose(0,1)
        self.assertFalse(z.is_contiguous())
        torch.testing.assert_close(m.lm_head(z),y.transpose(0,1),atol=0,rtol=0)
    def test_singleton_rows_and_permutation_extension_invariance(self):
        h=torch.nn.Linear(3,5,bias=False,dtype=torch.bfloat16).requires_grad_(False)
        p=FixedRowFP32Projection(h);x=torch.arange(18,dtype=torch.bfloat16).reshape(2,3,3)/16
        with patch('llm_intervention_precision.F.linear',wraps=F.linear) as calls:y=p(x)
        self.assertEqual(len(calls.call_args_list),6)
        self.assertTrue(all(c.args[0].shape==(1,3) for c in calls.call_args_list))
        torch.testing.assert_close(p(x[:,[2,0,1]]),y[:,[2,0,1]],atol=0,rtol=0)
        extended=torch.cat([x,torch.zeros(2,2,3,dtype=x.dtype)],dim=1)
        torch.testing.assert_close(p(extended)[:,:3],y,atol=0,rtol=0)

    def test_full_selected_supervised_and_preference_gradients(self):
        batch=fixture();m=toy()
        for objective in ('sft','dpo'):
            records=[]
            for selected in (False,True):
                m.zero_grad(set_to_none=True)
                score=sequence_log_probs_canonical(m,batch,selected=selected)
                loss=(supervised_loss_canonical(m,batch,selected=selected) if objective=='sft'
                      else preference_loss(score.reshape(1,2),torch.zeros(1,2),beta=.1))
                loss.backward()
                records.append((loss.detach(),{n:p.grad.clone() for n,p in m.named_parameters() if p.requires_grad}))
            torch.testing.assert_close(records[0][0],records[1][0],atol=0,rtol=0)
            for n,g in records[0][1].items():torch.testing.assert_close(g,records[1][1][n],atol=0,rtol=0)

    def test_explicit_scalar_softmax_and_input_gradient_oracle(self):
        h=torch.nn.Linear(2,3,bias=False,dtype=torch.bfloat16).requires_grad_(False)
        with torch.no_grad():h.weight.copy_(torch.tensor([[.5,-.25],[.25,.5],[-.5,.25]]))
        p=FixedRowFP32Projection(h);x=torch.tensor([[[.5,-.5]]],dtype=torch.bfloat16,requires_grad=True)
        logits=p(x);loss=-F.log_softmax(logits,dim=-1)[0,0,1];loss.backward()
        raw=[.375,-.125,-.375];z=sum(math.exp(v) for v in raw);prob=[math.exp(v)/z for v in raw]
        self.assertAlmostEqual(float(loss.detach()),-math.log(prob[1]),places=6)
        expected=[sum((prob[j]-(j==1))*float(h.weight[j,k]) for j in range(3)) for k in range(2)]
        torch.testing.assert_close(x.grad,torch.tensor([[expected]],dtype=torch.bfloat16),atol=0,rtol=0)

    def test_invalid_path_projection_and_input_rejected(self):
        m=toy()
        with self.assertRaises(ValueError):sequence_log_probs_canonical(m,fixture(),selected=1)
        with self.assertRaises(ValueError):install_canonical_projection(m)
        with self.assertRaises(ValueError):m.lm_head(torch.zeros(2,3,dtype=torch.bfloat16))
        with self.assertRaises(ValueError):m.lm_head(torch.zeros(2,0,3,dtype=torch.bfloat16))

if __name__=='__main__':unittest.main(verbosity=2)
