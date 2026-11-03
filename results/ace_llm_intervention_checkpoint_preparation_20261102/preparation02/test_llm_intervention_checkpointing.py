"""Artificial eval-mode recomputation checks without models or GPU."""
import copy
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from llm_intervention_checkpointing import DecoderCheckpointing


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[.3,-.2],[.4,.1]]))
        self.reference = False
    def forward(self, x, *, scale=1., use_cache=False):
        return torch.sin(x @ self.weight) * scale


def model():
    m=nn.Module();m.model=nn.Module();m.model.layers=nn.ModuleList([Block(),Block()]);m.eval();return m


class CheckpointChecks(unittest.TestCase):
    def test_frozen_input_parameter_gradients_and_numerical_oracle(self):
        m=model();c=DecoderCheckpointing(m,expected_layers=2);before=copy.deepcopy(m.state_dict());records=[]
        x=torch.tensor([[.2,.4],[-.1,.3]])
        for enabled in (False,True):
            c.set_enabled(enabled);m.zero_grad(set_to_none=True);h=x
            for layer in m.model.layers:h=layer(h,scale=.7,use_cache=False)
            loss=(h**2).sum();loss.backward()
            records.append((loss.detach(),[p.grad.clone() for p in m.parameters()]))
        self.assertFalse(x.requires_grad)
        torch.testing.assert_close(records[0][0],records[1][0],atol=0,rtol=0)
        for a,b in zip(records[0][1],records[1][1]):torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertEqual(c.forward_calls,2);self.assertEqual(c.recompute_calls,2)
        for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
        # Independent central difference for one trainable scalar.
        w=m.model.layers[0].weight;old=w[0,0].item();eps=.001;losses=[]
        with torch.no_grad():
            for sign in (-1,1):
                w[0,0]=old+sign*eps;h=x
                for layer in m.model.layers:h=torch.sin(h @ layer.weight)*.7
                losses.append(float((h**2).sum()))
            w[0,0]=old
        self.assertAlmostEqual(float(records[0][1][0][0,0]),(losses[1]-losses[0])/(2*eps),places=5)

    def test_reference_or_mode_mutation_rejected(self):
        for attribute in ('reference','training'):
            m=model();c=DecoderCheckpointing(m,expected_layers=2);c.set_enabled(True)
            loss=m.model.layers[0](torch.ones(1,2)).sum()
            setattr(m.model.layers[0],attribute,True)
            with self.assertRaisesRegex(ValueError,'changed before'):loss.backward()

    def test_no_grad_bypass_and_layout_cache_rejections(self):
        m=model();c=DecoderCheckpointing(m,expected_layers=2);c.set_enabled(True)
        with torch.no_grad():m.model.layers[0](torch.ones(1,2))
        self.assertEqual(c.forward_calls,0)
        with self.assertRaises(ValueError):DecoderCheckpointing(m,expected_layers=2)
        with self.assertRaises(ValueError):c.set_enabled(1)
        with self.assertRaises(ValueError):m.model.layers[0](torch.ones(1,2),use_cache=True)
        m.train()
        with self.assertRaises(ValueError):m.model.layers[0](torch.ones(1,2))

if __name__=='__main__':unittest.main(verbosity=2)
