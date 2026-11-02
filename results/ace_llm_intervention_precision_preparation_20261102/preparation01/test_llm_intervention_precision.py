"""Artificial local arithmetic checks, no pretrained model or GPU."""
import unittest
from types import SimpleNamespace
import torch
from llm_intervention_precision import FrozenFP32Projection, install_fp32_projection


def head():
    h=torch.nn.Linear(3, 4, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():h.weight.copy_(torch.arange(12).reshape(4,3)/16)
    h.requires_grad_(False)
    return h


class PrecisionChecks(unittest.TestCase):
    def test_explicit_fp32_forward_and_input_gradient_oracle(self):
        h=head(); p=FrozenFP32Projection(h)
        x=torch.tensor([[[.5,-.25,.75],[.25,.5,-.5]]],dtype=torch.bfloat16,requires_grad=True)
        expected=x.detach().float() @ h.weight.detach().float().T
        y=p(x); torch.testing.assert_close(y,expected,atol=0,rtol=0)
        y.sum().backward()
        torch.testing.assert_close(x.grad,h.weight.float().sum(0).to(torch.bfloat16).expand_as(x),atol=0,rtol=0)
        self.assertIsNone(h.weight.grad);self.assertFalse(p.weight.requires_grad)

    def test_tied_embedding_storage_and_original_parameter_preserved(self):
        h=head(); embedding=torch.nn.Embedding(4,3,dtype=torch.bfloat16);embedding.weight=h.weight
        model=SimpleNamespace(lm_head=h);before=embedding.weight.clone();pointer=embedding.weight.data_ptr()
        r=install_fp32_projection(model)
        self.assertTrue(r['original_parameter_retained']);self.assertTrue(r['independent_storage'])
        self.assertEqual(embedding.weight.data_ptr(),pointer);self.assertEqual(embedding.weight.dtype,torch.bfloat16)
        model.lm_head(torch.ones(1,1,3,dtype=torch.bfloat16))
        torch.testing.assert_close(embedding.weight,before,atol=0,rtol=0)
        with self.assertRaises(ValueError):install_fp32_projection(model)

    def test_autocast_does_not_reduce_projection_precision(self):
        p=FrozenFP32Projection(head());x=torch.ones(1,1,3,dtype=torch.bfloat16)
        with torch.autocast('cpu',dtype=torch.bfloat16):y=p(x)
        self.assertEqual(y.dtype,torch.float32)

    def test_invalid_weights_inputs_and_casts_rejected(self):
        h=head();h.requires_grad_(True)
        with self.assertRaises(ValueError):FrozenFP32Projection(h)
        with self.assertRaises(ValueError):FrozenFP32Projection(torch.nn.Linear(3,4,dtype=torch.bfloat16))
        h=head()
        with torch.no_grad():h.weight[0,0]=float('nan')
        with self.assertRaises(ValueError):FrozenFP32Projection(h)
        p=FrozenFP32Projection(head())
        with self.assertRaises(ValueError):p(torch.ones(1,1,3))
        p.to(torch.bfloat16)
        with self.assertRaises(ValueError):p(torch.ones(1,1,3,dtype=torch.bfloat16))

    def test_ten_minute_envelope_and_guard(self):
        import json
        from pathlib import Path
        from supervise_llm_intervention_precise_gpu import absolute_deadlines, authenticate_allocation
        self.assertEqual(absolute_deadlines(100.,110.),
            {'supervisor_end':670.,'worker_end':645.,'startup_end':290.})
        root=Path(__file__).resolve().parents[2]
        raw=json.loads((root/'results/ace_llm_intervention_task_runtime_20261102/gpu06/terminal/gpu06/scheduler_raw.json').read_text())['stdout']
        raw=raw.replace('TimeLimit=00:05:00','TimeLimit=00:10:00')
        resources={'account':'ucb736_asc1','partition':'aa100','qos':'gpu-testing','requested_cpus':4,'max_allocated_cpus':8,'memory_mib':12288,'wall_seconds':600,'gpus':1,'gpu_type':'a100_3g.20gb'}
        authenticate_allocation(raw,'35209536',resources)
        for old,new in [('TimeLimit=00:10:00','TimeLimit=00:20:00'),('Account=ucb736_asc1','Account=wrong'),('gres/gpu:a100_3g.20gb=1','gres/gpu:a100_3g.20gb=2')]:
            with self.assertRaises(ValueError):authenticate_allocation(raw.replace(old,new),'35209536',resources)

    def test_custom_loss_capture_observes_preference_gradient(self):
        from test_llm_intervention_gradient_probe import model_fixture
        from test_llm_intervention_action_scoring import fixture,loss_and_gradients
        from llm_intervention_gradient_probe import capture_backward
        from llm_intervention_learning import sequence_log_probs,preference_loss
        from llm_intervention_action_scoring import sequence_log_probs_selected
        import copy
        a=model_fixture();b=copy.deepcopy(a)
        def objective(model,batch,selected):
            score=(sequence_log_probs_selected if selected else sequence_log_probs)(model,batch)
            return preference_loss(score.reshape(1,2),torch.zeros(1,2),beta=.1)
        expected,gradients=loss_and_gradients(b,fixture(),lambda m,b:objective(m,b,True))
        observed=capture_backward(a,fixture(),selected=True,loss_function=objective)
        self.assertEqual(observed['loss'],float(expected))
        for name,g in gradients.items():torch.testing.assert_close(g,observed['gradients'][name],atol=0,rtol=0)
        self.assertIn('projection_input_gradients',observed)
        self.assertFalse(a.lm_head._forward_pre_hooks)

    def test_discrepancies_remain_complete_before_rejection(self):
        import tempfile,json
        from pathlib import Path
        from qualify_llm_intervention_precise_gpu import compare_and_record,finish_comparison_gate
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);records=[]
            a={str(i):torch.ones(1) for i in range(112)}
            b={str(i):torch.ones(1)*2 for i in range(112)}
            for stage in ('sft','dpo'):
                for left,right in (('full-0','full-1'),('selected-0','selected-1'),('full-0','selected-0'),('full-1','selected-1')):
                    records.append(compare_and_record(out,f'{stage}-{left}__{right}',[(1.,a),(1.,b)]))
            self.assertEqual(len(list(out.glob('comparison-*.json'))),8)
            with self.assertRaises(ValueError):finish_comparison_gate(out,records,['sft-full-0__full-1'])
            gate=json.loads((out/'comparison_gate.json').read_text())
            self.assertEqual(len(gate['tolerance_failures']),8)
            self.assertEqual(gate['optimizer_updates'],0);self.assertFalse(gate['passed'])

if __name__=='__main__':unittest.main(verbosity=2)
