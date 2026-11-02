"""New artificial observation and shortened-supervisor checks, no pretrained model."""
import copy,json
from pathlib import Path
import unittest
import torch
from test_llm_intervention_action_scoring import CausalToy,fixture,loss_and_gradients
from llm_intervention_learning import install_adapters,supervised_loss
from llm_intervention_gradient_probe import capture_backward,compare_captures
from supervise_llm_intervention_gradients import authenticate_allocation,absolute_deadlines


def model_fixture():
    m=CausalToy()
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(390).get_state());install_adapters(m,rank=2,alpha=4.)
    with torch.no_grad():m.q_proj.B.fill_(.02);m.v_proj.B.fill_(.03)
    return m

class ProbeChecks(unittest.TestCase):
    def test_observer_preserves_loss_gradients_weights_and_removes_hooks(self):
        a=model_fixture();b=copy.deepcopy(a);before={n:v.clone() for n,v in a.state_dict().items()}
        value,gradients=loss_and_gradients(b,fixture(),supervised_loss)
        observed=capture_backward(a,fixture(),selected=False)
        self.assertEqual(observed['loss'],float(value));self.assertEqual(set(observed['gradients']),set(gradients))
        for n,g in gradients.items():torch.testing.assert_close(observed['gradients'][n],g,atol=0,rtol=0)
        for n,v in a.state_dict().items():torch.testing.assert_close(v,before[n],atol=0,rtol=0)
        self.assertFalse(a.lm_head._forward_pre_hooks);self.assertTrue(all(p.grad is None for p in a.parameters()))
    def test_selected_projection_boundary_alignment(self):
        m=model_fixture();a=capture_backward(m,fixture(),selected=False);b=capture_backward(m,fixture(),selected=True)
        r=compare_captures(a,b,atol=2e-7,rtol=2e-5)
        self.assertEqual(r['failed_gradient_tensors'],0);self.assertTrue(r['projection_inputs']['bitwise_equal'])
        self.assertTrue(r['projection_input_gradients']['within_tolerance'])
        self.assertEqual(a['projection_inputs'].shape[1],9);self.assertEqual(b['projection_inputs'].shape[1],3)
    def test_mismatches_are_retained_and_not_rejected(self):
        m=model_fixture();a=capture_backward(m,fixture(),selected=True);b=copy.deepcopy(a)
        first=next(iter(b['gradients']));b['gradients'][first][0,0]+=1
        r=compare_captures(a,b);self.assertEqual(r['failed_gradient_tensors'],1)
        self.assertAlmostEqual(r['max_gradient_absolute_difference'],1.,places=6)
        b['positions']=b['positions']+1
        with self.assertRaises(ValueError):compare_captures(a,b)
    def test_failed_forward_removes_hooks(self):
        m=model_fixture()
        with torch.no_grad():m.lm_head.weight[0,0]=float('nan')
        with self.assertRaises(ValueError):capture_backward(m,fixture(),selected=True)
        self.assertFalse(m.lm_head._forward_pre_hooks);self.assertTrue(all(p.grad is None for p in m.parameters()))
    def test_five_minute_deadlines_and_allocation(self):
        self.assertEqual(absolute_deadlines(100.,110.),{'supervisor_end':380.,'worker_end':355.,'startup_end':290.})
        base=Path(__file__).resolve().parents[2]
        raw=json.loads((base/'results/ace_llm_intervention_task_runtime_20261102/gpu05/terminal/gpu05/scheduler_raw.json').read_bytes())['stdout']
        raw=raw.replace('TimeLimit=00:15:00','TimeLimit=00:05:00')
        resources={'account':'ucb736_asc1','partition':'aa100','qos':'gpu-testing','requested_cpus':4,'max_allocated_cpus':8,'memory_mib':12288,'wall_seconds':300,'gpus':1,'gpu_type':'a100_3g.20gb'}
        authenticate_allocation(raw,'35208932',resources)
        for old,new in [('TimeLimit=00:05:00','TimeLimit=00:15:00'),('Account=ucb736_asc1','Account=wrong'),('gres/gpu:a100_3g.20gb=1','gres/gpu:a100_3g.20gb=2')]:
            with self.assertRaises(ValueError):authenticate_allocation(raw.replace(old,new),'35208932',resources)

if __name__=='__main__':unittest.main(verbosity=2)
