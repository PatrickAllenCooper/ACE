import copy
import os
from pathlib import Path
import unittest

from delivery_prospective_design import structural_values, shared_actions, world_spec
from delivery_prospective_models import (runtime, initial_models, online_context,
    online_entry, original_update, normalizers, validate_input, predict_graph)

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get('ACE_DELIVERY_RUNNER_SOURCE', '/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-final-history-20261006/source'))


def fixture(size=5):
    spec = world_spec(size, 0, PROJECT)  # development fixture, never a confirmation seed
    actions = shared_actions(spec, 'random', 17, 60)
    rows = [{'query_index': i + 1, 'clamps': action, 'node_values': structural_values(spec, action)}
            for i, action in enumerate(actions)]
    return {'order': spec['order'], 'parents': spec['parents'], 'roots': list(actions[0]),
            'target': 'X3' if size == 5 else 'X30', 'rows': rows}


class ProspectiveModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch, _, _ = runtime(SOURCE)
        torch.set_num_threads(1)

    def test_original_update_parity_and_duplicate_buffer_winner(self):
        torch, Oracle, _ = runtime(SOURCE)
        data = fixture()
        a, oa, _ = initial_models(data, SOURCE, 0)
        b, ob, _ = initial_models(data, SOURCE, 0)
        ca, cb = online_context(data), online_context(data)
        for row in data['rows'][:50]:
            ca.replay_buffer.append(online_entry(data, row)); cb.replay_buffer.append(online_entry(data, row))
        for row in data['rows'][50:52]:
            entry = online_entry(data, row)
            original_update(ca, a, oa, entry, SOURCE, epochs=5)
            cb.replay_buffer.append(entry)
            Oracle._train_node_mlps_on(cb, b, ob, entry, 5)
        self.assertEqual([e['_query_index'] for e in ca.replay_buffer], list(range(3, 53)))
        for n in a:
            for key, value in a[n].state_dict().items():
                self.assertTrue(torch.equal(value, b[n].state_dict()[key]))
            for pa, pb in zip(a[n].parameters(), b[n].parameters()):
                self.assertTrue(torch.equal(oa[n].state[pa]['exp_avg'], ob[n].state[pb]['exp_avg']))

    def test_calibration_cannot_use_later_training_or_test_values(self):
        data = fixture(); original = normalizers(data)
        changed = copy.deepcopy(data)
        for row in changed['rows'][50:]:
            row['node_values']['X2'] = 1e8
        self.assertEqual(original, normalizers(changed))
        self.assertEqual(original['X2'], {'lo': [-3.], 'hi': [3.]})

    def test_only_joint_roots_can_be_directly_intervened(self):
        data = fixture(); data['rows'][0]['clamps']['X2'] = 1.
        with self.assertRaises(ValueError): validate_input(data)

    def test_predicted_parent_chain_and_intervened_node_error_zero(self):
        import torch
        class Affine(torch.nn.Module):
            def __init__(self, scale, offset): super().__init__(); self.scale=scale; self.offset=offset
            def forward(self, x): return x[:, 0] * self.scale + self.offset
        order=['R','M','Y']; parents={'R':[], 'M':['R'], 'Y':['M']}
        models={'M':Affine(2.,1.), 'Y':Affine(3.,-1.)}
        self.assertEqual(predict_graph(order, parents, models, {'R':2.}), {'R':2.,'M':5.,'Y':14.})
        self.assertEqual(predict_graph(order, parents, models, {'R':2.,'M':7.})['Y'], 20.)

    def test_thirty_node_adapter_agrees_with_audited_mechanisms(self):
        import torch
        import numpy as np
        from experiments.large_scale_scm import LargeScaleSCM
        spec=world_spec(30,0,PROJECT)
        state=np.random.get_state()
        try:
            np.random.seed(0); reference=LargeScaleSCM(n_nodes=30,coeff_seed=0)
        finally: np.random.set_state(state)
        reference.noise_std=0.
        clamps={n:(i-2.)/2 for i,n in enumerate(spec['order'][:5])}
        expected=structural_values(spec,clamps); values={}
        for node in spec['order']:
            values[node]=torch.tensor([clamps[node]],dtype=torch.float64) if node in clamps else reference.mechanisms(values,node)
            self.assertAlmostEqual(float(values[node].item()),expected[node],places=6)


if __name__ == '__main__': unittest.main()
