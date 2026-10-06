import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from delivery_prospective_io import collect_history,deny_evaluation_reads,evaluate_states
from delivery_prospective_design import world_spec,shared_actions,structural_values
from runner_delivery_confirmation import journal_count,read
from test_delivery_prospective_models import SOURCE, PROJECT, fixture
from delivery_prospective_models import runtime


class ProspectiveIOTests(unittest.TestCase):
    def test_failed_generation_is_charged_and_cannot_replace_attempt(self):
        spec=world_spec(5,0,PROJECT);actions=shared_actions(spec,'random',17,400)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'history'
            with patch('delivery_prospective_io.structural_values',side_effect=RuntimeError('analytic failed response')):
                with self.assertRaises(RuntimeError):collect_history(spec,actions,out)
            self.assertEqual(journal_count(out/'queries.ndjson'),1)
            self.assertEqual(read(out/'failure.json')['rows_completed'],0)
            self.assertFalse((out/'receipt.json').exists())
            with self.assertRaises(FileExistsError):collect_history(spec,actions,out)

    def test_heldout_action_cannot_enter_collection(self):
        spec=world_spec(5,0,PROJECT);actions=shared_actions(spec,'evaluation',18,400)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'history'
            with self.assertRaises(ValueError):collect_history(spec,actions,out)
            self.assertFalse(out.exists())

    def test_fit_cannot_read_test_responses_or_world_coefficients(self):
        for name in ('/study/evaluation/input.json','/study/world.json','/study/actions.json','/study/scores.json'):
            with self.assertRaises(PermissionError):deny_evaluation_reads('open',(name,'r',0))
        deny_evaluation_reads('open',('/study/train/input.json','r',0))

    def test_vectorized_evaluator_uses_predicted_parents_and_training_variance(self):
        import numpy as np
        torch,_,MLP=runtime(SOURCE);torch.set_num_threads(1)
        training=fixture();spec=world_spec(5,0,PROJECT)
        evaluation={k:v for k,v in training.items() if k!='rows'}
        evaluation['rows']=[{'query_index':i+1,'clamps':a,'node_values':structural_values(spec,a)}
                            for i,a in enumerate(shared_actions(spec,'evaluation',18,60))]
        states={}
        for node,pa in spec['parents'].items():
            if not pa:continue
            model=MLP(len(pa))
            with torch.no_grad():
                for parameter in model.parameters():parameter.zero_()
                if node=='X3':
                    model.net[0].weight[0,1]=1.;model.net[0].bias[0]=100.
                    model.net[2].weight[0,0]=1.;model.net[4].weight[0,0]=1.;model.net[4].bias[0]=-100.
                else:model.net[4].bias[0]=4. if node=='X2' else .5
            states[node]=model.state_dict()
        score,predicted=evaluate_states(training,evaluation,states,SOURCE,'delivery')
        self.assertTrue(np.all(predicted['X3']==4.))
        changed=copy.deepcopy(evaluation)
        for row in changed['rows']:row['node_values']['X2']=999.
        score_changed,predicted_changed=evaluate_states(training,changed,states,SOURCE,'delivery')
        self.assertTrue(np.array_equal(predicted_changed['X3'],predicted['X3']))
        self.assertEqual(score_changed['nmse'],score['nmse'])
        self.assertAlmostEqual(score['training_target_variance'],float(np.var([r['node_values']['X3'] for r in training['rows']])))
        self.assertIsNone(score['snapped_error'])


if __name__=='__main__':unittest.main()
