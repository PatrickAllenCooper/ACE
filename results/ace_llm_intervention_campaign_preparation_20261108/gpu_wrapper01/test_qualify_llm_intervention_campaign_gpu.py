"""New saved-evidence/resource gates only. No subprocesses or models."""
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

import qualify_llm_intervention_campaign_gpu as gate


class ResourceHelper:
    @staticmethod
    def scheduler_fields(text):
        return dict(item.split('=',1) for item in text.split())
    @staticmethod
    def memory_mib(text):
        return int(text[:-1])*(1024 if text.endswith('G') else 1)


class CampaignGPUAdmissionTests(unittest.TestCase):
    def test_one_slice_exact_resources_and_negative_alternatives(self):
        text = ('JobId=123 UserId=paco0228(7) Account=ucb736_asc2 Partition=aa100 QOS=gpu-testing '
            'JobState=RUNNING NumNodes=1 NumTasks=1 NumCPUs=4 CPUs/Task=4 '
            'TimeLimit=00:10:00 Requeue=0 Restarts=0 '
            'AllocTRES=cpu=4,mem=16G,node=1,gres/gpu:a100_3g.20gb=1 '
            'ReqTRES=cpu=4,mem=16G,node=1,gres/gpu:a100_3g.20gb=1')
        self.assertEqual(gate.allocation(text,'123',ResourceHelper)['allocated_resources']['mem'],'16G')
        for old,new in [('16G','12G'),('TimeLimit=00:10:00','TimeLimit=00:20:00'),
            ('Account=ucb736_asc2','Account=other'),('Restarts=0','Restarts=1'),
            ('gres/gpu:a100_3g.20gb=1','gres/gpu:a100_3g.20gb=2'),
            ('cpu=4,mem=16G,node=1,gres/gpu:a100_3g.20gb=1',
             'cpu=4,mem=16G,node=1,gres/gpu:a100_3g.20gb=1,gres/gpu:a100=1')]:
            with self.subTest(new=new),self.assertRaises(ValueError):
                gate.allocation(text.replace(old,new),'123',ResourceHelper)

    def fixture(self, root):
        cpu = root/'cpu';campaign=cpu/'worker/artifacts/campaign'
        phase=cpu/'worker/artifacts/cpu-phase';campaign.mkdir(parents=True);phase.mkdir()
        pins={p:'1'*64 for p in gate.POLICIES[:5]}
        bundle={'campaign_sha256':'2'*64,'campaign':{'policy_ids':list(gate.POLICIES),
            'scope':'artificial','sources':[{},{}]}}
        terminal={'status':'cpu_phase_complete','failed_policies':[],'attempt_pins':pins,
            'campaign_sha256':'2'*64,'numeric_training_updates':6,'llm_training_updates':0,
            'gpu_services_started':0,'private_scores_computed':False}
        gate.save(phase/'bundle.json',bundle);gate.save(phase/'terminal.json',terminal)
        gate.save(campaign/'initial.json',{'initial':'fabricated'})
        excluded='worker/artifacts/campaign'
        a={'schema':'ace-paired-campaign-cpu-acceptance-v1','status':'artificial_cpu_phase_accepted',
            'cpu_packet_sha256':'3'*64,'job_id':'17','allocation':{'account':'ucb736_asc2','partition':'acpu',
            'qos':'cpu-normal','cpus':2,'memory_mib':7680,'gpus':0,'state':'COMPLETED','exit_code':'0:0'},
            'cpu_output':str(cpu),'campaign_root':str(campaign),'cpu_output_excluded_directory':excluded,
            'cpu_output_files':gate.inventory(cpu,exclude=excluded),'campaign_files':gate.inventory(campaign),
            'bundle_path':'worker/artifacts/cpu-phase/bundle.json',
            'bundle_sha256':gate.sha((phase/'bundle.json').read_bytes()),
            'phase_terminal_path':'worker/artifacts/cpu-phase/terminal.json',
            'phase_terminal_sha256':gate.sha((phase/'terminal.json').read_bytes()),
            'attempt_pins':pins,'scientific_qualified':False}
        p={'cpu_packet_sha256':'3'*64,'paths':{'cpu_output':str(cpu),'campaign':str(campaign)}}
        return a,p,bundle

    def test_exact_cpu_admission_and_partial_failure_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            a,p,b=self.fixture(Path(d))
            self.assertEqual(gate.admit_cpu(a,p,{}),b)
            for key,value in [('status','partial'),('scientific_qualified',True),
                              ('cpu_output_excluded_directory','worker')]:
                bad=dict(a,**{key:value})
                with self.subTest(key=key),self.assertRaises(ValueError):gate.admit_cpu(bad,p,{})
            bad=copy.deepcopy(a);bad['attempt_pins'].pop('learned_ranker')
            with self.assertRaises(ValueError):gate.admit_cpu(bad,p,{})

    def test_independent_cpu_bytes_and_once_only_campaign_gate(self):
        with tempfile.TemporaryDirectory() as d:
            a,p,b=self.fixture(Path(d));c=Path(p['paths']['campaign'])
            gate.save(c/'new.json',{'surplus':True})
            with self.assertRaisesRegex(ValueError,'campaign inventory'):gate.admit_cpu(a,p,{})
            # Updating a local membership list alone cannot authorize a second GPU attempt.
            (c/'policy-05').mkdir();a['campaign_files']=gate.inventory(c)
            with self.assertRaisesRegex(ValueError,'already started'):gate.admit_cpu(a,p,{})

    def test_no_follow_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'original').write_bytes(b'ok');(root/'alias').symlink_to(root/'original')
            with self.assertRaises(OSError):gate.read(root/'alias')
            with self.assertRaises(ValueError):gate.inventory(root)
        with self.assertRaisesRegex(ValueError,'duplicate'):gate.decode(b'{"a":1,"a":2}')
        with self.assertRaises(ValueError):gate.decode(b'{"a":NaN}')


if __name__=='__main__':unittest.main()
