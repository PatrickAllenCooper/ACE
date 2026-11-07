"""Fabricated scheduler and origin checks, no imported runtime or model."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import qualify_llm_intervention_staged_import as q

class StagedImportTests(unittest.TestCase):
    def test_allocation_exact_and_seven_negative_fields(self):
        text='JobId=123 Account=ucb736_asc2 Partition=acpu QOS=cpu-normal NumCPUs=1 NumTasks=1 NumNodes=1 TimeLimit=00:30:00 Requeue=0 AllocTRES=cpu=1,mem=3G,node=1,billing=1'
        changes=[('ucb736_asc2','ucb736_asc1'),('NumCPUs=1','NumCPUs=2'),('mem=3G','mem=4G'),('Requeue=0','Requeue=1'),('TimeLimit=00:30:00','TimeLimit=01:00:00'),('node=1,billing=1','node=1,billing=1,gres/gpu:a100=1'),('NumTasks=1','NumTasks=2')]
        with tempfile.TemporaryDirectory() as d,patch.dict(q.os.environ,{'SLURM_JOB_ID':'123'}):
            root=Path(d)
            for i,s in enumerate([text]+[text.replace(a,b) for a,b in changes]):
                out=root/str(i);out.mkdir()
                with patch.object(q.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=s,stderr='')):
                    if i==0:self.assertEqual(q.allocation(out),'123')
                    else:
                        with self.assertRaises(ValueError):q.allocation(out)
                self.assertTrue((out/'allocation.json').is_file())

    def test_copied_prefix_must_be_actual_interpreter(self):
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(staged_prefix=d)
            probe=SimpleNamespace()
            with self.assertRaisesRegex(ValueError,'prefix'):
                q.copied_worker(args,probe)

    def test_pinned_capture_rejects_before_execution(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'module.py';p.write_text('raise RuntimeError("executed")\n')
            with self.assertRaisesRegex(ValueError,'pin differs'):
                q.pin_module(p,'0'*64,'artificial_not_imported')

if __name__=='__main__':unittest.main()
