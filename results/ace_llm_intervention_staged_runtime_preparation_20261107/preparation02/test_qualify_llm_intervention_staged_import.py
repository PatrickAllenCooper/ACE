"""Fabricated scheduler and origin checks, no imported runtime or model."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch, Mock
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

    def test_receipt_latency_does_not_extend_absolute_child_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            for i,delay in enumerate((20,186)):
                root=Path(d)/str(i);root.mkdir()
                clock=[100.]
                proc=Mock(pid=123)
                proc.wait.return_value=0
                cleanup={'group_still_exists':False,'errors':[],'returncode':0}
                kill=Mock(return_value=cleanup)
                original=q.write
                def write(path,value):
                    if path.name=='worker.started.json':clock[0]+=delay
                    original(path,value)
                with patch.object(q.time,'monotonic',side_effect=lambda:clock[0]), patch.object(q.signal,'setitimer') as timer, patch.object(q.subprocess,'Popen',return_value=proc), patch.object(q,'write',side_effect=write):
                    if delay==20:
                        self.assertEqual(q.run_bounded_child(['artificial'],{},root,1840,kill),(0,20,cleanup))
                        proc.wait.assert_called_once_with(timeout=165)
                    else:
                        with self.assertRaisesRegex(TimeoutError,'original child deadline'):
                            q.run_bounded_child(['artificial'],{},root,1840,kill)
                        proc.wait.assert_not_called()
                    self.assertEqual(timer.call_args_list[0].args,(q.signal.ITIMER_REAL,185))
                    self.assertEqual(timer.call_args_list[-1].args,(q.signal.ITIMER_REAL,1740-delay))
                kill.assert_called_once_with(proc)
                self.assertTrue((root/'worker.cleanup.json').is_file())
                self.assertFalse((root/'terminal.json').exists())

    def test_no_time_left_rejects_before_spawn(self):
        with tempfile.TemporaryDirectory() as d,patch.object(q.time,'monotonic',return_value=100),patch.object(q.subprocess,'Popen') as spawn:
            with self.assertRaisesRegex(TimeoutError,'insufficient'):
                q.run_bounded_child(['artificial'],{},Path(d),250,lambda p:None)
            spawn.assert_not_called()

if __name__=='__main__':unittest.main()
