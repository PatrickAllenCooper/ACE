"""New diagnostics/deadline fixtures, no scientific libraries or models."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def load(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class ReplayDiagnostics(unittest.TestCase):
    def setUp(self):
        self.q=load('qualify_delivery_replay_runtime')
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.fd=os.open(self.root,os.O_RDONLY)

    def tearDown(self):
        self.q.stop_diagnostics();os.close(self.fd);self.tmp.cleanup()

    def test_phase_records_are_exclusive_ordered_and_not_acceptance(self):
        self.q.diagnostic_phase(self.fd,'first_checkpoint_entered')
        self.q.diagnostic_phase(self.fd,'live_inventory_start',objects=12)
        a=json.loads((self.root/'runtime_phase_000.json').read_text())
        b=json.loads((self.root/'runtime_phase_001.json').read_text())
        self.assertTrue(a['diagnostic_only']);self.assertNotIn('qualified',a)
        self.assertEqual(b['details'],{'objects':12})
        self.assertGreaterEqual(b['monotonic'],a['monotonic'])
        self.assertGreaterEqual(b['process_cpu_seconds'],a['process_cpu_seconds'])
        self.q._PHASE_COUNTER=0
        with self.assertRaises(FileExistsError):self.q.diagnostic_phase(self.fd,'replacement')
        self.assertEqual(json.loads((self.root/'runtime_phase_000.json').read_text()),a)
        with self.assertRaises(ValueError):self.q.diagnostic_phase(self.fd,'../escape')

    def test_phase_symlink_and_stack_collision_do_not_overwrite(self):
        target=self.root/'target';target.write_text('preserve')
        (self.root/'runtime_phase_000.json').symlink_to(target)
        with self.assertRaises(OSError):self.q.diagnostic_phase(self.fd,'phase')
        (self.root/'runtime_stacks.log').symlink_to(target)
        with self.assertRaises(OSError):self.q.start_diagnostics(self.fd)
        self.assertEqual(target.read_text(),'preserve')

    def test_watchdog_is_periodic_nonterminating_and_owned_stream_closes(self):
        with patch.object(self.q.faulthandler,'dump_traceback_later') as start,patch.object(self.q.faulthandler,'cancel_dump_traceback_later') as stop:
            self.q.start_diagnostics(self.fd)
            stream=self.q._DIAGNOSTIC_STREAM
            start.assert_called_once_with(120,repeat=True,file=stream,exit=False)
            with self.assertRaises(ValueError):self.q.start_diagnostics(self.fd)
            self.q.stop_diagnostics();stop.assert_called_once();self.assertTrue(stream.closed)
            self.assertIsNone(self.q._DIAGNOSTIC_STREAM)

    def test_real_stdlib_watchdog_can_be_armed_and_cancelled(self):
        self.q.start_diagnostics(self.fd)
        self.assertTrue((self.root/'runtime_stacks.log').is_file())
        self.q.stop_diagnostics()
        self.assertIsNone(self.q._DIAGNOSTIC_STREAM)

    def test_watchdog_setup_failure_closes_stream(self):
        with patch.object(self.q.faulthandler,'dump_traceback_later',side_effect=RuntimeError('fixture')):
            with self.assertRaisesRegex(RuntimeError,'fixture'):self.q.start_diagnostics(self.fd)
        self.assertIsNone(self.q._DIAGNOSTIC_STREAM)
        self.assertFalse((self.root/'runtime_phase_000.json').exists())

    def test_exact_ninety_minute_cap_and_actual_allocation(self):
        s=load('supervise_delivery_qualified_replay')
        reg={'wall_seconds':5400,'slurm_wall_limit':'01:30:00','account':'ucb736_asc1','partition':'acpu','qos':'cpu-normal','rss_bytes':3*2**30}
        for module in (self.q,s):
            self.assertEqual(module.wall_time_policy(reg),5350)
            for seconds,label in [(5400,'01:00:00'),(5400.0,'01:30:00'),(5401,'01:30:01'),(7200,'02:00:00')]:
                with self.assertRaises(ValueError):module.wall_time_policy(dict(reg,wall_seconds=seconds,slurm_wall_limit=label))
        raw='JobId=123 Account=ucb736_asc1 Partition=acpu QOS=cpu-normal NumNodes=1 NumCPUs=1 NumTasks=1 CPUs/Task=1 Requeue=0 Restarts=0 TimeLimit=01:30:00 MinMemoryNode=3G AllocTRES=cpu=1,mem=3G,node=1 JobState=RUNNING'
        env={'SLURM_CPUS_PER_TASK':'1','SLURM_NTASKS':'1','SLURM_JOB_NUM_NODES':'1'}
        with patch.dict(os.environ,env),patch.object(s.subprocess,'check_output',return_value=raw):
            self.assertEqual(s.allocation('123',reg)['verified_fields']['TimeLimit'],'01:30:00')
        for changed in [raw.replace('TimeLimit=01:30:00','TimeLimit=01:00:00'),raw.replace('AllocTRES=cpu=1,mem=3G,node=1','AllocTRES=cpu=1,mem=3G,node=1,gres/gpu=1')]:
            with patch.dict(os.environ,env),patch.object(s.subprocess,'check_output',return_value=changed),self.assertRaises(ValueError):s.allocation('123',reg)


if __name__=='__main__':unittest.main(verbosity=2)
