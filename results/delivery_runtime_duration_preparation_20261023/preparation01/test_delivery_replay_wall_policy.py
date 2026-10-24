"""New measured-duration proposal checks; no target runtime or model imports."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch


def load(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


s=load('supervise_delivery_qualified_replay')
q=load('qualify_delivery_replay_runtime')


class WallPolicy(unittest.TestCase):
    def test_exact_caps_and_cleanup_margin(self):
        for seconds,label,deadline in [(1800,'00:30:00',1750),(3600,'01:00:00',3550)]:
            for module in (s,q):
                self.assertEqual(module.wall_time_policy({'wall_seconds':seconds,'slurm_wall_limit':label}),deadline)

    def test_mismatch_missing_type_and_over_cap_rejected(self):
        invalid=[{}, {'wall_seconds':3600}, {'slurm_wall_limit':'01:00:00'},
                 {'wall_seconds':3600,'slurm_wall_limit':'00:30:00'},
                 {'wall_seconds':1800,'slurm_wall_limit':'01:00:00'},
                 {'wall_seconds':3601,'slurm_wall_limit':'01:00:01'},
                 {'wall_seconds':7200,'slurm_wall_limit':'02:00:00'},
                 {'wall_seconds':3600.0,'slurm_wall_limit':'01:00:00'},
                 {'wall_seconds':True,'slurm_wall_limit':'01:00:00'}]
        for reg in invalid:
            for module in (s,q):
                with self.subTest(reg=reg,module=module.__name__),self.assertRaises(ValueError):
                    module.wall_time_policy(reg)

    def allocation(self,seconds,label,actual,extra=''):
        reg={'account':'ucb736_asc1','partition':'acpu','qos':'cpu-normal',
             'rss_bytes':3*2**30,'wall_seconds':seconds,'slurm_wall_limit':label}
        raw=('JobId=123 Account=ucb736_asc1 Partition=acpu QOS=cpu-normal '
             'NumNodes=1 NumCPUs=1 NumTasks=1 CPUs/Task=1 Requeue=0 Restarts=0 '
             'TimeLimit='+actual+' MinMemoryNode=3G AllocTRES=cpu=1,mem=3G,node=1'+extra+' JobState=RUNNING')
        env={'SLURM_CPUS_PER_TASK':'1','SLURM_NTASKS':'1','SLURM_JOB_NUM_NODES':'1'}
        with patch.dict(os.environ,env),patch.object(s.subprocess,'check_output',return_value=raw):
            return s.allocation('123',reg)

    def test_actual_allocation_must_equal_frozen_cap(self):
        for seconds,label in [(1800,'00:30:00'),(3600,'01:00:00')]:
            self.assertEqual(self.allocation(seconds,label,label)['verified_fields']['TimeLimit'],label)
            for actual in ['00:15:00','00:30:00','00:30:01','01:00:00','01:00:01']:
                if actual!=label:
                    with self.subTest(label=label,actual=actual),self.assertRaisesRegex(ValueError,'wall limit'):
                        self.allocation(seconds,label,actual)

    def test_longer_time_does_not_admit_accelerators(self):
        with self.assertRaisesRegex(ValueError,'accelerator'):
            self.allocation(3600,'01:00:00','01:00:00',',gres/gpu=1')


if __name__=='__main__':unittest.main(verbosity=2)
