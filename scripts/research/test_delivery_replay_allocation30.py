"""New 30-minute proposal's actual allocation parser; fabricated scheduler only."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch


spec=importlib.util.spec_from_file_location('proposed_supervisor',
    Path(__file__).with_name('supervise_delivery_qualified_replay.py'))
supervisor=importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


class Allocation30(unittest.TestCase):
    def check(self,limit):
        raw=('JobId=123 Account=ucb736_asc1 Partition=acpu QOS=cpu-normal '
             'NumNodes=1 NumCPUs=1 NumTasks=1 CPUs/Task=1 Requeue=0 Restarts=0 '
             'TimeLimit='+limit+' MinMemoryNode=3G AllocTRES=cpu=1,mem=3G,node=1 '
             'JobState=RUNNING')
        reg={'account':'ucb736_asc1','partition':'acpu','qos':'cpu-normal',
             'rss_bytes':3*2**30,'wall_seconds':1800,'slurm_wall_limit':'00:30:00'}
        env={'SLURM_CPUS_PER_TASK':'1','SLURM_NTASKS':'1','SLURM_JOB_NUM_NODES':'1'}
        with patch.dict(os.environ,env),patch.object(supervisor.subprocess,'check_output',return_value=raw):
            return supervisor.allocation('123',reg)

    def test_exact_thirty_minute_allocation(self):
        result=self.check('00:30:00')
        self.assertEqual(result['verified_fields']['TimeLimit'],'00:30:00')
        self.assertEqual(result['verified_fields']['AllocTRES'],'cpu=1,mem=3G,node=1')

    def test_old_or_over_cap_allocation_rejected(self):
        for limit in ('00:15:00','00:30:01','01:00:00'):
            with self.subTest(limit=limit),self.assertRaisesRegex(ValueError,'wall limit'):
                self.check(limit)


if __name__=='__main__':unittest.main(verbosity=2)
