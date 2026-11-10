"""One new numeric-profile check; no original source main or scientific calls."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import types
import unittest

ROOT=Path(__file__).resolve().parent
ENTRY='finalize_llm_intervention_scientific_campaign_cpu.py'


def extracted(filename, functions, constants):
    nodes=[]
    for n in ast.parse((ROOT/'implementation'/filename).read_bytes()).body:
        if isinstance(n,ast.Assign):
            names={x.id for t in n.targets for x in ast.walk(t) if isinstance(x,ast.Name)}
            if names and names<=set(constants):nodes.append(n)
        elif isinstance(n,ast.FunctionDef) and n.name in functions:nodes.append(n)
    env=dict(math=math,re=re,json=json,shlex=shlex,os=types.SimpleNamespace(environ={}),
             sha=lambda b:hashlib.sha256(b).hexdigest())
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pure numeric profile delta>','exec'),env)
    return env


class ProfileDelta(unittest.TestCase):
    def test_4200_profile_boundaries_and_all_outer_propagation(self):
        numeric=json.loads((ROOT/'numeric-delta.json').read_bytes())
        # Exact permitted source changes: no logic or history refactor is hidden.
        for before in sorted((ROOT/'predecessor').glob('*.py')):
            text=before.read_text()
            for change in numeric.get(before.name,[]):
                self.assertEqual(text.count(change['before']),change['occurrences'])
                text=text.replace(change['before'],change['after'])
            self.assertEqual(text,(ROOT/'implementation'/before.name).read_text(),before.name)
        h=extracted(ENTRY,{'need','clock_deadlines','allocation'},
            {'RESOURCES','PREPARATION','REPORT_STARTUP','REPORT_STARTUP_CAP','WORK','CLEANUP','HARD','DEADLINES'})
        expected=dict(preparation=1500,report_startup=120,report_startup_cap=1620,work=3320,cleanup=3350,hard=4190)
        self.assertEqual(h['DEADLINES'],expected)
        self.assertEqual(h['clock_deadlines'](100,1599),dict(preparation=1600,report_startup=1719,
            report_startup_cap=1720,work=3420,cleanup=3450,hard=4290))
        self.assertEqual(h['clock_deadlines'](100,1400)['report_startup'],1520) # Newly allowed preparation range.
        self.assertEqual(h['clock_deadlines'](100,300)['report_startup'],420) # Relative120 unchanged.
        with self.assertRaises(ValueError):h['clock_deadlines'](100,1600)
        scheduler='JobId=17 JobState=RUNNING Account=ucb736_asc2 Partition=acpu QOS=cpu-normal NumCPUs=2 NumTasks=1 NumNodes=1 TimeLimit=01:10:00 Requeue=0 Restarts=0 UserId=paco0228(17) AllocTRES=cpu=2,node=1,mem=7680M ReqTRES=cpu=2,node=1,mem=7680M'
        self.assertEqual(h['allocation'](scheduler,'17')['TimeLimit'],'01:10:00')
        with self.assertRaises(ValueError):h['allocation'](scheduler.replace('01:10:00','01:00:00'),'17')
        b=extracted('build_packet.py',{'need','raw','budget','job_script'},
                    {'ENTRY','ENTRY_SHA','ROOT','RESOURCES','DEADLINES','FROZEN_DEADLINES','GPU_LIMITS'})
        c=extracted('pull_terminal.py',set(),{'RESOURCES'})
        self.assertEqual(b['RESOURCES'],h['RESOURCES']);self.assertEqual(c['RESOURCES'],h['RESOURCES'])
        self.assertEqual(h['RESOURCES']['wall_seconds'],4200);self.assertEqual(b['DEADLINES'],expected)
        self.assertEqual(b['FROZEN_DEADLINES'],dict(startup=120,work=1140,cleanup=1170,hard=1190))
        self.assertEqual(b['GPU_LIMITS'],dict(session_reserve_seconds=600,session_cap_seconds=1200,work_seconds=2820,final_seconds=2860))
        self.assertEqual(b['ENTRY_SHA'],hashlib.sha256((ROOT/'implementation'/ENTRY).read_bytes()).hexdigest())
        script=b['job_script']({'metadata':{'cpu-packet':{'runtime_source':'/fabricated/runtime'}}},{},'pass')
        self.assertIn(b'#SBATCH --time=01:10:00',script);self.assertNotIn(b'#SBATCH --time=01:00:00',script)
        b['normalized']=lambda rows:rows
        for cpu,passes in ((37073,True),(45600,True),(45601,False)):
            b['cost']=lambda rows,cpu=cpu:(cpu,4458)
            base=dict(terminal_jobs=48,submission_journals=98,outstanding_reservation=0,
                related_accounting=[None]*48,journal_pins={str(i):'a'*64 for i in range(98)},
                closed_cpu_seconds=cpu,closed_gpu_slice_seconds=4458,preparation_allowance_cpu_seconds=3600,
                cpu_admission_seconds=cpu+3600)
            if passes:
                result=b['budget'](base);self.assertEqual(result['candidate_cpu_reservation'],8400)
                self.assertEqual(result['total_cpu_seconds'],cpu+12000)
            else:
                with self.assertRaises(ValueError):b['budget'](base)
        launcher=(ROOT/'implementation/launch_once.py').read_text()
        self.assertIn("duration(fields.get('MaxTime',''))>=4200",launcher)
        self.assertIn('duration(qr[0][1])>=4200',launcher)
        self.assertEqual(launcher.count("C['baseline']['closed_cpu_seconds']+12000"),2)
        self.assertIn('signal.alarm(1200)',launcher) # Metadata-only transport cap retained.
        collector=(ROOT/'implementation/pull_terminal.py').read_text()
        self.assertIn("[job,'acpu','cpu-normal','70']",collector)
        self.assertIn('0<=elapsed<=4200',collector)
        self.assertIn('len(expected)==100',collector)
        self.assertIn('len(prior)==48',collector)


if __name__=='__main__':unittest.main(verbosity=2)
