"""New phase-clock delta only: extracted stdlib functions and fabricated boundaries."""
import ast
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import tempfile
import traceback
import types
import unittest

ROOT = Path(__file__).resolve().parent
CHILD = 'finalize_llm_intervention_scientific_campaign_cpu.py'


def definitions(filename, names, namespace=None):
    tree = ast.parse((ROOT/'implementation'/filename).read_bytes())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in selected} == set(names)
    env = dict(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<extracted fabricated delta>', 'exec'), env)
    return env


def constants(filename):
    # These assignments are stdlib values only; never execute the source module.
    env = {}
    wanted = {'VERSION', 'RESOURCES', 'DEADLINES', 'PREPARATION', 'REPORT_STARTUP', 'REPORT_STARTUP_CAP',
              'WORK', 'CLEANUP', 'HARD', 'FROZEN_DEADLINES', 'ENTRY_SHA', 'ENTRY', 'ROOT'}
    for n in ast.parse((ROOT/'implementation'/filename).read_bytes()).body:
        if isinstance(n, ast.Assign):
            keys = {v.id for t in n.targets for v in ast.walk(t) if isinstance(v, ast.Name)}
            if keys and keys <= wanted:
                exec(compile(ast.Module(body=[n], type_ignores=[]), '<literal clock constants>', 'exec'), env)
    return env


def child_env(extra=()):
    env = dict(constants(CHILD), math=math, json=json, hashlib=hashlib, re=re, Path=Path, os=os,
               contextlib=contextlib, traceback=traceback, sys=types.SimpleNamespace(platform='darwin', stderr=io.StringIO()),
               datetime=datetime, timezone=timezone)
    return definitions(CHILD, ['need','sha','pin','raw','decode','save','clock_deadlines',
        'arm_until','authentication_deadlines', *extra], env)


class Clock:
    def __init__(self, now=110):
        self.now, self.cpu, self.alarms = now, 0.0, []
    def monotonic(self): return self.now
    def process_time(self): return self.cpu
    def setitimer(self, which, remaining): self.alarms.append(remaining)
    def signal(self, *args): return 'previous'


class PhaseClockDelta(unittest.TestCase):
    def test_absolute_preparation_startup_and_owned_marker_boundaries(self):
        env = child_env(); clock = Clock()
        env['time'] = clock
        env['signal'] = types.SimpleNamespace(ITIMER_REAL=1, setitimer=clock.setitimer)
        plan = env['clock_deadlines'](100, 1299)
        self.assertEqual(plan, dict(preparation=1300, report_startup=1419, report_startup_cap=1420,
                                   work=2920, cleanup=2950, hard=3690))
        self.assertEqual(env['clock_deadlines'](100, 300)['report_startup'], 420)
        for value in (True, math.nan, math.inf, 99, 1300):
            with self.subTest(auth=value), self.assertRaises(ValueError):
                env['clock_deadlines'](100, value)
        for value in (True, math.nan, -1):
            with self.assertRaises(ValueError): env['clock_deadlines'](value)
        marker = dict(schema=env['VERSION'],packet_sha256='a'*64,pid=17,started_monotonic=100,monotonic=1299)
        self.assertEqual(env['authentication_deadlines'](marker,100,'a'*64,17),plan)
        for key,value in [('packet_sha256','b'*64),('pid',True),('pid',18),('started_monotonic',101),('schema','old')]:
            with self.assertRaises(ValueError):
                env['authentication_deadlines'](dict(marker,**{key:value}),100,'a'*64,17)
        env['arm_until'](1300); self.assertEqual(clock.alarms,[1190])
        clock.now=1300
        with self.assertRaises(ValueError): env['arm_until'](1300)
        self.assertEqual(clock.alarms,[1190])

    def test_parent_observes_two_markers_without_restarting_origin(self):
        for case in ('complete','missing_auth','late_auth','late_work','wrong_work_pin'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                out=Path(directory); (out/'worker').mkdir(); clock=Clock()
                env=child_env(('supervise',))
                env['time']=clock
                env['signal']=types.SimpleNamespace(SIGINT=2,SIGTERM=15,SIG_IGN=0,ITIMER_REAL=1,
                    signal=clock.signal,setitimer=clock.setitimer)
                env['resource']=types.SimpleNamespace(RUSAGE_CHILDREN=1,
                    getrusage=lambda _:types.SimpleNamespace(ru_utime=0,ru_stime=0,ru_maxrss=0))
                env['read']=lambda p:Path(p).read_bytes()
                proc=types.SimpleNamespace(pid=17,poll=lambda:0 if (out/'worker/work-start.json').exists() else None)
                env['subprocess']=types.SimpleNamespace(Popen=lambda *a,**k:proc)
                sleeps=[]
                def sleep(_):
                    sleeps.append(clock.now)
                    if len(sleeps)==1:
                        clock.now=1300 if case in ('late_auth','missing_auth') else 1299
                        if case=='missing_auth': return
                        env['save'](out/'worker/authentication-complete.json',dict(schema=env['VERSION'],
                            packet_sha256='a'*64,pid=17,started_monotonic=100,monotonic=clock.now))
                    else:
                        clock.now=1419 if case=='late_work' else 1418
                        captured=(out/'worker/authentication-complete.json').read_bytes()
                        env['save'](out/'worker/work-start.json',dict(packet_sha256='a'*64,pid=17,
                            monotonic=clock.now,report_startup_deadline=1419,
                            authentication_sha256='b'*64 if case=='wrong_work_pin' else env['sha'](captured)))
                clock.sleep=sleep
                owned=types.SimpleNamespace(process=None,confirm_closed=lambda:True)
                kills=[]
                def kill(p):
                    self.assertIs(p,proc);kills.append(p)
                    return dict(group_still_exists=False,errors=[],returncode=0,kill_sent=False)
                if case=='complete':
                    env['supervise']([],{},out,100,'a'*64,kill,owned)
                    self.assertEqual(clock.alarms[:3],[1190,120,1502])
                else:
                    with self.assertRaises(ValueError): env['supervise']([],{},out,100,'a'*64,kill,owned)
                self.assertEqual(len(kills),1)
                record=json.loads((out/'worker.resources.json').read_bytes())
                self.assertEqual(record['cleanup_deadline'],2950)
                self.assertEqual(record['postcheck_deadline'],3690)
                self.assertEqual(record['first_failure'] is None,case=='complete')

    def test_phase_records_elapsed_cpu_and_preserves_primary_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            env=child_env(('PhaseTimings',));clock=Clock();env['time']=clock
            timing=env['PhaseTimings'](directory,100,'parent')
            with timing.phase('full_authentication') as details:
                clock.now+=25;clock.cpu+=3;details['files']=2
            record=json.loads((Path(directory)/'timing-000.end.json').read_bytes())
            self.assertEqual((record['elapsed_seconds'],record['process_cpu_seconds'],record['finished_elapsed_seconds']),(25,3,35))
            self.assertEqual(record['details'],{'files':2})
            with self.assertRaisesRegex(TimeoutError,'phase expired'):
                with timing.phase('inventory_gpu_output'):
                    clock.now+=1;clock.cpu+=0.25;raise TimeoutError('phase expired')
            record=json.loads((Path(directory)/'timing-001.end.json').read_bytes())
            self.assertFalse(record['completed']);self.assertEqual(record['failure']['type'],'TimeoutError')
            original=env['save']
            def publication(path,value):
                if str(path).endswith('.end.json'):raise OSError('publication failed')
                return original(path,value)
            env['save']=publication
            with self.assertRaisesRegex(ValueError,'first failure'):
                with timing.phase('postchecks'):raise ValueError('first failure')
            self.assertTrue((Path(directory)/'timing-002.start.json').exists())
            self.assertIn('publication failed',env['sys'].stderr.getvalue())

    def test_operational_adoption_profile_budget_and_scientific_gate_delta(self):
        env=dict(constants('build_packet.py'),json=json,shlex=shlex,sha=lambda b:hashlib.sha256(b).hexdigest())
        env=definitions('build_packet.py',['need','raw','clock_admission','budget','job_script'],env)
        self.assertEqual(env['RESOURCES']['wall_seconds'],3600)
        self.assertEqual(env['DEADLINES'],constants(CHILD)['DEADLINES'])
        self.assertEqual(env['FROZEN_DEADLINES'],dict(startup=120,work=1140,cleanup=1170,hard=1190))
        self.assertEqual(env['ENTRY_SHA'],hashlib.sha256((ROOT/'implementation'/CHILD).read_bytes()).hexdigest())
        script=env['job_script']({'metadata':{'cpu-packet':{'runtime_source':'/fabricated/runtime'}}},{},'pass')
        self.assertIn(b'#SBATCH --time=01:00:00',script)
        self.assertEqual(constants('pull_terminal.py')['RESOURCES'],env['RESOURCES'])
        roles=('freeze','finalizer-source-manifest','finalizer-source-review','cpu-acceptance','gpu-acceptance',
               'gpu-recovery-acceptance','gpu-baseline','finalizer-timing-evidence','finalizer-timing-review')
        pins={key:hashlib.sha256(key.encode()).hexdigest() for key in roles}
        adoption=dict(schema='ace-finalizer-clock-adoption-v1',status='adopted',scope='operational_finalization',
            remote_root=env['ROOT'],freeze_sha256=pins['freeze'],source_commit='1'*40,
            source_manifest_sha256=pins['finalizer-source-manifest'],source_review_sha256=pins['finalizer-source-review'],
            entry_sha256=env['ENTRY_SHA'],predecessor_entry_sha256='b4d18f4be2320069ffc5f2aed2060efcfb8d49bfcc6e8b00f8099b91598f8745',
            original_deadlines=env['FROZEN_DEADLINES'],original_wall_seconds=1200,
            resources=env['RESOURCES'],deadlines=env['DEADLINES'],cpu_acceptance_sha256=pins['cpu-acceptance'],
            gpu_acceptance_sha256=pins['gpu-acceptance'],gpu_recovery_acceptance_sha256=pins['gpu-recovery-acceptance'],
            baseline_sha256=pins['gpu-baseline'],timing_evidence_sha256=pins['finalizer-timing-evidence'],
            timing_review_sha256=pins['finalizer-timing-review'],full_path_timing_supported=True,
            required_findings=0,scientific_inputs_unchanged=True,scientific_qualified=False,execution_qualified=False)
        metadata={'finalizer-clock-adoption':adoption,'finalizer-source-manifest':{'source_commit':'1'*40}}
        env['clock_admission'](metadata,pins)
        child=child_env(('admit_clock',))
        baseline=b'{}\n';authority=dict(adoption,baseline_sha256=child['sha'](baseline))
        authority_raw=child['raw'](authority)
        packet=dict(clock_adoption_sha256=child['sha'](authority_raw),files={child['ENTRY']:env['ENTRY_SHA']},
            operational_source_commit='1'*40,freeze_sha256=pins['freeze'],cpu_acceptance_sha256=pins['cpu-acceptance'],
            gpu_acceptance_sha256=pins['gpu-acceptance'],gpu_recovery_acceptance_sha256=pins['gpu-recovery-acceptance'])
        data={'finalizer-clock-adoption.json':authority_raw,'gpu-success-baseline.json':baseline}
        child['admit_clock'](packet,data)
        with self.assertRaises(ValueError):child['admit_clock'](dict(packet,clock_adoption_sha256='0'*64),data)
        for key,value in [('status','prospective'),('full_path_timing_supported',False),('required_findings',False),
                          ('timing_evidence_sha256','0'*64),('baseline_sha256','0'*64)]:
            changed=copy.deepcopy(metadata);changed['finalizer-clock-adoption'][key]=value
            with self.assertRaises(ValueError): env['clock_admission'](changed,pins)
        env['cost']=lambda rows:(40000,3000);env['normalized']=lambda rows:rows
        base=dict(terminal_jobs=48,submission_journals=98,outstanding_reservation=0,related_accounting=[None]*48,
            journal_pins={str(i):'a'*64 for i in range(98)},closed_cpu_seconds=40000,closed_gpu_slice_seconds=3000,
            preparation_allowance_cpu_seconds=3600,cpu_admission_seconds=43600)
        result=env['budget'](base)
        self.assertEqual((result['candidate_cpu_reservation'],result['total_cpu_seconds']),(7200,50800))
        env['cost']=lambda rows:(46801,3000)
        with self.assertRaises(ValueError):env['budget'](dict(base,closed_cpu_seconds=46801,cpu_admission_seconds=50401))
        old=ast.parse((ROOT/'predecessor'/CHILD).read_bytes());new=ast.parse((ROOT/'implementation'/CHILD).read_bytes())
        nodes=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        a,b=nodes(old),nodes(new)
        for name in ('read','inventory','checked_inventory','admit_evidence','admit_recovery','runtime_record',
                     'validate_report','upstream_shape','OwnedChild'):
            self.assertEqual(a[name],b[name],name)
        def private_calls(tree):
            return [ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Call)
                    and isinstance(n.func,ast.Attribute) and n.func.attr=='finalize_campaign']
        self.assertEqual(private_calls(old),private_calls(new))


if __name__ == '__main__':
    unittest.main(verbosity=2)
