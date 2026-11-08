"""New timing/telemetry boundary tests, extracted stdlib functions and fake GPU data.

No campaign, environment, model, GPU process, response or numerical import runs.
The old supervision suite is not repeated. AST equality checks preserve its
process ownership and cleanup code across this scientific-profile derivative.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import threading
import time
import types
import unittest

HERE=Path(__file__).parent
SOURCE=HERE/'llm_intervention_scientific_gpu_phase.py'
OLD=HERE/'llm_intervention_campaign_gpu_phase.py'
TREE=ast.parse(SOURCE.read_text())
SELECTED={'GPUPhaseInterrupted','_need','_read','_save','_gpu_active','_activity','_Watch','_scientific_limits'}
ENV=dict(os=os,Path=Path,stat=stat,threading=threading,time=time,datetime=datetime,timezone=timezone,
    MAX_REQUESTS=576,FRESHNESS=90.,finite=lambda x:type(x) in (int,float) and math.isfinite(x),
    integer=lambda x,lo=0:type(x) is int and x>=lo,
    _raw=lambda x:(json.dumps(x,sort_keys=True,allow_nan=False)+'\n').encode(),
    _sha=lambda x:hashlib.sha256(x).hexdigest(),campaign_api=types.SimpleNamespace(_decode=json.loads))
exec(compile(ast.Module(body=[n for n in TREE.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
    and n.name in SELECTED],type_ignores=[]),str(SOURCE),'exec'),ENV)


class ScientificPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output=Path(os.environ['ACE_SCIENTIFIC_PHASE_TEST_OUTPUT']);cls.output.mkdir()

    def test_explicit_bounded_profiles(self):
        for reserve,cap,work,final in [(600,1000,2620,2670),(800,1200,3540,3570)]:
            ENV['_scientific_limits'](dict(session_reserve_seconds=reserve,session_cap_seconds=cap,
                work_seconds=work,final_seconds=final),10.,50.,10.+work,10.+final)

    def test_invalid_limits_fail_before_execution(self):
        base=dict(session_reserve_seconds=600,session_cap_seconds=1000,work_seconds=2620,final_seconds=2670)
        cases=[{},dict(base,extra=1)]
        cases.extend(dict(base,**{k:v}) for k,v in [('session_reserve_seconds',119),('session_cap_seconds',1201),
            ('session_cap_seconds',599),('session_reserve_seconds',True),('work_seconds',3541),
            ('work_seconds',240),('final_seconds',3571),('final_seconds',2639)])
        for bad in cases:
            with self.subTest(bad=bad),self.assertRaises(ValueError):ENV['_scientific_limits'](bad,10.,50.,2630.,2680.)

    def test_inherited_deadlines_and_remaining_reservations(self):
        limits=dict(session_reserve_seconds=600,session_cap_seconds=1000,work_seconds=2620,final_seconds=2670)
        for times in [(10.,9.,2630.,2680.),(10.,900.,2630.,2680.),(10.,50.,2631.,2680.),
                      (10.,50.,2630.,2681.),(10.,float('nan'),2630.,2680.)]:
            with self.subTest(times=times),self.assertRaises(ValueError):ENV['_scientific_limits'](limits,*times)

    def watcher(self, name):
        root=self.output/name;root.mkdir();worker=root/'worker';worker.mkdir();obs=root/'observations';obs.mkdir()
        now=time.monotonic()
        w=ENV['_Watch'](pid='frozen_llm',worker=worker,attempt=root/'absent-attempt',output=obs,
            outer_started=now,started=now,deadline=now+100,sample=lambda pid:{'returncode':0,'owned_rows':[f'{pid}, fake-gpu, 128']},worlds=24)
        w.policy=types.SimpleNamespace(process=types.SimpleNamespace(pid=123456,poll=lambda:None))
        return w

    def generation(self,w,n):
        p=w.worker/f'generation-{n:04d}.json'
        p.write_text(json.dumps(dict(ordinal=n,cuda_interval_ms=1.,peak_gpu_allocated_bytes=100,
            new_training_updates=0,tensor_versions_unchanged=True,environment_callbacks_available=False,
            prompt_ids=[1],sequence_ids=[1,2]))+'\n')

    def test_progress_continues_beyond_artificial_eight_request_limit(self):
        w=self.watcher('nine-positive')
        for n in range(9):self.generation(w,n)
        w.observe()
        self.assertEqual(len(w.generations),9)
        self.assertIsNotNone(w.first_verified)
        self.assertIsNone(w.fault)
        self.assertFalse(w.thread.is_alive())

    def test_last_permitted_request_and_overflow_uncertainty(self):
        w=self.watcher('last-request');w.generations={n:{'sha256':'fixture','cuda_interval_ms':1.,'peak_gpu_allocated_bytes':100} for n in range(575)}
        self.generation(w,575);w.observe();self.assertEqual(len(w.generations),576)
        self.generation(w,576);w.observe();self.assertEqual(len(w.generations),576)
        saved=json.loads((w.output/'telemetry-00001.json').read_text())
        self.assertTrue(any('generation-0576.json' in s for s in saved['uncertainties']))
        self.assertFalse(w.thread.is_alive())

    def test_qualified_ownership_and_cleanup_code_preserved(self):
        old=ast.parse(OLD.read_text())
        def top(t,n):return next(v for v in t.body if isinstance(v,(ast.ClassDef,ast.FunctionDef)) and v.name==n)
        for name in ('_hold_owned_cleanup','_activity','_callbacks','_gpu_active','_gpu_empty'):
            self.assertEqual(ast.dump(top(old,name)),ast.dump(top(TREE,name)))
        for name in ('arm_cleanup','check','interrupt','factory','retain'):
            def nested(t):return next(v for v in top(t,'run_gpu_phase').body if isinstance(v,ast.FunctionDef) and v.name==name)
            self.assertEqual(ast.dump(nested(old)),ast.dump(nested(TREE)))
        for name in ('__init__','owned_process','run'):
            def member(t):return next(v for v in top(t,'_Watch').body if isinstance(v,ast.FunctionDef) and v.name==name)
            self.assertEqual(ast.dump(member(old)),ast.dump(member(TREE)))


if __name__=='__main__':unittest.main()
