"""Retention boundary tests: metadata-only Python stubs; no models or worlds.

Only Git blob retrieval is faked for the end-to-end stub attempts. Source-file
hash checks, freeze pins, process execution, result reconciliation, and cleanup
use the new supervisor. No existing component/mismatch test suite is imported.
"""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('retention_supervisor',Path(__file__).with_name('supervise_foundation_retention.py'))
sup=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sup)

STUB='''import json,os,resource,sys,time
from pathlib import Path
out=Path(sys.argv[sys.argv.index('--output')+1])
mode=sys.argv[sys.argv.index('--mode')+1]
plan=json.loads((out/'plan.json').read_text())['cells']
(out/'stub_runtime.json').write_text(json.dumps({'environment':os.environ.get('ACE_RETENTION_SUPERVISED'),
    'cpu_limits':resource.getrlimit(resource.RLIMIT_CPU),'threads':os.environ.get('OMP_NUM_THREADS'),
    'freeze_sha256':sys.argv[sys.argv.index('--freeze-sha256')+1]}))
ACTION=__ACTION__
rows=[]
for i,item in enumerate(plan):
    d=out/str(item['seed'])/item['variant'];d.mkdir(parents=True,exist_ok=True)
    (d/(item['method']+'.started.json')).write_text(json.dumps(item))
    if ACTION=='sleep' or (ACTION=='partial' and i==1):
        print('stub sleeping',flush=True);time.sleep(10)
    row=dict(item,status='failed' if ACTION=='failed_cell' and i==0 else 'complete')
    if ACTION=='wrong_identity' and i==0:row['seed']=123456
    (d/(item['method']+'.json')).write_text(json.dumps(row));rows.append(row)
if ACTION=='bad_marker':
    (out/'complete.json').write_text('{broken')
else:
    n=6 if mode=='pilot' else 1
    complete={'mode':mode,'cells':rows,'planned_cells':len(plan),
        'completed_cells':sum(r['status']=='complete' for r in rows),
        'training_responses_total':n*160,'private_responses_total':n*3072,
        'stub_metadata_only':True}
    if ACTION=='wrong_counts':complete['training_responses_total']+=1
    (out/'complete.json').write_text(json.dumps(complete))
'''


class RetentionSupervision(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='ace-retention-supervisor-stub-')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.counter=0

    def packet(self,mode='fixture',action='success'):
        self.counter+=1
        root=self.root/str(self.counter);root.mkdir()
        repo=root/'repo';repo.mkdir()
        pins={}
        for rel in sup.SOURCES:
            p=repo/rel;p.parent.mkdir(parents=True,exist_ok=True)
            if rel.endswith('/supervise_foundation_retention.py'):
                raw=Path(sup.__file__).read_bytes()
            elif rel.endswith('/foundation_retention_pilot.py'):
                raw=STUB.replace('__ACTION__',repr(action)).encode()
            else:raw=b'# metadata-only stub source; never imported\n'
            p.write_bytes(raw);pins[rel]=hashlib.sha256(raw).hexdigest()
        f={'schema':'ace-retention-freeze-v1','mode':mode,'sources':pins,
           'source_revision':'a'*40,'repository':str(repo),'dependencies':dict.fromkeys(sup.DEPENDENCIES,'stub-version'),
           'python':str(Path(sys.executable).absolute()),'python_version':sys.version,
           'checkpoint':str(root/'unused-no-model'),'checkpoint_sha256':sup.CHECKPOINT_SHA256,
           'limit_seconds':sup.LIMITS[mode],'stop_unix':sup.STOP_UNIX,
           'output':str(root/'attempt'),'lock':str(root/'controller.lock')}
        if mode=='pilot':self.qualify(f,root)
        args=argparse.Namespace(output=Path(f['output']),lock=Path(f['lock']),mode=mode,
            worker=repo/'scripts/research/foundation_retention_pilot.py',python=Path(sys.executable),
            freeze=root/'freeze.json',freeze_sha256='')
        self.seal(args,f)
        return args,f

    def qualify(self,f,root,terminal_change=None,freeze_change=None):
        """Fabricate receipt metadata only, never run a qualification/model."""
        self.counter+=1
        d=root/('qualification-'+str(self.counter));d.mkdir()
        fixture=copy.deepcopy(f);fixture.pop('fixture',None)
        fixture.update(mode='fixture',limit_seconds=120,output=str(d/'attempt'))
        if freeze_change:freeze_change(fixture)
        fp=d/'freeze.json';sup.write(fp,fixture)
        t={'status':'complete','reason':'exited','mode':'fixture','exit_code':0,'error':None,
           'freeze_sha256':sup.sha(fp),'planned_cells':36,
           'cells':[dict(r,status='complete') for r in sup.planned_cells('fixture')],
           'child_cpu_s':.1,'elapsed_s':.2,'peak_child_rss_bytes':1024*1024}
        if terminal_change:terminal_change(t)
        tp=d/'terminal.json';sup.write(tp,t)
        f['fixture']={'terminal_path':str(tp),'terminal_sha256':sup.sha(tp),
                      'freeze_path':str(fp),'freeze_sha256':sup.sha(fp)}

    def seal(self,args,f):
        # Test-only packet editing before an attempt; production writer is exclusive.
        args.freeze.write_text(json.dumps(f));args.freeze_sha256=sup.sha(args.freeze)

    def git_blob(self,command,deadline,cancellation):
        sup.deadline_check(deadline,cancellation)
        self.assertEqual(command[:3],['git','--no-replace-objects','-C'])
        self.assertEqual(command[4],'show')
        revision,rel=command[5].split(':',1)
        self.assertEqual(revision,'a'*40)
        return (Path(command[3])/rel).read_bytes()

    def run_attempt(self,args):
        # Hold the same lock lifetime as main(), including terminal publication.
        with args.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with patch.object(sup,'bounded_git',side_effect=self.git_blob):
                code=sup.supervise(args)
        return code,json.loads((args.output/'terminal.json').read_text())

    def test_plan_and_source_closure(self):
        self.assertEqual(len(sup.SOURCES),9)
        self.assertIn('docs/development/guidance/ace_foundation_mismatch_protocol_2026-10-09.md',sup.SOURCES)
        self.assertIn('scripts/research/foundation_flexible_control.py',sup.SOURCES)
        self.assertIn('scripts/research/foundation_retention_selection.py',sup.SOURCES)
        self.assertEqual(sup.METHODS,('grammar32','rbf24','pfn24','prechange24','raw','local','interval','combined','combined_no_pfn'))
        self.assertEqual(len(sup.planned_cells('fixture')),36)
        self.assertEqual({r['seed'] for r in sup.planned_cells('fixture')},{223456})
        self.assertEqual(len(sup.planned_cells('pilot')),216)
        self.assertEqual({r['seed'] for r in sup.planned_cells('pilot')},set(range(93000,93006)))

    def test_bad_freeze_preserves_both_full_unattempted_plans(self):
        for mode in ('fixture','pilot'):
            with self.subTest(mode=mode):
                args,_=self.packet(mode);args.freeze_sha256='0'*64
                code,t=self.run_attempt(args)
                self.assertEqual(code,1);self.assertEqual(t['cells'],[dict(r,status='unattempted') for r in sup.planned_cells(mode)])
                self.assertIsNone(t['child_cpu_s']);self.assertFalse((args.output/'child.json').exists())

    def test_metadata_only_complete_fixture_and_pilot(self):
        for mode in ('fixture','pilot'):
            with self.subTest(mode=mode):
                args,_=self.packet(mode);code,t=self.run_attempt(args)
                self.assertEqual(code,0);self.assertEqual(t['status'],'complete')
                self.assertEqual(t['planned_cells'],len(sup.planned_cells(mode)))
                runtime=json.loads((args.output/'stub_runtime.json').read_text())
                self.assertEqual(runtime['environment'],'1');self.assertEqual(runtime['threads'],'1')
                self.assertEqual(runtime['cpu_limits'],[sup.LIMITS[mode],sup.LIMITS[mode]+1])
                self.assertEqual(runtime['freeze_sha256'],args.freeze_sha256)

    def test_fixture_cannot_have_failed_cells_but_pilot_can(self):
        for mode in ('fixture','pilot'):
            args,_=self.packet(mode,'failed_cell');code,t=self.run_attempt(args)
            self.assertEqual(code,1 if mode=='fixture' else 0)
            self.assertEqual(t['cells'][0]['status'],'failed')

    def test_full_fixture_identity_and_resource_gate(self):
        changes={
            'old24':lambda t:t.update(cells=t['cells'][:24],planned_cells=24),
            'wrong_seed':lambda t:t['cells'][0].update(seed=123456),
            'duplicate':lambda t:t['cells'].__setitem__(1,t['cells'][0]),
            'failure':lambda t:t['cells'][0].update(status='failed'),
            'supervisor_error':lambda t:t.update(error='failure'),
            'cpu_limit':lambda t:t.update(child_cpu_s=120),
            'rss_limit':lambda t:t.update(peak_child_rss_bytes=6*1024**3+1),
        }
        for label,change in changes.items():
            with self.subTest(case=label):
                args,f=self.packet('pilot');self.qualify(f,args.freeze.parent,terminal_change=change)
                self.seal(args,f);code,t=self.run_attempt(args)
                self.assertEqual(code,1);self.assertFalse((args.output/'child.json').exists())
                self.assertEqual(len(t['cells']),216)

    def test_fixture_source_and_runtime_must_match(self):
        for key in ('sources','dependencies','python','python_version','checkpoint','checkpoint_sha256'):
            with self.subTest(key=key):
                args,f=self.packet('pilot')
                def change(q):
                    if isinstance(q[key],dict):q[key][next(iter(q[key]))]='different'
                    else:q[key]='different'
                self.qualify(f,args.freeze.parent,freeze_change=change)
                self.seal(args,f);code,_=self.run_attempt(args)
                self.assertEqual(code,1);self.assertFalse((args.output/'child.json').exists())

    def test_source_drift_and_declared_dependency_checkpoint_guards(self):
        for change in ('source','dependency','checkpoint','closure','mode'):
            with self.subTest(case=change):
                args,f=self.packet()
                if change=='source':args.worker.write_text('raise RuntimeError("must not execute")')
                elif change=='dependency':f['dependencies'].pop('torch')
                elif change=='checkpoint':f['checkpoint_sha256']='0'*64
                elif change=='closure':f['sources'].pop('scripts/research/foundation_flexible_control.py')
                else:f['mode']='pilot'
                self.seal(args,f);code,_=self.run_attempt(args)
                self.assertEqual(code,1);self.assertFalse((args.output/'child.json').exists())

    def test_output_lock_binding_and_exclusive_attempt(self):
        for field in ('output','lock'):
            args,f=self.packet();f[field]+='-wrong';self.seal(args,f)
            code,_=self.run_attempt(args)
            self.assertEqual(code,1);self.assertFalse((args.output/'child.json').exists())
        args,_=self.packet();self.run_attempt(args)
        with self.assertRaises(FileExistsError):self.run_attempt(args)
        argv=[sup.__file__]
        for field in ('output','python','worker','freeze','lock','mode','freeze_sha256'):
            argv+=['--'+field.replace('_','-'),str(getattr(args,field))]
        with args.lock.open('a') as held:
            fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with patch.object(sys,'argv',argv),self.assertRaises(BlockingIOError):sup.main()

    def test_invalid_completion_preserves_expected_identities(self):
        for action in ('bad_marker','wrong_identity','wrong_counts'):
            with self.subTest(action=action):
                args,_=self.packet(action=action);code,t=self.run_attempt(args)
                self.assertEqual(code,1);self.assertEqual(t['reason'],'invalid_completion')
                self.assertEqual([{k:r[k] for k in ('seed','variant','method')} for r in t['cells']],sup.planned_cells('fixture'))
                if action=='wrong_identity':self.assertEqual(t['cells'][0]['status'],'invalid_record')

    def test_timeout_preserves_completed_interrupted_and_unattempted(self):
        with patch.dict(sup.LIMITS,fixture=.4):
            args,_=self.packet(action='partial');code,t=self.run_attempt(args)
        self.assertEqual(code,1);self.assertEqual(t['reason'],'wall_timeout')
        self.assertEqual([r['status'] for r in t['cells'][:2]],['complete','interrupted'])
        self.assertTrue(all(r['status']=='unattempted' for r in t['cells'][2:]))
        pid=json.loads((args.output/'child.json').read_text())['pid']
        with self.assertRaises(ProcessLookupError):os.kill(pid,0)

    def test_cancel_during_creation_and_reaping_keeps_full_plan(self):
        original_popen=sup.subprocess.Popen;original_wait=sup.os.wait4
        def create(*a,**kw):
            child=original_popen(*a,**kw);os.kill(os.getpid(),signal.SIGTERM);return child
        def reap(*a,**kw):
            result=original_wait(*a,**kw)
            if result[0]:os.kill(os.getpid(),signal.SIGTERM)
            return result
        for boundary in ('create','reap'):
            args,_=self.packet(action='sleep' if boundary=='create' else 'success')
            obj,name,fn=(sup.subprocess,'Popen',create) if boundary=='create' else (sup.os,'wait4',reap)
            with patch.object(obj,name,side_effect=fn):code,t=self.run_attempt(args)
            self.assertEqual(code,1);self.assertEqual(t['reason'],'cancelled');self.assertEqual(len(t['cells']),36)
            pid=json.loads((args.output/'child.json').read_text())['pid']
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)

    def test_preflight_expiry_no_spawn_and_bounded_owned_process(self):
        with patch.object(sup.subprocess,'Popen',side_effect=AssertionError('expired spawn')):
            with self.assertRaises(TimeoutError):sup.bounded_git(['unused'],time.monotonic()-1,[])
        spawned=[];original=sup.subprocess.Popen
        def track(*a,**kw):
            child=original(*a,**kw);spawned.append(child.pid);return child
        with patch.object(sup.subprocess,'Popen',side_effect=track),self.assertRaises(TimeoutError):
            sup.bounded_git([sys.executable,'-c','import time;time.sleep(10)'],time.monotonic()+.05,[])
        with self.assertRaises(ProcessLookupError):os.kill(spawned[0],0)
        args,_=self.packet()
        original_validate=sup.validate_stage
        def expire(*a,**kw):
            original_validate(*a,**kw)
            # Emulate time consumed by preflight, without waiting 120 seconds.
            sup.time.monotonic=lambda:10**20
        with patch.object(sup,'validate_stage',side_effect=expire),patch.object(sup.time,'monotonic',wraps=time.monotonic):
            code,t=self.run_attempt(args)
        self.assertEqual(code,1);self.assertFalse((args.output/'child.json').exists())
        self.assertTrue(all(r['status']=='unattempted' for r in t['cells']))


if __name__=='__main__':unittest.main()
