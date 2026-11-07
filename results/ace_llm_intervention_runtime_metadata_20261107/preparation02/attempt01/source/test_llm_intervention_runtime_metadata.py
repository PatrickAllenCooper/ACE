"""New stdlib/fabricated controls only; no real numerical package imports."""
import copy
import importlib.util
import io
import json
import marshal
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import types
import unittest
from unittest.mock import Mock, patch

import qualify_llm_intervention_runtime_metadata as q


class MetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['ACE_RUNTIME_METADATA_TEST_OUTPUT'])
        cls.root.mkdir(exist_ok=False)

    def setUp(self):
        self.path = self.root/self._testMethodName; self.path.mkdir()

    def cached_pair(self, folder, *, stale=False, wrong=False):
        folder.mkdir(parents=True, exist_ok=False)
        source = folder/'source.py'; source.write_bytes(b'def f(x):\n    return (x, -0.0, 7)\n')
        os.utime(source, (200, 200))
        body = source.read_bytes() if not wrong else b'def f(x):\n    return (x, -0.0, 8)\n'
        header = importlib.util.MAGIC_NUMBER + (0).to_bytes(4,'little') + (199 if stale else 200).to_bytes(4,'little') + source.stat().st_size.to_bytes(4,'little')
        cache = folder/'cache.pyc'; cache.write_bytes(header+marshal.dumps(compile(body,'/captured/fixture.py','exec',dont_inherit=True,optimize=0)))
        return cache, source

    def fake_metadata(self, prefix):
        rows = []
        for name, version in [('numpy','2.2.5'),('numpy','2.2.6'),('torch','2.9.1'),('scipy','1.15.3'),('transformers','4.57.3')]:
            d = types.SimpleNamespace(metadata={'Name':name},version=version,_path=prefix/(name+'-'+version+'.dist-info'),locate_file=lambda p:prefix)
            rows.append(d)
        defaults = {r.metadata['Name']:r for r in rows}
        return types.SimpleNamespace(distributions=lambda:iter(rows),distribution=lambda n:defaults[n],version=lambda n:defaults[n].version)

    def snapshot(self, prefix):
        return {'python_version':'3.10.19','prefix':str(prefix),'base_prefix':str(prefix),'executable':str(prefix/'bin/python'),
                'sys_path':[str(prefix/'lib/python3.10')], 'mixed_module_paths':[], 'mixed_native_paths':[],
                'packages':{n:{'loaded':True,'version':v,'origin':str(prefix/n/'__init__.py')} for n,v in q.ACTUAL.items() if n!='python'}}

    def test_exact_packet_source_and_scheduler_guards(self):
        base = Path(q.__file__).parent
        sources = {name:{'path':str(path),'sha256':q.sha(path.read_bytes())} for name,path in (
            ('self',Path(q.__file__)),('helper',base/'llm_intervention_runtime_archive.py'),('probe',base/'probe.py'))}
        p = {'version':q.VERSION,'sources':sources,'archive':{'path':str(self.path/'not-opened.tar'),'sha256':q.ARCHIVE_SHA},
             'manifest':{'path':str(self.path/'not-opened-manifest.json'),'sha256':q.MANIFEST_SHA},'runtime_source':q.RUNTIME_SOURCE,
             'output':str(self.path/'future-output'),'resources':q.RESOURCES,'limits':q.LIMITS,'actual_versions':q.ACTUAL,
             'numpy_metadata_versions':['2.2.5','2.2.6']}
        for label, mutate in [('valid',lambda v:None),('boolean_cpu',lambda v:v['resources'].__setitem__('cpus',True)),
                              ('archive_pin',lambda v:v['archive'].__setitem__('sha256','0'*64)),
                              ('wrong_torch',lambda v:v['actual_versions'].__setitem__('torch','2.9.1')),
                              ('hidden_numpy',lambda v:v.__setitem__('numpy_metadata_versions',['2.2.5']))]:
            value=copy.deepcopy(p); mutate(value); file=self.path/(label+'.json');q.write(file,value)
            args=types.SimpleNamespace(packet=str(file),packet_sha256=q.sha(file.read_bytes()),**{n+'_sha256':s['sha256'] for n,s in sources.items()})
            with patch.object(q,'load_verified',side_effect=AssertionError('no dependency execution during authentication')):
                if label=='valid':self.assertEqual(q.authenticate(args)[0],value)
                else:
                    with self.assertRaises(ValueError):q.authenticate(args)
        text='JobId=42 JobState=RUNNING Account=ucb736_asc2 Partition=acpu QOS=cpu-normal NumCPUs=1 NumTasks=1 NumNodes=1 TimeLimit=00:10:00 Requeue=0 Restarts=0 AllocTRES=cpu=1,mem=3G,node=1,billing=1'
        for i,value in enumerate((text,text.replace('mem=3G','mem=8G'),text+',gres/gpu=1',text.replace('Requeue=0','Requeue=1'))):
            out=self.path/('scheduler-'+str(i));out.mkdir()
            with patch.dict(q.os.environ,{'SLURM_JOB_ID':'42','SLURM_JOB_GPUS':'','SLURM_STEP_GPUS':''}),patch.object(q.subprocess,'run',return_value=types.SimpleNamespace(stdout=value,stderr='',returncode=0)):
                if i==0:self.assertEqual(q.allocation(out),'42')
                else:
                    with self.assertRaises(ValueError):q.allocation(out)

    def test_pinned_tiny_archive_stage_retains_six_cache_source_members(self):
        files={name:b'fabricated cache/source bytes '+name.encode() for pair in q.CACHE_PAIRS for name in pair}
        directories={str(parent) for name in files for parent in Path(name).parents if str(parent)!='.'}
        archive=self.path/'tiny.tar';records=[]
        with tarfile.open(archive,'w',format=tarfile.PAX_FORMAT) as tar:
            for name in sorted(set(files)|directories):
                info=tarfile.TarInfo(name);info.mode=0o700 if name in directories else 0o600
                row={'path':name,'mode':info.mode,'source_mode':info.mode}
                if name in directories:
                    info.type=tarfile.DIRTYPE;row['kind']='directory';tar.addfile(info)
                else:
                    b=files[name];info.size=len(b);row.update(kind='file',bytes=len(b),sha256=q.sha(b));tar.addfile(info,io.BytesIO(b))
                records.append(row)
        manifest={'version':'ace-runtime-archive-v1','source':q.RUNTIME_SOURCE,'objects':len(records),'file_bytes':sum(map(len,files.values())),
                  'archive_bytes':archive.stat().st_size,'archive_sha256':q.sha(archive.read_bytes()),'records':records}
        mp=self.path/'manifest.json';q.write(mp,manifest)
        helper_file=Path(q.__file__).parent/'llm_intervention_runtime_archive.py'
        helper=q.load_verified(q.pinned(helper_file,q.HELPER_SHA),helper_file,'tiny_stage_helper')
        packet={'archive':{'path':str(archive),'sha256':manifest['archive_sha256']},'manifest':{'path':str(mp),'sha256':q.sha(mp.read_bytes())},'runtime_source':q.RUNTIME_SOURCE}
        with patch.object(helper,'pack',side_effect=AssertionError('never repack')):
            stage=q.stage_runtime(packet,helper,self.path/'new-runtime',lambda *a,**k:None)
        q.write(self.path/'stage.json',stage)
        self.assertEqual(set(stage['cache_source_sha256']),set(files))
        for name,b in files.items():self.assertEqual((self.path/'new-runtime'/name).read_bytes(),b)
        bad=copy.deepcopy(packet);bad['archive']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'independent archive'):q.stage_runtime(bad,helper,self.path/'not-created',lambda *a,**k:None)
        self.assertFalse((self.path/'not-created').exists())

    def test_cache_timestamp_rejection_and_eligible_code_comparison(self):
        for label, stale, wrong in [('stale',True,True),('eligible_match',False,False),('eligible_mismatch',False,True)]:
            cache,source=self.cached_pair(self.path/label,stale=stale,wrong=wrong)
            before=(cache.read_bytes(),source.read_bytes())
            if stale:
                with patch.object(q.marshal,'load',side_effect=AssertionError('stale cache must not be decoded')):result=q.inspect_cache(cache,source)
            else:result=q.inspect_cache(cache,source)
            q.write(self.path/(label+'.json'),result)
            self.assertEqual(result['admissible_for_diagnosis'],stale or not wrong)
            self.assertEqual(result['source_stat_mtime'],200.)
            self.assertEqual(result['cpython_timestamp_reject_expected'],stale)
            self.assertFalse(result['historical_hash_difference_explained'])
            self.assertEqual(before,(cache.read_bytes(),source.read_bytes()))
        cache,source=self.cached_pair(self.path/'bad_magic');cache.write_bytes(b'BAD!'+cache.read_bytes()[4:])
        result=q.inspect_cache(cache,source);q.write(self.path/'bad_magic.json',result)
        self.assertFalse(result['admissible_for_diagnosis'])

    def test_duplicate_metadata_defaults_and_mixed_origins_remain_distinct(self):
        prefix=self.path/'runtime';prefix.mkdir()
        inventory=q.metadata_inventory(self.fake_metadata(prefix));q.write(self.path/'metadata.json',inventory)
        q.check_metadata(inventory,prefix)
        self.assertEqual([r['version'] for r in inventory['numpy_entries']],['2.2.5','2.2.6'])
        self.assertEqual(inventory['resolved_defaults']['numpy']['version'],'2.2.6')
        self.assertEqual(inventory['resolved_defaults']['torch']['version'],'2.9.1')
        module=types.ModuleType('numpy');module.__version__='2.2.5';module.__file__=str(prefix/'numpy/__init__.py')
        snapshot=q.runtime_snapshot(prefix,q.RUNTIME_SOURCE,modules={'numpy':module},maps='100-200 r-xp 0 00:00 0 /usr/lib/libc.so\n')
        self.assertEqual(snapshot['packages']['numpy']['version'],'2.2.5')
        self.assertEqual(snapshot['mixed_module_paths'],[])
        module.__file__=q.RUNTIME_SOURCE+'/numpy/__init__.py'
        bad=q.runtime_snapshot(prefix,q.RUNTIME_SOURCE,modules={'numpy':module},maps='100-200 r-xp 0 00:00 0 /opt/foreign/lib.so\n')
        q.write(self.path/'mixed.json',bad)
        self.assertTrue(bad['mixed_module_paths']);self.assertTrue(bad['mixed_native_paths'])
        with self.assertRaises(ValueError):q.check_prefix(prefix,bad)
        inventory['numpy_entries'].pop()
        with self.assertRaises(ValueError):q.check_metadata(inventory,prefix)

    def test_fabricated_worker_captures_success_and_import_failure(self):
        for mode in ('success','import_failure'):
            root=self.path/mode;root.mkdir();prefix=root/'runtime';prefix.mkdir()
            expected={}
            for i,(cache_rel,source_rel) in enumerate(q.CACHE_PAIRS):
                cache,source=self.cached_pair(root/('cache-input-'+str(i)),stale=True)
                for relative,original in ((cache_rel,cache),(source_rel,source)):
                    path=prefix/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(original.read_bytes());expected[relative]=q.sha(path.read_bytes())
            stage={'staged_prefix':str(prefix),'manifest_sha256':q.MANIFEST_SHA,'archive_sha256':q.ARCHIVE_SHA,'cache_source_sha256':expected}
            q.write(root/'stage.json',stage)
            args=types.SimpleNamespace(staged_prefix=str(prefix),stage_sha256=q.sha((root/'stage.json').read_bytes()),packet_sha256='f'*64)
            modules={name:types.ModuleType(name) for name in ('torch','torch.nn','torch.nn.attention','transformers','transformers.models',
                'transformers.models.qwen3','transformers.models.qwen3.modeling_qwen3','numpy','scipy','scipy.linalg')}
            for name,m in modules.items():m.__file__=str(prefix/(name.replace('.','/')+'.py'))
            for name,version in q.ACTUAL.items():
                if name!='python':modules[name].__version__=version
            modules['torch.nn.attention'].sdpa_kernel=object();modules['torch.nn.attention'].SDPBackend=object()
            modules['transformers'].AutoModelForCausalLM=object();modules['transformers'].AutoTokenizer=object()
            if mode=='success':modules['transformers.models.qwen3.modeling_qwen3'].Qwen3ForCausalLM=object()
            np=modules['numpy'];np.array=lambda v:v;np.dot=lambda a,b:sum(x*y for x,y in zip(a,b))
            def solve(a,b):
                determinant=a[0][0]*a[1][1]-a[0][1]*a[1][0]
                return [(b[0]*a[1][1]-b[1]*a[0][1])/determinant,(a[0][0]*b[1]-a[1][0]*b[0])/determinant]
            np.linalg=types.SimpleNamespace(solve=solve);modules['scipy'].linalg=modules['scipy.linalg'];modules['scipy.linalg'].solve=solve
            def snapshot(p,out,name):
                value=self.snapshot(p);q.write(out/(name+'.json'),value);(out/(name+'.maps.txt')).write_text('fabricated maps\n');return value
            metadata=q.metadata_inventory(self.fake_metadata(prefix))
            with patch.dict(sys.modules,modules),patch.object(q,'capture_snapshot',side_effect=snapshot),patch.object(q,'metadata_inventory',return_value=metadata),patch.object(q.signal,'setitimer'):
                if mode=='success':q.worker({'output':str(root)},args,time.monotonic(),time.process_time())
                else:
                    with self.assertRaises(ImportError):q.worker({'output':str(root)},args,time.monotonic(),time.process_time())
            worker=root/'worker';terminal=json.loads((worker/'terminal.json').read_bytes())
            self.assertEqual(terminal['status'],'diagnostic_complete' if mode=='success' else 'failed')
            self.assertTrue((worker/'metadata-before-imports.json').exists());self.assertTrue((worker/'final-runtime.json').exists())
            self.assertTrue((worker/'final-runtime.maps.txt').exists())
            phases=[json.loads(p.read_bytes())['phase'] for p in sorted(worker.glob('phase-*.json'))]
            self.assertLess(phases.index('metadata_capture_complete'),phases.index('torch_start'))
            self.assertLess(phases.index('torch_complete'),phases.index('transformers_auto_complete'))
            self.assertFalse(terminal['service_qualified']);self.assertFalse(terminal['source_only_execution'])
            if mode=='success':self.assertEqual(q.validate_worker(root,args.packet_sha256,args.stage_sha256),q.sha((worker/'terminal.json').read_bytes()))
            else:self.assertTrue((worker/'failure-detail.json').exists())

    def test_absolute_child_deadline_includes_start_validation_and_receipt_io(self):
        for label,start_delay,validate_delay,cleanup_delay in [('valid',20.,1.,0.),('late_receipt',186.,0.,0.),('late_validation',0.,181.,0.),('late_cleanup',0.,0.,6.)]:
            out=self.path/label;out.mkdir();clock=[100.];proc=Mock(pid=123);proc.wait.return_value=0
            cleanup={'group_still_exists':False,'errors':[],'returncode':0};original=q.write
            def write(path,value):
                if path.name=='worker.started.json':clock[0]+=start_delay
                original(path,value)
            def validate():clock[0]+=validate_delay;return 'f'*64
            def kill(p):clock[0]+=cleanup_delay;return cleanup
            killed=Mock(side_effect=kill)
            with patch.object(q.time,'monotonic',side_effect=lambda:clock[0]),patch.object(q.signal,'setitimer') as timer,patch.object(q.signal,'signal'),patch.object(q.subprocess,'Popen',return_value=proc),patch.object(q,'write',side_effect=write):
                if label=='valid':
                    q.run_child(['fabricated'],{},out,680.,killed,validate);proc.wait.assert_called_once_with(timeout=165.)
                else:
                    with self.assertRaises(TimeoutError):q.run_child(['fabricated'],{},out,680.,killed,validate)
                self.assertEqual(timer.call_args_list[0].args,(q.signal.ITIMER_REAL,185.))
                killed.assert_called_once_with(proc)
        with patch.object(q.time,'monotonic',return_value=100.),patch.object(q.subprocess,'Popen') as spawn:
            with self.assertRaises(ValueError):q.run_child([],{},self.path,299.,lambda p:None,lambda:None)
            spawn.assert_not_called()

    def test_child_bootstrap_checks_captured_bytes_before_execution(self):
        file=self.path/'bootstrap-target.py';marker=self.path/'executed.json'
        file.write_text('import json,sys\nfrom pathlib import Path\nPath('+repr(str(marker))+').write_text(json.dumps({"file":__file__,"argv":sys.argv}))\n')
        args=types.SimpleNamespace(packet=str(self.path/'diagnostic.json'),packet_sha256='a'*64,self_sha256=q.sha(file.read_bytes()),helper_sha256=q.HELPER_SHA,probe_sha256=q.PROBE_SHA)
        command=q.child_command({'sources':{'self':{'path':str(file)}}},args,Path('/fabricated-prefix'),'b'*64)
        self.assertEqual(command[1:7],['-I','-B','-X','importtime','-c',q.BOOTSTRAP])
        command[0]=sys.executable
        bad=list(command);bad[8]='0'*64
        for label,cmd in [('bad',bad),('good',command)]:
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=5)
            q.write(self.path/(label+'.json'),{'command':cmd,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
            self.assertEqual(result.returncode==0,label=='good');self.assertEqual(marker.exists(),label=='good')
        self.assertEqual(json.loads(marker.read_bytes())['file'],str(file))

    def test_authoritative_publication_survives_link_interrupt(self):
        original=q.os.link
        for mode in ('normal','before_link','after_link'):
            out=self.path/mode;out.mkdir()
            def link(a,b):
                if mode=='before_link':raise OSError('fabricated pre-link failure')
                original(a,b)
                if mode=='after_link':raise KeyboardInterrupt('fabricated post-link interrupt')
            with patch.object(q.os,'link',side_effect=link):
                if mode=='normal':q.publish(out,{'status':'diagnostic_complete'})
                else:
                    with self.assertRaises(OSError if mode=='before_link' else KeyboardInterrupt):q.publish(out,{'status':'diagnostic_complete'})
            self.assertEqual((out/'terminal.json').exists(),mode!='before_link')
            self.assertFalse((out/'failure.json').exists())
            if mode=='after_link':self.assertEqual(json.loads((out/'publication-diagnostic.json').read_bytes())['authority'],'terminal.json')


class MetadataDeltaTests(unittest.TestCase):
    """v2-only fabricated fixtures. Never execute captured third-party sources."""
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['ACE_RUNTIME_METADATA_DELTA_OUTPUT']).resolve()
        cls.root.mkdir(exist_ok=False)

    def setUp(self):
        self.path = self.root/self._testMethodName; self.path.mkdir()

    def fixture(self, label, *, generate=True):
        base = self.path/label; base.mkdir()
        prefix = base/'runtime'; prefix.mkdir()
        root, root_stat = q.create_child_temp(prefix)
        substitutions = {'assign_module_interface_cls':'module_interface_cls = None', 'args':'*args',
                         'kwargs':'**kwargs', 'arg_types':'*args, **kwargs', 'arrow_and_return_type':'',
                         'arrow_and_future_return_type':'', 'jit_script_decorator':''}
        generator = ("_FILE_PREFIX = '_remote_module_'\ndef instantiate_non_scriptable_remote_module_template():\n"
                     '    str_dict = dict('+', '.join(k+'='+repr(v) for k,v in substitutions.items())+')\n'
                     "    raise AssertionError('fixture generator must never execute')\n").encode()
        template = ("_TEMPLATE_PREFIX = "+repr('{assign_module_interface_cls}\ndef forward(self, {arg_types}):\n    return None\n')+
                    "\n_REMOTE_FORWARD_TEMPLATE_ENABLE_MOVING_CPU_TENSORS_TO_CUDA = "+repr('def forward_async(self, {arg_types}):\n    return None\ndef _remote_forward({arg_types}):\n    return None\n')+
                    "\nraise AssertionError('fixture template must never execute')\n").encode()
        sources, modules = {}, {}
        for name,(relative,_) in q.PROVENANCE_SOURCES.items():
            b = generator if name == q.GENERATOR else template if name == q.TEMPLATE else b"raise AssertionError('fixture source must never execute')\n"
            p = prefix/'lib/python3.10/site-packages'/relative; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b)
            sources[name] = (relative,q.sha(b))
            m = types.ModuleType(name); m.__file__ = str(p); m.__spec__ = importlib.util.spec_from_file_location(name,p); modules[name] = m
        lazy = type('_LazyModule',(types.ModuleType,),{'__module__':q.LAZY_UTILS,
                    '__getattr__':lambda self,name: (_ for _ in ()).throw(AssertionError('no lazy traversal'))})
        modules[q.LAZY_UTILS]._LazyModule = lazy
        m = lazy('transformers'); m.__dict__.update(vars(modules['transformers'])); m._objects = {'__version__':'4.57.3'}; modules['transformers'] = m
        for name in ('numpy','scipy','torch'):
            m = types.ModuleType(name); m.__file__ = str(prefix/(name+'.py')); m.__version__ = q.ACTUAL[name]; modules[name] = m
        with patch.object(q,'PROVENANCE_SOURCES',sources): contract = q.provenance_contract(prefix)
        value = dict(base=base,prefix=prefix,root=root,root_stat=root_stat,sources=sources,contract=contract,modules=modules,
                     baseline=[str(prefix/'lib/python3.10')],generator=generator,template=template)
        if generate:self.generate(value)
        return value

    def generate(self, f):
        owner = q.tempfile.TemporaryDirectory(dir=f['root'])
        owner._finalizer.detach()  # Retain every artificial fixture, including failures.
        folder = Path(owner.name); path = folder/(q.GENERATED_MODULE+'.py')
        path.write_text(f['contract']['expected_source'])
        f['modules'][q.GENERATOR]._TEMP_DIR = owner
        f['modules'][q.GENERATOR].INSTANTIATED_TEMPLATE_DIR_PATH = str(folder)
        m = types.ModuleType(q.GENERATED_MODULE); m.__file__ = str(path); m.__spec__ = importlib.util.spec_from_file_location(q.GENERATED_MODULE,path)
        def artificial_function(*args,**kwargs): return None
        for name in ('forward','forward_async','_remote_forward'):
            fn = types.FunctionType(artificial_function.__code__.replace(co_filename=str(path),co_name=name),{})
            fn.__module__ = q.GENERATED_MODULE; setattr(m,name,fn)
        f['modules'][q.GENERATED_MODULE] = m; f.update(file=path,folder=folder,owner=owner)

    def snapshot(self, f, *, search=None, maps=''):
        with patch.object(q.sys,'prefix',str(f['prefix'])),patch.object(q.sys,'base_prefix',str(f['prefix'])),\
             patch.object(q.sys,'executable',str(f['prefix']/'bin/python')),\
             patch.object(q.sys,'path',search if search is not None else f['baseline']+[str(f['folder'])]):
            return q.runtime_snapshot(f['prefix'],q.RUNTIME_SOURCE,modules=f['modules'],maps=maps,
                contract=f['contract'],temp_root=f['root'],temp_stat=f['root_stat'])

    def test_delta_canonical_static_generation_and_lazy_version(self):
        f=self.fixture('canonical');snapshot=self.snapshot(f)
        q.write(f['base']/'snapshot.json',snapshot);q.check_prefix(f['prefix'],snapshot,f['baseline'])
        proof=snapshot['generated_provenance'];self.assertTrue(proof['verified'])
        self.assertEqual(proof['source_content'],f['contract']['expected_source'])
        self.assertEqual(proof['source_sha256'],q.sha(f['file'].read_bytes()))
        self.assertEqual(snapshot['mixed_modules'],{q.GENERATED_MODULE:str(f['file'])})
        version=snapshot['packages']['transformers']
        self.assertIsNone(version['direct_version']);self.assertFalse(version['direct_version_present'])
        self.assertEqual(version['version'],'4.57.3');self.assertEqual(version['version_resolution'],'transformers_pinned_lazy_objects')
        with self.assertRaises(FileExistsError):q.create_child_temp(f['prefix'])
        packet={'version':'ace-runtime-metadata-diagnostic-v1','sources':{},'archive':{},'manifest':{},
                'runtime_source':q.RUNTIME_SOURCE,'output':str(f['base']/'uncreated'),'resources':q.RESOURCES,
                'limits':q.LIMITS,'actual_versions':q.ACTUAL,'numpy_metadata_versions':['2.2.5','2.2.6']}
        q.write(f['base']/'old-version-packet.json',packet)
        args=types.SimpleNamespace(packet=f['base']/'old-version-packet.json',packet_sha256=q.sha(q.raw(packet)))
        with self.assertRaisesRegex(ValueError,'fixed diagnostic contract'):q.authenticate(args)

    def test_delta_forged_origin_and_extra_paths_rejected(self):
        for mode in ('file','spec','loader','function','generator','extra_search','extra_module','extra_native'):
            with self.subTest(mode=mode):
                f=self.fixture(mode);m=f['modules'][q.GENERATED_MODULE];search=None;maps=''
                if mode=='file':m.__file__=str(f['base']/'forged.py')
                elif mode=='spec':m.__spec__.origin=str(f['base']/'forged.py')
                elif mode=='loader':m.__spec__.loader.path=str(f['base']/'forged.py')
                elif mode=='function':m.forward.__code__=m.forward.__code__.replace(co_filename='/forged.py')
                elif mode=='generator':f['modules'][q.GENERATOR].__spec__.origin='/forged-generator.py'
                elif mode=='extra_search':search=f['baseline']+[str(f['folder']),'/tmp']
                elif mode=='extra_module':f['modules']['forged_alias']=m
                elif mode=='extra_native':maps='100-200 r-xp 0 00:00 0 /opt/foreign.so\n'
                snapshot=self.snapshot(f,search=search,maps=maps);q.write(f['base']/'rejected.json',snapshot)
                with self.assertRaises(ValueError):q.check_prefix(f['prefix'],snapshot,f['baseline'])

    def test_delta_content_pins_and_nonliteral_generator_rejected(self):
        f=self.fixture('content');f['file'].write_text(f['contract']['expected_source']+'# forged\n')
        snapshot=self.snapshot(f);q.write(f['base']/'rejected.json',snapshot)
        self.assertIn('# forged',snapshot['generated_provenance']['source_content'])
        with self.assertRaisesRegex(ValueError,'pinned literal expansion'):q.check_prefix(f['prefix'],snapshot,f['baseline'])
        f=self.fixture('source_pin');source=Path(f['contract']['sources'][q.TEMPLATE]['path']);source.write_bytes(source.read_bytes()+b'# forged\n')
        with patch.object(q,'PROVENANCE_SOURCES',f['sources']):
            with self.assertRaisesRegex(ValueError,'pinned provenance source'):q.provenance_contract(f['prefix'])
        with self.assertRaisesRegex(ValueError,'static string literal'):
            q.static_generated_source(f['generator'].replace(b"args='*args'",b"args=str('must not execute')"),f['template'])

    def test_delta_ownership_symlink_and_inventory_rejected(self):
        for mode in ('ownership','permissions','symlink','inventory','owner_path'):
            with self.subTest(mode=mode):
                f=self.fixture(mode);original=Path.lstat
                def fake_stat(path):
                    s=original(path)
                    if mode=='ownership' and path==f['folder']:
                        return types.SimpleNamespace(**{k:(getattr(s,k)+1 if k=='st_uid' else getattr(s,k))
                            for k in ('st_dev','st_ino','st_uid','st_gid','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns')})
                    return s
                if mode=='permissions':f['folder'].chmod(0o755)
                elif mode=='symlink':
                    saved=f['base']/'retained-original.py';f['file'].rename(saved);f['file'].symlink_to(saved)
                elif mode=='inventory':(f['folder']/'unexpected.py').write_text('forged\n')
                elif mode=='owner_path':f['owner'].name=str(f['base']/'forged-owner')
                with patch.object(Path,'lstat',fake_stat):snapshot=self.snapshot(f)
                q.write(f['base']/'rejected.json',snapshot)
                self.assertFalse(snapshot['generated_provenance']['verified'])
                with self.assertRaises(ValueError):q.check_prefix(f['prefix'],snapshot,f['baseline'])

    def test_delta_lazy_version_wrong_class_origin_value_and_conflict(self):
        for mode in ('class','origin','missing','nonstr','wrong','conflict'):
            with self.subTest(mode=mode):
                f=self.fixture(mode);m=f['modules']['transformers']
                if mode=='class':
                    fake=types.ModuleType('transformers');fake.__dict__.update(vars(m));f['modules']['transformers']=fake
                elif mode=='origin':m.__spec__.origin='/forged-transformers.py'
                elif mode=='missing':m._objects={}
                elif mode=='nonstr':m._objects={'__version__':4573}
                elif mode=='wrong':m._objects={'__version__':'4.57.2'}
                elif mode=='conflict':m.__version__='4.57.2'
                snapshot=self.snapshot(f);q.write(f['base']/'rejected.json',snapshot)
                self.assertEqual(snapshot['packages']['transformers']['version_resolution'],'rejected')
                self.assertIsNone(snapshot['packages']['transformers']['version'])
                with self.assertRaisesRegex(ValueError,'package version observation'):q.check_prefix(f['prefix'],snapshot,f['baseline'])

    def test_delta_worker_retains_generated_evidence_on_success_and_failure(self):
        original_snapshot=q.runtime_snapshot
        for mode in ('success','content_failure'):
            f=self.fixture(mode,generate=False);out=f['base']/'output';out.mkdir()
            stage={'staged_prefix':str(f['prefix']),'archive_sha256':q.ARCHIVE_SHA,'manifest_sha256':q.MANIFEST_SHA,'cache_source_sha256':{},
                   'temporary_root':{'path':str(f['root']),'stat':f['root_stat']}}
            q.write(out/'stage.json',stage)
            args=types.SimpleNamespace(staged_prefix=str(f['prefix']),stage_sha256=q.sha(q.raw(stage)),packet_sha256='a'*64)
            observed={}
            def imports(event):
                self.generate(f)
                if mode=='content_failure':f['file'].write_text(f['contract']['expected_source']+'# forged\n')
                observed.update(f['modules']);q.sys.path.append(str(f['folder']))
                return f['modules']['numpy'],f['modules']['scipy']
            def snapshot(prefix,original,**kw):
                return original_snapshot(prefix,original,modules=observed,maps='',**kw)
            fake_scipy=f['modules']['scipy'];fake_linalg=types.ModuleType('scipy.linalg');fake_scipy.linalg=fake_linalg
            metadata={'numpy_entries':[{'version':'2.2.5'},{'version':'2.2.6'}],'distributions':[],'resolved_defaults':{}}
            with patch.object(q,'PROVENANCE_SOURCES',f['sources']),patch.object(q,'original_imports',side_effect=imports),\
                 patch.object(q,'runtime_snapshot',side_effect=snapshot),patch.object(q,'capture_caches'),\
                 patch.object(q,'metadata_inventory',return_value=metadata),patch.object(q,'arithmetic',return_value={'passed':True}),\
                 patch.object(q.sys,'prefix',str(f['prefix'])),patch.object(q.sys,'base_prefix',str(f['prefix'])),\
                 patch.object(q.sys,'executable',str(f['prefix']/'bin/python')),patch.object(q.sys,'path',list(f['baseline'])),\
                 patch.object(q.sys,'version_info',(3,10,19)),patch.object(q.tempfile,'gettempdir',return_value=str(f['root'])),\
                 patch.dict(os.environ,{k:str(f['root']) for k in ('TMPDIR','TMP','TEMP')}),patch.object(q.signal,'setitimer'),\
                 patch.dict(sys.modules,{'scipy':fake_scipy,'scipy.linalg':fake_linalg}):
                if mode=='success':q.worker({'output':str(out)},args,time.monotonic(),time.process_time())
                else:
                    with self.assertRaisesRegex(ValueError,'pinned literal expansion'):q.worker({'output':str(out)},args,time.monotonic(),time.process_time())
            terminal=json.loads((out/'worker/terminal.json').read_bytes())
            self.assertEqual(terminal['status'],'diagnostic_complete' if mode=='success' else 'failed')
            self.assertEqual(terminal['version'],q.VERSION);self.assertFalse(terminal['runtime_adopted'])
            self.assertFalse(terminal['service_qualified']);self.assertEqual(terminal['environment_responses'],0)
            self.assertEqual(terminal['generated_module_verified'],mode=='success')
            self.assertIn('provenance-sources.json',terminal['evidence_sha256'])
            final=json.loads((out/'worker/final-runtime.json').read_bytes())
            self.assertIn('source_content',final['generated_provenance'])
            if mode=='success':q.validate_worker(out,args.packet_sha256,args.stage_sha256)


if __name__=='__main__':unittest.main(verbosity=2)
