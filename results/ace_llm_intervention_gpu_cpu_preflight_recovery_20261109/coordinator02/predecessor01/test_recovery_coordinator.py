"""New artificial recovery joins only. Never invoke coordinator main/SSH/locks/science."""
import ast,contextlib,copy,importlib.util,json,os,pathlib,tempfile,unittest
HERE=pathlib.Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('gpu02_metadata',HERE/'authoring/build_packet.py')
b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
H=lambda v:b.sha(v.encode())
info=lambda v:dict(bytes=len(v.encode()),sha256=H(v))


def fixture():
    priorrows=[[str(100+i),'ace_llm_fixture','COMPLETED','0:0','2','1','ucb736_asc2','cpu=2,mem=7680M',''] for i in range(46)]
    row=[b.FAILED_GPU_JOB,'ace_llm_scientific_campaign_gpu01','FAILED','1:0','4','245','ucb736_asc2','cpu=4,mem=15G,gres/gpu:a100_3g.20gb=1','']
    jp={f'/toy/root-{i}/submission.{kind}.json':H(str(i)+kind) for i in range(47) for kind in ('intent','result')}
    pins={k:H(k) for k in ('cpu-baseline','cpu-acceptance','cpu-packet','gpu-failure-acceptance','gpu-failure-result','gpu-failure-inventory','gpu-failure-inventory-after')}
    pins.update({'gpu-failure-packet':b.FAILED_GPU_PACKET_SHA,'gpu-failure-seal':b.FAILED_GPU_SEAL_SHA,'gpu-failure-disposition':b.FAILED_GPU_DISPOSITION_SHA})
    previous=dict(terminal_jobs=46,submission_journals=94,related_accounting=priorrows,journal_pins=jp,closed_cpu_seconds=92,closed_gpu_slice_seconds=0)
    p=dict(schema='ace-scientific-campaign-gpu-launch-v1',source_commit=b.SOURCE_COMMIT,remote_root=b.OLDROOT,
           resources=b.RESOURCES,limits=b.LIMITS,baseline_sha256=pins['cpu-baseline'],cpu_acceptance_sha256=pins['cpu-acceptance'],cpu_packet_sha256=pins['cpu-packet'],new_input_files={'input.py':H('input')})
    r=dict(job_id=b.FAILED_GPU_JOB,returncode=0,ambiguous=False,intent_sha256=H('remoteintent'))
    files={'launch/input.py':info('input'),'launch/packet.json':dict(bytes=1,sha256=pins['gpu-failure-packet']),
           'launch/submission.intent.json':dict(bytes=1,sha256=r['intent_sha256']),
           'launch/submission.result.json':dict(bytes=1,sha256=pins['gpu-failure-result'])}
    dirs=['launch'];cf={};cd=[]
    for new,old in [('cpu','launch'),('training','training'),('token','cpu'),('failed-token01','failed-token01'),('common','common'),('failed01','failed01')]:
        files[new+'/saved']=info(new);dirs.append(new);cf[old+'/saved']=info(new);cd.append(old)
    journals={**jp,b.OLDROOT+'/submission.intent.json':r['intent_sha256'],b.OLDROOT+'/submission.result.json':pins['gpu-failure-result']}
    obs=dict(original_hardlink_groups=[],terminal_gate=True,root_present=True,files=files,directories=dirs,parent_accounting=row,related_accounting=priorrows+[row],journal_pins=journals)
    transport=dict(returncode=0,python_source=f'ROOT={b.OLDROOT!r}\nJOB={b.FAILED_GPU_JOB!r}\n',stdout=json.dumps(obs))
    d=dict(status='terminal_custody_with_binding_disagreements',job_id=b.FAILED_GPU_JOB,remote_root=b.OLDROOT,
           packet_sha256=pins['gpu-failure-packet'],submission_result_sha256=pins['gpu-failure-result'],custody_bytes_verified=True,
           current_lineage_verified=False,input_pin_mismatches=[],errors=['actual/requested GPU identity/resource/no-requeue/wall contract differs or unavailable'],
           allocation={'resource_contract_matched':False},remote_files=files,remote_directories=dirs)
    sealed={'files':{n:dict(bytes=1,sha256=pins[k]) for n,k in [('disposition.json','gpu-failure-disposition'),('inventory_transport.json','gpu-failure-inventory'),('inventory_after_transport.json','gpu-failure-inventory-after')]},'directories':[]}
    accepted=dict(schema='ace-scientific-campaign-gpu-failure-acceptance-v1',status='accepted_failed_prepolicy_attempt',
        job_id=b.FAILED_GPU_JOB,required_findings=0,acceptance=True,complete_CURC_backup_verified_from_saved_records=True,
        all_selected_local_files_verified_in_both_copies=True,complete_local_raw_copy=False,
        new_policy_attempts=0,new_scientific_responses=0,new_training_updates=0,new_world_realizations=0,
        original_collector_returncode=1,retroactive_collector_success=False,originals_changed=False,
        outstanding_reservation=0,related_terminal_jobs=47,submission_journals=96,scheduler_state='FAILED',scheduler_exit_code='1:0',
        source_commit=b.SOURCE_COMMIT,closed_cpu_seconds=1072,closed_gpu_slice_seconds=245,cpu_admission_seconds=4672,
        no_policy_work_proof_sha256=H('artificial proof'),resource_reconciliation_sha256=H('artificial reconciliation'),source_boundary_sha256=H('artificial boundary'))
    accepted['input_pins']={n:dict(sha256=pins[k]) for n,k in {'original_disposition':'gpu-failure-disposition',
        'original_failure_seal':'gpu-failure-seal','original_submission_result':'gpu-failure-result','outer_packet':'gpu-failure-packet',
        'predecessor_acceptance':'cpu-acceptance','predecessor_baseline':'cpu-baseline'}.items()}
    accepted['input_pins']['submitted_entry']={'sha256':b.ENTRY_SHA}
    authority=dict(failed_gpu01_acceptance_sha256=pins['gpu-failure-acceptance'])
    baseline=dict(terminal_jobs=47,submission_journals=96,outstanding_reservation=0,preparation_allowance_cpu_seconds=3600,cpu_cap_seconds=57600,gpu_cap_seconds=7200,
                  input_pins=dict(predecessor=pins['cpu-baseline'],acceptance=pins['gpu-failure-acceptance'],disposition=pins['gpu-failure-disposition']),
                  related_accounting=priorrows+[row],journal_pins=journals,closed_cpu_seconds=1072,closed_gpu_slice_seconds=245,cpu_admission_seconds=4672)
    m={'gpu-failure-acceptance':accepted,'gpu-recovery-authority':authority,'gpu-failure-baseline':baseline,'cpu-baseline':previous,
       'gpu-failure-disposition':d,'gpu-failure-packet':p,'gpu-failure-result':r,
       'gpu-failure-seal':dict(identical=True,copies={'/toy/a':sealed,'/toy/b':sealed}),
       'gpu-failure-inventory':transport,'gpu-failure-inventory-after':copy.deepcopy(transport),
       'cpu-inventory':dict(stdout=json.dumps(dict(files=cf,directories=cd)))}
    return m,pins


class RecoveryTests(unittest.TestCase):
    def test_failed_collector_reconciliation_and_exact_charge_extension(self):
        m,pins=fixture();before=copy.deepcopy(m);old=b.failed_gpu_history(m,pins)
        self.assertEqual(m,before);self.assertEqual(set(old['files']),{'input.py','packet.json','submission.intent.json','submission.result.json'})
        for alter in ('pretend_clean','wrong_cost','wrong_journal','missing_before_after_file','wrong_accepted_status'):
            with self.subTest(alter=alter):
                x=copy.deepcopy(m)
                if alter=='pretend_clean':x['gpu-failure-disposition']['current_lineage_verified']=True
                if alter=='wrong_cost':x['gpu-failure-baseline']['closed_cpu_seconds']-=1
                if alter=='wrong_journal':x['gpu-failure-baseline']['journal_pins'].pop(next(iter(x['gpu-failure-baseline']['journal_pins'])))
                if alter=='missing_before_after_file':
                    o=json.loads(x['gpu-failure-inventory-after']['stdout']);del o['files']['launch/input.py'];x['gpu-failure-inventory-after']['stdout']=json.dumps(o)
                if alter=='wrong_accepted_status':
                    x['gpu-failure-acceptance']=dict(x['gpu-failure-acceptance'],status='not_accepted')
                with self.assertRaises(ValueError):b.failed_gpu_history(x,pins)

    def test_failed_root_only_exception_and_exclusive_append(self):
        tree=ast.parse((HERE/'authoring/launch_once.py').read_bytes());remote=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE_COMMON' for t in n.targets)))
        fn=next(n for n in ast.parse(remote).body if isinstance(n,ast.FunctionDef) and n.name=='failed_gpu_unchanged')
        with contextlib.nullcontext(tempfile.mkdtemp(dir=HERE/'fixtures')) as td:
            parent=pathlib.Path(td).resolve();old=parent/'ace_llm_intervention_scientific_campaign_gpu_old';old.mkdir();(old/'evidence').write_bytes(b'old')
            root=parent/'ace_llm_intervention_scientific_campaign_gpu_new'
            env=dict(pathlib=pathlib,os=os,root=root,need=b.need,sha=b.sha,read=lambda p,**kw:p.read_bytes(),
                     C=dict(failed_gpu_root=str(old),gpu_output=str(old/'gpu'),failed_gpu_inventory=dict(files={'evidence':info('old')},directories=[])))
            exec(compile(ast.Module(body=[fn],type_ignores=[]),'<only recovery guard>','exec'),env)
            self.assertTrue(env['failed_gpu_unchanged']())
            for mode in ('dangling_gpu','changed_file','extra_directory','unknown_sibling'):
                with self.subTest(mode=mode):
                    if mode=='dangling_gpu':(old/'gpu').symlink_to(parent/'absent')
                    if mode=='changed_file':(old/'evidence').write_bytes(b'changed')
                    if mode=='extra_directory':(old/'extra').mkdir()
                    if mode=='unknown_sibling':(parent/'ace_llm_intervention_scientific_campaign_gpu_other').mkdir()
                    with self.assertRaises(ValueError):env['failed_gpu_unchanged']()
                    if mode=='dangling_gpu':(old/'gpu').unlink()
                    if mode=='changed_file':(old/'evidence').write_bytes(b'old')
                    if mode=='extra_directory':(old/'extra').rmdir()
                    if mode=='unknown_sibling':(parent/'ace_llm_intervention_scientific_campaign_gpu_other').rmdir()

    def test_only_operational_sources_and_four_metadata_added(self):
        original={f'input-{i}.json':f'fixed-{i}'.encode() for i in range(95)}
        entry=b"EXTRA_SOURCES={'qualify_llm_intervention_scientific_campaign_gpu.py'}\n"
        data={**original,b.ENTRY:entry};helpers={b.PREFLIGHT:b'# pure fixture\n',b.HISTORY:b'# metadata fixture\n'}
        derived=('VERSION='+repr(b.VERSION)+'\nRECOVERY_FILES='+repr(b.RECOVERY_FILES)+'\nEXTRA_SOURCES='+repr({b.ENTRY,b.PREFLIGHT,b.HISTORY})+'\nPREFLIGHT_SHA='+repr(b.sha(helpers[b.PREFLIGHT]))+'\nHISTORY_SHA='+repr(b.sha(helpers[b.HISTORY]))+'\n').encode()
        captured={**helpers,b.ENTRY:derived};sourcepins={n:b.sha(v) for n,v in captured.items()}
        ctx={'operational_sources':captured,'metadata':{'cpu-packet':{'files':{n:b.sha(v) for n,v in original.items()}},
             'gpu-recovery-authority':{'operational_source_files':sourcepins},
             'cpu-history-expected':{'execution_context':{'source_files_sha256':b.digest(sourcepins)}}}}
        supplied={n[:-5]:b'{}' for n in b.RECOVERY_FILES.values()}
        result,sources=b.inject_recovery(ctx,data,supplied,[b.ENTRY])
        self.assertEqual({n:result[n] for n in original},original)
        self.assertEqual(set(result)-set(data),{b.PREFLIGHT,b.HISTORY}|set(b.RECOVERY_FILES.values()))
        self.assertEqual(sources,sorted(captured))
        for mode in ('cpu_change','helper_change'):
            with self.subTest(mode=mode):
                x=copy.deepcopy(ctx);d=dict(data)
                if mode=='cpu_change':d['input-0.json']=b'new'
                else:x['operational_sources'][b.HISTORY]=b'changed helper'
                with self.assertRaises(ValueError):b.inject_recovery(x,d,supplied,[b.ENTRY])

    def test_original_guards_and_once_submission_static(self):
        original=ast.parse((HERE/'predecessor/build_packet.py').read_bytes());new=ast.parse((HERE/'authoring/build_packet.py').read_bytes())
        functions=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,ast.FunctionDef)}
        a,c=functions(original),functions(new)
        for n in ('history','token_recovery_history','admit_cpu_bytes','coordinator_locks','configure_storage','literal','cost','normalized'):
            self.assertEqual(a[n],c[n],n)
        launch=(HERE/'authoring/launch_once.py').read_text()
        self.assertEqual(launch.count("['sbatch','--parsable'"),1)
        self.assertIn('fresh=reconcile();verify();failed_gpu_unchanged()',launch)
        self.assertIn("baseline=m['gpu-failure-baseline']",launch)
        self.assertIn("s['remote_root']==OLDROOT",(HERE/'authoring/build_packet.py').read_text())
        self.assertIn("'--output',OUTPUT",(HERE/'authoring/build_packet.py').read_text())
        self.assertEqual(b.RESOURCES['wall_seconds'],2880)

if __name__=='__main__':unittest.main(verbosity=2)
