"""NEW artificial coordinator metadata checks only; no project module imports or main calls."""
from pathlib import Path
import ast,copy,json,os,sys,tempfile,unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
import build_packet as b
HERE=Path(__file__).resolve().parent
H=lambda s:b.sha(str(s).encode())


def history_fixture():
    rows=[[str(i),'ace_llm_fixture_prior','FAILED','1:0','2','1','ucb736_asc2','cpu=2,mem=7680M',''] for i in range(1,42)]
    previous=dict(terminal_jobs=41,submission_journals=84,related_accounting=rows,
        journal_pins={f'/fixture/old/{i}.json':H(i) for i in range(84)},closed_cpu_seconds=82,closed_gpu_slice_seconds=0,
        outstanding_reservation=0,preparation_allowance_cpu_seconds=3600,cpu_cap_seconds=57600,gpu_cap_seconds=7200,cpu_admission_seconds=3682)
    m={'collection-failure-baseline':previous};pins={'collection-failure-baseline':H('failed-baseline')};prior_pin=pins['collection-failure-baseline']
    for i,(prefix,root,count,journals,cpus,gpus,wall) in enumerate(b.STAGES,42):
        for kind in ('acceptance','launch-packet','result','disposition','inventory','baseline'):pins[prefix+'-'+kind]=H(prefix+kind)
        job=str(i);result=dict(job_id=job,returncode=0,ambiguous=False,intent_sha256=H(prefix+'intent'))
        row=[job,'ace_llm_fixture_'+prefix,'COMPLETED','0:0',str(cpus),str(10+i),'ucb736_asc2',f'cpu={cpus},mem=1M'+(',gres/gpu=1,gres/gpu:a100_3g.20gb=1' if gpus else ''),'']
        schemas=dict(collection='ace-scientific-collection-launch-v1',token='ace-scientific-tokenization-launch-v1',training='ace-scientific-training-gpu-launch-v1',cpu='ace-scientific-cpu-campaign-launch-v1',gpu='ace-scientific-campaign-gpu-launch-v1')
        packet=dict(schema=schemas[prefix],remote_root=root,source_commit=b.SOURCE_COMMIT,baseline_sha256=prior_pin,
            resources=dict(cpus=cpus,wall_seconds=wall,memory_mib=dict(collection=7680,token=11520,training=15360,cpu=7680,gpu=15360)[prefix],
                gpus=gpus,account='ucb736_asc2',partition='aa100' if gpus else 'acpu',qos='gpu-testing' if gpus else 'cpu-normal'),new_input_files={'inputs/fixture.bin':H(prefix)})
        files={'launch/'+n:dict(bytes=1,sha256=h) for n,h in {**packet['new_input_files'],'packet.json':pins[prefix+'-launch-packet'],
            'submission.intent.json':result['intent_sha256'],'submission.result.json':pins[prefix+'-result']}.items()}
        jp={**previous['journal_pins'],root+'/submission.intent.json':result['intent_sha256'],root+'/submission.result.json':pins[prefix+'-result']}
        allrows=previous['related_accounting']+[row];cpu,gpu=b.cost(allrows)
        base=dict(previous,terminal_jobs=count,submission_journals=journals,related_accounting=allrows,journal_pins=jp,
            closed_cpu_seconds=cpu,closed_gpu_slice_seconds=gpu,cpu_admission_seconds=cpu+3600,
            input_pins=dict(predecessor=prior_pin,acceptance=pins[prefix+'-acceptance'],disposition=pins[prefix+'-disposition']))
        obs=dict(terminal_gate=True,root_present=True,parent_accounting=row,related_accounting=allrows,files=files,journal_pins=jp)
        disp=dict(job_id=job,remote_root=root,packet_sha256=pins[prefix+'-launch-packet'],submission_result_sha256=pins[prefix+'-result'],
            custody_bytes_verified=True,current_lineage_verified=True,errors=[],input_pin_mismatches=[],exact_prior_accounting_matched=True,
            exact_prior_and_current_journals_matched=True,observed_related_jobs=count,observed_submission_journals=journals,raw_files=len(files),raw_bytes=len(files))
        m.update({prefix+'-acceptance':dict(job_id=job),prefix+'-launch-packet':packet,prefix+'-result':result,prefix+'-disposition':disp,
            prefix+'-inventory':dict(returncode=0,python_source='ROOT='+repr(root)+'\nJOB='+repr(job)+'\n',stdout=json.dumps(obs)),prefix+'-baseline':base})
        previous=base;prior_pin=pins[prefix+'-baseline']
    return m,pins


def authority_fixture():
    pins={k:H(k) for k in ('gpu-packet','cpu-packet','cpu-acceptance','freeze','namespace-admission','cpu-phase-terminal')}
    attempts={p:H(p) for p in b.POLICIES}
    campaign={n:H(n) for n in ('campaign.json','source-plan-pins.json','start.json')}
    for i,p in enumerate(b.POLICIES):
        for n in ('attempt.json','attempt.candidate.json'):campaign[f'policy-{i:02d}/'+n]=attempts[p]
    paths=dict(cpu_inputs=b.CPU_ROOT+'/inputs',cpu_output=b.CPU_ROOT+'/cpu',campaign=b.CPU_ROOT+'/cpu/worker/artifacts/campaign',gpu_inputs=b.GPU_ROOT+'/inputs',gpu_output=b.GPU_ROOT+'/gpu')
    c=dict(cpu_output=paths['cpu_output'],campaign_root=paths['campaign'],cpu_output_excluded_directory='worker/artifacts/campaign',
        bundle_path='worker/artifacts/cpu-phase/campaign-bundle.json',bundle_sha256=H('bundle'),
        phase_terminal_path='worker/artifacts/cpu-phase/terminal.json',phase_terminal_sha256=pins['cpu-phase-terminal'],
        attempt_pins={p:attempts[p] for p in b.POLICIES[:5]},campaign_files={n:h for n,h in campaign.items() if n.split('/')[0] not in {'policy-05','policy-06','policy-07'}})
    c['cpu_output_files']={c['bundle_path']:c['bundle_sha256'],c['phase_terminal_path']:c['phase_terminal_sha256']}
    files={'opaque.json':H('opaque'),'cpu-packet.json':pins['cpu-packet'],'cpu-acceptance.json':pins['cpu-acceptance']}
    gp=dict(schema='ace-scientific-paired-campaign-gpu-v1',scope='development',resources=b.GPU_RESOURCES,limits=b.GPU_LIMITS,
        freeze_sha256=pins['freeze'],namespace_admission_sha256=pins['namespace-admission'],cpu_packet_sha256=pins['cpu-packet'],cpu_acceptance_sha256=pins['cpu-acceptance'],paths=paths,files=files)
    g=dict(schema='ace-scientific-campaign-gpu-acceptance-v1',status='development_gpu_phase_accepted',scope='development',
        gpu_packet_sha256=pins['gpu-packet'],cpu_packet_sha256=pins['cpu-packet'],cpu_acceptance_sha256=pins['cpu-acceptance'],job_id='46',
        allocation=dict(account='ucb736_asc2',partition='aa100',qos='gpu-testing',requested_cpus=4,allocated_cpus=4,memory_mib=15360,wall_seconds=2880,gpus=1,gpu_type='a100_3g.20gb',state='COMPLETED',exit_code='0:0'),
        gpu_output=paths['gpu_output'],campaign_root=paths['campaign'],gpu_output_files={n:H(n) for n in ('terminal.json','controller/phase/terminal.json')},
        campaign_files=campaign,bundle_sha256=c['bundle_sha256'],attempt_pins=attempts,accepted_checks=dict(scheduler=True,runtime=True,cleanup=True,custody=True),
        terminal_pins={n:H(n) for n in ('terminal.json','controller/phase/terminal.json')},scientific_qualified=False)
    term=dict(schema='ace-scientific-campaign-cpu-v1',status='cpu_phase_complete',numeric_training_updates=162,llm_training_updates=0,completed_cpu_cells=120,
        declared_cells=192,gpu_services_started=0,attempt_pins=c['attempt_pins'],private_scores_computed=False,scientific_qualified=False,retry_allowed=False)
    m={'gpu-acceptance':g,'gpu-packet':gp,'cpu-acceptance':c,'cpu-packet':dict(files={'opaque.json':H('opaque')},cpu_output=paths['cpu_output']),
       'gpu-result':dict(job_id='46'),'gpu-launch-packet':dict(new_input_files={'inputs/packet.json':pins['gpu-packet']},gpu_packet_sha256=pins['gpu-packet']),
       'cpu-phase-terminal':term,'training-acceptance':dict(dpo_steps=2)}
    return m,pins


class NewMetadata(unittest.TestCase):
    def test_finalizer_frozen_contract_and_job(self):
        source=(HERE/'interfaces'/b.ENTRY).read_bytes();freeze=b.decode((HERE/'interfaces/scientific-freeze.json').read_bytes())
        self.assertEqual(b.sha(source),b.ENTRY_SHA);self.assertEqual(b.sha(b.raw(freeze)),b.FIXED['freeze'])
        self.assertEqual(b.literal(source,'RESOURCES'),b.RESOURCES)
        self.assertEqual(b.literal(source,'GPU_ACCEPTANCE_KEYS'),b.GPU_ACCEPTANCE_KEYS)
        stage=next(s for s in freeze['stages'] if s['stage']=='finalize')
        self.assertEqual(stage['deadlines'],b.DEADLINES);self.assertEqual(stage['remote_root'],b.ROOT)
        f=freeze['stage_closure']['stages']['finalizer'];gp=freeze['stage_closure']['stages']['GPUcampaign']
        self.assertEqual(set(f['input_source_files']),set(gp['input_source_files'])|{b.ENTRY,b.OLD_GPU_ENTRY})
        job=b.job_script({'metadata':{'cpu-packet':{'runtime_source':'/fixture/runtime'}}},{'fixture':True},'fixture bootstrap').decode()
        self.assertIn('#SBATCH --time=00:20:00',job);self.assertIn('--cpus-per-task=2',job);self.assertIn('--mem=7680M',job)
        self.assertIn('--no-requeue',job);self.assertNotIn('--gres',job);self.assertNotIn('--admission',job);self.assertIn('/finalization',job)

    def test_full_history46_94_and_actual_budget(self):
        m,p=history_fixture();b.history(m,p);base=m['gpu-baseline'];result=b.budget(base)
        self.assertEqual(result['candidate_cpu_reservation'],2400);self.assertEqual(result['candidate_gpu_reservation'],0)
        self.assertEqual(result['total_cpu_seconds'],b.cost(base['related_accounting'])[0]+6000)
        for change in ('old-row','old-journal','new-row','pin','count','gpu-count'):
            bad=copy.deepcopy(m)
            if change=='old-row':bad['gpu-baseline']['related_accounting'][0][1]='changed'
            elif change=='old-journal':bad['gpu-baseline']['journal_pins']['/fixture/old/0.json']=H('changed')
            elif change=='new-row':bad['gpu-baseline']['related_accounting'][-1][5]='9999'
            elif change=='pin':bad['gpu-baseline']['input_pins']['acceptance']=H('unreviewed')
            elif change=='count':bad['gpu-baseline']['terminal_jobs']=45
            else:bad['gpu-baseline']['related_accounting'][-1][7]='cpu=4,gres/gpu=2,gres/gpu:a100_3g.20gb=1'
            with self.subTest(change=change),self.assertRaises((ValueError,KeyError)):b.history(bad,p)

    def test_gpu_independent_acceptance_strict_no_partial_success(self):
        m,p=authority_fixture();b.gpu_authority(m,p)
        for change in ('unknown-key','check-false','check-numeric','wrong-job','wall','bundle','cpu-mutation','partial-attempt','early-finalization','terminal-pin'):
            bad=copy.deepcopy(m);g=bad['gpu-acceptance']
            if change=='unknown-key':g['required_findings']=0
            elif change=='check-false':g['accepted_checks']['runtime']=False
            elif change=='check-numeric':g['accepted_checks']['custody']=1
            elif change=='wrong-job':g['job_id']='123'
            elif change=='wall':g['allocation']['wall_seconds']=2890
            elif change=='bundle':g['bundle_sha256']=H('different')
            elif change=='cpu-mutation':g['campaign_files']['policy-00/attempt.json']=H('different')
            elif change=='partial-attempt':del g['attempt_pins']['dpo_llm']
            elif change=='early-finalization':g['campaign_files']['private/summary.json']=H('report')
            else:g['terminal_pins']['terminal.json']=H('missing')
            with self.subTest(change=change),self.assertRaises((ValueError,KeyError)):b.gpu_authority(bad,p)

    def test_exact_six_root_rosters_and_empty_directory_guard(self):
        m,p=authority_fixture();p['namespace-claim']=H('claim');g=m['gpu-acceptance'];c=m['cpu-acceptance']
        data={'namespace.json':p['namespace-claim'],'original-namespace.json':b.ORIGINAL_CLAIM_SHA}
        data.update({'launch/gpu/'+n:h for n,h in g['gpu_output_files'].items()})
        data.update({'cpu/cpu/'+n:h for n,h in c['cpu_output_files'].items()})
        data.update({'cpu/cpu/worker/artifacts/campaign/'+n:h for n,h in g['campaign_files'].items()})
        files={n:dict(bytes=7,sha256=h) for n,h in data.items()};dirs=set(b.RAW_ROOTS)
        for n in files:
            dirs.update(str(q) for q in Path(n).parents if str(q)!='.')
        obs=dict(files=files,directories=sorted(dirs));m['gpu-disposition']=dict(remote_files=files,remote_directories=sorted(dirs))
        ctx=dict(metadata=m,pins=p,observations={'gpu':obs});b.accepted_rosters(ctx)
        bad=copy.deepcopy(ctx);extra='cpu/cpu/worker/artifacts/campaign/private'
        bad['observations']['gpu']['directories'].append(extra);bad['metadata']['gpu-disposition']['remote_directories'].append(extra)
        with self.assertRaises(ValueError):b.accepted_rosters(bad)
        bad=copy.deepcopy(ctx);bad['observations']['gpu']['files']['extra.bin']=dict(bytes=1,sha256=H('x'))
        bad['metadata']['gpu-disposition']['remote_files']=bad['observations']['gpu']['files']
        with self.assertRaises(ValueError):b.accepted_rosters(bad)

    def test_final_packet_only_appends_opaque_prior_inputs(self):
        # Fabricated source constants/opaque payloads, no project code executes.
        keys=b.literal((HERE/'interfaces'/b.ENTRY).read_bytes(),'PACKET_KEYS')
        source=('VERSION='+repr(b.VERSION)+'\nRESOURCES='+repr(b.RESOURCES)+'\nGPU_ACCEPTANCE_KEYS='+repr(b.GPU_ACCEPTANCE_KEYS)+'\nPACKET_KEYS='+repr(keys)+'\n').encode()
        prior={'opaque.json':b'not JSON: private sentinel','qualify_llm_intervention_runtime_metadata.py':b"BOOTSTRAP='fixture only'\n"}
        additions={b.ENTRY:source,b.OLD_GPU_ENTRY:b'# metadata helper fixture\n'}
        allsources={n:v for n,v in {**prior,**additions}.items() if n.endswith('.py')}
        loaded=sorted(allsources);srcpins={n:b.sha(v) for n,v in allsources.items()}
        f=dict(source_pins={'scripts/research/'+n:h for n,h in srcpins.items()},stage_closure={'stages':{
            'GPUcampaign':dict(source_names=sorted(n for n in prior if n.endswith('.py')),input_source_files={n:b.sha(v) for n,v in prior.items() if n.endswith('.py')}),
            'finalizer':dict(source_names=loaded,input_source_files=srcpins,entrypoint_sha256=b.sha(source),helper_pins={b.OLD_GPU_ENTRY:b.sha(additions[b.OLD_GPU_ENTRY])})}})
        supplied={'gpu-packet':b'original packet bytes','gpu-acceptance':b'original acceptance bytes'}
        pins={n:H(n) for n in ('cpu-packet','cpu-acceptance','freeze','namespace-admission')};pins.update({n:b.sha(v) for n,v in supplied.items()})
        gp=dict(files={n:b.sha(v) for n,v in prior.items()},source_names=f['stage_closure']['stages']['GPUcampaign']['source_names'],paths={'campaign':'/fixture/original/campaign'})
        ctx=dict(metadata={'freeze':f,'gpu-packet':gp},pins=pins)
        with patch.object(b,'ENTRY_SHA',b.sha(source)),patch.object(b,'committed',side_effect=lambda commit,path:allsources[Path(path).name]):
            result,data,bootstrap=b.make_packet(ctx,prior,supplied)
            self.assertEqual(set(result),keys);self.assertEqual(data['opaque.json'],prior['opaque.json'])
            self.assertEqual(data['gpu-acceptance.json'],supplied['gpu-acceptance']);self.assertEqual(bootstrap,'fixture only')
            bad=dict(prior);bad['opaque.json']=b'mutated'
            with self.assertRaises(ValueError):b.make_packet(ctx,bad,supplied)

    def test_exclusive_files_and_independent_opaque_rosters(self):
        with tempfile.TemporaryDirectory(prefix='finalizer-fabricated-') as d:
            root=Path(d);(root/'empty').mkdir();b.write(root/'raw.bin',b'\x00\xffprivate fixture')
            v=b.tree(root);self.assertEqual(v['directories'],['empty'])
            with self.assertRaises(FileExistsError):b.write(root/'raw.bin',b'overwrite')
            os.link(root/'raw.bin',root/'alias.bin')
            with self.assertRaises(ValueError):b.tree(root)
            (root/'alias.bin').unlink();(root/'link').symlink_to(root/'raw.bin')
            with self.assertRaises(ValueError):b.tree(root)

    def test_cap_rejects_unknown_gpu_or_excess_budget(self):
        m,p=history_fixture();base=copy.deepcopy(m['gpu-baseline']);base['outstanding_reservation']=1
        with self.assertRaises(ValueError):b.budget(base)
        base=copy.deepcopy(m['gpu-baseline']);base['related_accounting'][0][5]='30000'
        cpu,gpu=b.cost(base['related_accounting']);base.update(closed_cpu_seconds=cpu,closed_gpu_slice_seconds=gpu,cpu_admission_seconds=cpu+3600)
        with self.assertRaises(ValueError):b.budget(base)
        with self.assertRaises(ValueError):b.cost([['1','fixture','FAILED','1:0','4','7','ucb736_asc2','cpu=4,gres/gpu=unknown']])

    def test_once_only_submission_and_monitor_interface_ast(self):
        launcher=(HERE/'launch_once.py').read_text();monitor=(HERE/'monitor_once.py').read_text()
        remote=b.literal(launcher,'REMOTE_COMMON');ast.parse(remote)
        self.assertEqual(launcher.count("['sbatch','--parsable'"),1)
        self.assertLess(launcher.index("write(root/'submission.intent.json',intent)"),launcher.index("['sbatch','--parsable'"))
        self.assertIn("fresh=reconcile();verify()",launcher);self.assertIn("packet['upstream_rosters']==ctx['upstream_rosters']",launcher)
        for x in ('ace-delivery-controller.lock','ace-foundation-controller.lock','blind_retry_forbidden=True','review_raw==b.committed','launcher_failure.json'):
            self.assertIn(x,launcher)
        self.assertIn(">=1200",remote);self.assertIn("'cpu-normal'",remote);self.assertNotIn("'sinfo'",remote)
        self.assertIn("C['upstream_rosters']",remote);self.assertIn("expected['directories']",remote)
        rm=b.literal(monitor,'REMOTE');ast.parse(rm)
        self.assertNotIn('json.loads',rm);self.assertNotIn('private/summary',rm);self.assertNotIn('sbatch',rm)
        self.assertIn('finalization/attempt.json',rm);self.assertIn('finalization/worker/terminal.json',rm)
        self.assertIn('finalization.candidate.json',rm);self.assertIn('acceptance=False',rm)
        self.assertIn('with b.coordinator_locks():',monitor)

if __name__=='__main__':unittest.main(verbosity=2)
