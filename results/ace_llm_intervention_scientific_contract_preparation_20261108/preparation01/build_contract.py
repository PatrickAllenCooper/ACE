"""Capture one committed scientific contract candidate. No draws or submissions."""
from pathlib import Path
import argparse, ast, datetime, hashlib, json, shutil, subprocess

REPO=Path('/Users/pat/code/ACE')
PARENT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
PROPOSAL='results/ace_llm_intervention_screen_protocol_preparation_20261107/test-attempt01/fixtures/prospective-proposal.json'
DESIGN='docs/development/guidance/ace_llm_intervention_scientific_contract_2026-11-08.md'
ACCOUNTING='results/ace_llm_intervention_scientific_cpu_preparation_20261108/admission-baseline01/accounting.json'
def sha(b):return hashlib.sha256(b).hexdigest()
def raw(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def need(ok,msg):
    if not ok:raise ValueError(msg)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('source-commit','closure-map','closure-map-sha256','reviews','reviews-sha256','candidate'):
        ap.add_argument('--'+n,required=True)
    a=ap.parse_args();rev=a.source_commit
    need(len(rev)==40 and all(c in '0123456789abcdef' for c in rev),'full source commit required')
    def committed(path):
        value=subprocess.check_output(['git','--no-replace-objects','show',rev+':'+path],cwd=REPO)
        need(value==(REPO/path).read_bytes(),'working/committed bytes differ: '+path)
        return value
    closure_raw=Path(a.closure_map).read_bytes();reviews_raw=Path(a.reviews).read_bytes()
    need(sha(closure_raw)==a.closure_map_sha256 and sha(reviews_raw)==a.reviews_sha256,'independent map pins differ')
    closure=json.loads(closure_raw);reviews=json.loads(reviews_raw)
    # The independently reviewed closure map specifies its exact union and per-stage roles.
    source_pins=closure['source_pins'];need(type(source_pins) is dict and source_pins,'source closure required')
    captured={}
    for path,pin in source_pins.items():
        value=committed(path);need(sha(value)==pin,'source pin differs: '+path);captured[path]=value
    def literal(path,name):
        tree=ast.parse(captured[path])
        for n in tree.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):
                return ast.literal_eval(n.value)
        raise ValueError('missing declared constant '+name)
    runtime=literal('scripts/research/llm_intervention_model_service.py','RUNTIME')
    model_files=literal('scripts/research/prepare_llm_intervention_campaign_cpu.py','MODEL_FILES')
    runtime_source=literal('scripts/research/qualify_llm_intervention_runtime_metadata.py','RUNTIME_SOURCE')
    review_data={}
    for path,pin in reviews.items():
        value=committed(path);need(sha(value)==pin,'review pin differs: '+path);review_data[path]=value
    proposal_raw=committed(PROPOSAL);proposal=json.loads(proposal_raw)
    need(sha(proposal_raw)=='8e3214958f671dfee5ef0a2d7adc70e8f5cdc26edf0fbeff02e7925921ff13da','original proposal changed')
    need(all(source_pins.get(p)==h for p,h in proposal['source_pins'].items()),'original protocol pins changed')
    design_raw=committed(DESIGN);accounting_raw=committed(ACCOUNTING)
    need(sha(accounting_raw)=='c5eb497d133a803f12b92ff7af07420cf4856d24e174162b9f634c4e0e27631f','accounting baseline differs')
    stage_rows=[
        ('realize','supervise_llm_intervention_scientific_cpu.py',300,2,7680,0,dict(startup=120,work=260,cleanup=280,hard=290)),
        ('collect','supervise_llm_intervention_scientific_cpu.py',900,2,7680,0,dict(startup=120,work=850,cleanup=875,hard=890)),
        ('tokenize','prepare_llm_intervention_scientific_training_cpu.py',600,3,11520,0,dict(startup=120,work=550,cleanup=570,hard=590)),
        ('train','supervise_llm_intervention_scientific_training_gpu.py',2700,4,15360,1,dict(startup=240,worker=2620,soft=2630,cleanup=2645,hard=2670,progress=180,telemetry=90)),
        ('cpu_campaign','prepare_llm_intervention_scientific_campaign_cpu.py',6000,2,7680,0,dict(startup=180,work=5940,hard=5980)),
        ('gpu_campaign','qualify_llm_intervention_scientific_campaign_gpu.py',2880,4,15360,1,dict(session_reserve_seconds=600,session_cap_seconds=1200,work_seconds=2820,final_seconds=2860)),
        ('finalize','finalize_llm_intervention_scientific_campaign_cpu.py',1200,2,7680,0,dict(startup=120,work=1140,cleanup=1170,hard=1190)),
    ]
    roots={
        'realize':'scientific_realization','collect':'scientific_collection','tokenize':'scientific_tokenization',
        'train':'scientific_training_gpu','cpu_campaign':'scientific_campaign_cpu',
        'gpu_campaign':'scientific_campaign_gpu','finalize':'scientific_finalization'}
    stages=[]
    for key,entry,wall,cpus,mem,gpu,deadlines in stage_rows:
        stages.append(dict(stage=key,entry=entry,entry_sha256=source_pins['scripts/research/'+entry],
            cpus=cpus,memory_mib=mem,gpus=gpu,gpu_type='a100_3g.20gb' if gpu else None,
            wall_seconds=wall,deadlines=deadlines,account='ucb736_asc2',partition='aa100' if gpu else 'acpu',
            qos='gpu-testing' if gpu else 'cpu-normal',nodes=1,tasks=1,requeue=False,max_restarts=0,
            remote_root='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_'+roots[key]+'_20261108_01',
            repository_root='results/ace_llm_intervention_'+roots[key]+'_20261108',
            exclusive_external_root=str(PARENT/('ACE-LLM-intervention-'+roots[key].replace('_','-')+'-20261108-01'))))
    need(11040+3600+sum(s['cpus']*s['wall_seconds'] for s in stages)==55560,'CPU total differs')
    need(1601+sum(s['gpus']*s['wall_seconds'] for s in stages)==7181,'GPU total differs')
    contract=dict(schema='ace-first-trained-intervention-screen-contract-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=rev,source_pins=source_pins,stage_closure=closure,
        design=dict(path=DESIGN,sha256=sha(design_raw)),component_reviews=reviews,
        original_proposal=dict(path=PROPOSAL,sha256=sha(proposal_raw),semantic_sha256='585536b1408054ecebc7b7bb0e80055fa153b5bc7a699d8f2927db25ee745ea8'),
        training_world_ids=list(range(202611070000,202611070032)),development_world_ids=list(range(202611070100,202611070124)),
        confirmation_executed=False,scientific_scope='development screen, not confirmation',
        policies=['uniform_random','menu_coverage','local_information','target_risk','learned_ranker','frozen_llm','sft_llm','dpo_llm'],
        training=dict(records=160,all_pairs=3584,sft_steps=160,dpo_steps_min=1,dpo_steps_max=128,
            selection='minimum hash among strict comparisons per nonterminal record, lexical tiebreak',
            zero_strict_pairs='fail without rescue',seed=109,rank=8,alpha=16,effective_multiplier=2,
            learning_rate=.0001,beta=.1,weight_decay=0,optimizer_betas=[.9,.999],optimizer_epsilon=1e-8,
            continuous_optimizer=True,reference='fixed post-SFT',arithmetic='qwen_screen_singletons',
            training_layout='canonical_selected_unpadded',base_dtype='bfloat16',adapter_dtype='float32',
            numerical_hidden_units=128,numerical_same_exposure=True,training_prompt_max_tokens=4744,
            context_tokens=8192,max_new_tokens=32,no_truncation=True,checkpoint_selection='final SFT and final DPO only'),
        endpoints=dict(cells=192,descriptive_pairs=28,initial_cost_per_cell=8,acquisition_cap=24,
            normalized_mse_tolerance=.04,public_target_scale=2,queries=8,
            retain_ever_and_final_attainment=True,retain_actual_spending=True,
            private_policy_feedback=False,training_cost_separate=True,confirmatory_tests=False),
        response_limits=dict(training_attempts=9216,training_cost_units=10240,development_cost_units=6144),
        stages=stages,numeric_internal_seconds=50,numeric_timeout_seconds=55,
        admission_accounting=dict(baseline_sha256=sha(accounting_raw),closed_cpu_seconds=11040,
            closed_gpu_seconds=1601,prep_cpu_allowance=3600,max_projected_cpu_seconds=55560,max_projected_gpu_seconds=7181,
            cpu_cap=57600,gpu_cap=7200,new_reservations=0),
        runtime_archive_sha256='6815ab9784b6025b8bb4439e40a1a012b9697d434bfc83bd5f36e27abc262d98',
        runtime_manifest_sha256='b44610f638f0873da8588f1cf23039638203a855959a6ed12cf0ec6c2551d63a',
        runtime=runtime,runtime_source=runtime_source,model_files=model_files,
        runtime_archive_path='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_staged_runtime_20261107_01/qualification/packed/runtime.tar',
        runtime_manifest_path='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_staged_runtime_20261107_01/qualification/packed/manifest.json',
        model_directory='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_20261102_01/model',
        model_revision='c1899de289a04d12100db370d81485cdf75e47ca',
        sizing_acceptance_sha256='555b294fccd604f49bb6634c6d07eac8206bcfaefc710bc1523d71d5dc3047f5',
        independent_contract_adoption_required=True,namespace_admission_required=True,
        independent_actual_acceptance_between_stages=True,scientific_worlds_generated=0,
        no_automatic_retry=True,publication_authorized=False)
    local=REPO/'results/ace_llm_intervention_scientific_contract_20261108'/a.candidate
    external=PARENT/('ACE-LLM-intervention-scientific-contract-20261108-'+a.candidate)
    need(a.candidate.isalnum(),'simple exclusive candidate name required')
    local.parent.mkdir(parents=True,exist_ok=True);external.mkdir(mode=0o700,exist_ok=False)
    def save(name,value):
        p=external/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(value)
    for n,b in captured.items():save('source/'+n,b)
    for n,b in review_data.items():save('reviews/'+Path(n).name,b)
    for n,b in {'scientific-freeze.json':raw(contract),'design.md':design_raw,'proposal.json':proposal_raw,
                'accounting.json':accounting_raw,'closure-map.json':closure_raw,'review-pins.json':reviews_raw,
                'build_contract.py':Path(__file__).read_bytes()}.items():save(n,b)
    files={p.relative_to(external).as_posix():sha(p.read_bytes()) for p in sorted(external.rglob('*')) if p.is_file()}
    save('inventory.json',raw(dict(files=files,source_commit=rev,scope='unadopted contract candidate only',scientific_execution=False)))
    shutil.copytree(external,local)
    for p in external.rglob('*'):
        if p.is_file():need(p.read_bytes()==(local/p.relative_to(external)).read_bytes(),'mirror mismatch')
    print(json.dumps(dict(path=str(local),sha256=sha(raw(contract)),files=len(files)+1,source_files=len(source_pins),adopted=False)))

if __name__=='__main__':main()
