"""One admitted scientific CPU campaign stage: matched ranker, then five arms.

Parent supplies independent packet/source/admission pins, the combined freeze
made before any scientific draw, accepted training/CPU/realization inputs and
three prospective development services. No CLI defaults authorize execution.
Only supplied 24 development descriptors are decoded; no worlds are generated.
No tokenizer or LLM weights are loaded. Private descriptors stay in controller
artifacts; task subprocesses receive public envelopes and fixed policy args.

Reuse the old campaign's staging, runtime observations and explicit-deadline
owned cleanup primitive without editing modules or patching their globals. A new
small supervisor retains the live child through outer finalization. The closed
scientific assembler and campaign mechanics are unchanged. All outputs are
exclusive, all failures retained; success still requires independent acceptance.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
import types

ENTRY = 'prepare_llm_intervention_scientific_campaign_cpu.py'
OLD = 'prepare_llm_intervention_campaign_cpu.py'
OLD_PIN = '850ca3d4d7ac4799f68d74bd2d0b1196d10809751a4bdf878633122fc6a9c2c4'
VERSION = 'ace-scientific-campaign-cpu-v1'
LINEAGE = 'llm_intervention_scientific_training_lineage.py'
BUILDER = 'llm_intervention_scientific_campaign.py'
OWNED = 'llm_intervention_cpu_owned_supervision.py'
PROPOSAL_BYTE_SHA = '8e3214958f671dfee5ef0a2d7adc70e8f5cdc26edf0fbeff02e7925921ff13da'
PROPOSAL_SHA = '585536b1408054ecebc7b7bb0e80055fa153b5bc7a699d8f2927db25ee745ea8'
MAX_CAPTURE = 256*1024**2


def _base():
    path=Path(__file__).with_name(OLD)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:raw=f.read(1024**2+1)
    if hashlib.sha256(raw).hexdigest()!=OLD_PIN:raise ValueError('closed campaign helper differs')
    m=types.ModuleType('_scientific_campaign_base');m.__file__=str(path)
    exec(compile(raw,str(path),'exec'),vars(m))
    return m


B = _base()
need,sha,pin,canonical,digest,decode,save = B.need,B.sha,B.pin,B.canonical,B.digest,B.decode,B.save
CPU_POLICIES,LLM_POLICIES,NUMERIC_POLICIES = B.CPU_POLICIES,B.LLM_POLICIES,B.NUMERIC_POLICIES
SOURCE_NAMES = B.SOURCE_NAMES | {ENTRY,LINEAGE,BUILDER,OWNED}
NUMERIC_SOURCE_NAMES = B.NUMERIC_SOURCE_NAMES
HELPER_PINS = {**B.HELPER_PINS,OLD:OLD_PIN,OWNED:'1c08423bb8fdd69ad637fbf5366978ea4a630c7ef206cc1b89ab30e3617a14e8',
    LINEAGE:'f3b328203d5758418d0afd677513822e8d7f134acca10545aab2d3fda1d3fc35',
    BUILDER:'e3072b8a4c18ef459154fc71ebec8874992d4cee7c829e55b4203cf5111565d8',
    'llm_intervention_service_runtime.py':'117dc22bd56889906612bff0c975095918e062974fa537757b3e93c981dbfbc0'}
DATA_NAMES = {'proposal.json','scientific-freeze.json','realization-terminal.json','realization-acceptance.json',
    'gpu-training-acceptance.json','cpu-preparation-acceptance.json','prepared.json','numerical-plan.json',
    'engine-terminal.json','sft.checkpoint.json','final.checkpoint.json','sft.adapter.pt','final.adapter.pt',
    'numeric-bindings.json'} | {f'descriptor-{i:03d}.json' for i in range(24)} | {
    f'{p}.{kind}.json' for p in LLM_POLICIES for kind in ('freeze','launch')}
INPUT_NAMES = SOURCE_NAMES | DATA_NAMES | set(HELPER_PINS)
PACKET_KEYS = {'schema','scope','source_commit','files','source_names','cpu_output','archive','manifest',
    'runtime_source','runtime','model_directory','model_files','resources','deadlines','numeric_timeout_seconds',
    'numeric_work_seconds','protocol_source_pins','freeze_sha256','namespace_admission_sha256','world_pins',
    'expected_training','numerical_plan_sha256','persistent_specs','services','service_identity_pins',
    'outer_provenance_receipt_sha256'}
BASE_RESOURCES = dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=2,memory_mib=7680,gpus=0,requeue=False)
TRAIN_ROSTER = tuple(f'policy_train/{w}/snapshot-{i:03d}' for w in range(202611070000,202611070032) for i in range(5))


def read(path,maximum=MAX_CAPTURE):return B.read(path,maximum)
def capture(path,expected,maximum=MAX_CAPTURE):return B.capture(path,expected,maximum)
def paths(packet):return B.paths(packet)
def failure(exc):return dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),retry_allowed=False,
    scientific_qualified=False,response_accounting_source='original campaign charged journals; never infer zero from failure')


def resource_contract(p):
    r=p['resources'];d=p['deadlines']
    need(type(r) is dict and set(r)==set(BASE_RESOURCES)|{'wall_seconds'}
        and all(type(r[k]) is type(v) and r[k]==v for k,v in BASE_RESOURCES.items())
        and type(r['wall_seconds']) is int and 60<=r['wall_seconds']<=6000,'explicit bounded CPU allocation required')
    need(type(d) is dict and set(d)=={'startup','work','hard'} and all(type(v) is int for v in d.values())
        and 0<d['startup']<d['work']<d['hard']<r['wall_seconds'] and d['hard']-d['work']>=10,'caller absolute deadlines/cleanup margin required')
    need(type(p['numeric_timeout_seconds']) is int and type(p['numeric_work_seconds']) is int
        and 0<p['numeric_work_seconds']<p['numeric_timeout_seconds']<=60,'caller numeric decision deadlines required')


def admission_contract(p,a,packet_pin):
    expected=dict(schema=VERSION+'/admission',packet_sha256=packet_pin,source_commit=p['source_commit'],
        freeze_sha256=p['freeze_sha256'],namespace_admission_sha256=p['namespace_admission_sha256'],
        stage='matched_ranker_and_development_cpu',output=p['cpu_output'],resources=p['resources'],deadlines=p['deadlines'],
        full_contract_frozen_before_first_scientific_draw=True,source_runtime_scheduler_qualified=True,
        namespace_owned=True,accounting_admitted=True,independent_acceptances_verified=True,retry_allowed=False)
    need(a==expected,'independent full-contract/stage admission required')


def accepted_inputs(p,data,lineage_api):
    """Stdlib-only authority/byte bindings; no descriptors or tensors initialized."""
    kwargs={arg:data[name] for arg,name in (
        ('acceptance_raw','gpu-training-acceptance.json'),('prepared_raw','prepared.json'),
        ('engine_terminal_raw','engine-terminal.json'),('sft_checkpoint_raw','sft.checkpoint.json'),
        ('sft_adapter_raw','sft.adapter.pt'),('final_checkpoint_raw','final.checkpoint.json'),('final_adapter_raw','final.adapter.pt'))}
    lineage=lineage_api.admit_training(**kwargs,acceptance_sha256=p['files']['gpu-training-acceptance.json'],expected=p['expected_training'])
    expected=p['expected_training'];cpu=decode(data['cpu-preparation-acceptance.json'])
    need(expected['scientific_freeze_sha256']==p['freeze_sha256']==sha(data['scientific-freeze.json'])
        and expected['cpu_preparation_acceptance_sha256']==sha(data['cpu-preparation-acceptance.json'])
        and expected['model_files_sha256']==digest(p['model_files'])
        and expected['runtime_contract_sha256']==digest(p['runtime']),'accepted frozen CPU/model/runtime lineage differs')
    prepared=decode(data['prepared.json'])
    cpu_keys={'schema','status','scope','required_findings','job_id','cpu_packet_sha256','prepared_sha256','plan_sha256',
        'freeze_sha256','namespace_admission_sha256','complete_custody_verified','runtime_qualified_for_this_cpu_execution',
        'owned_process_cleanup_verified','training_execution_qualified','scientific_qualified',
        'numerical_plan_file_sha256','numerical_plan_sha256','exposure_sha256'}
    need(set(cpu)==cpu_keys and cpu['schema']=='ace-scientific-training-preparation-acceptance-v1'
        and cpu['scope']=='policy_train' and cpu['status']=='accepted' and type(cpu['required_findings']) is int and cpu['required_findings']==0
        and type(cpu['job_id']) is str and re.fullmatch('[1-9][0-9]*',cpu['job_id'])
        and pin(cpu['cpu_packet_sha256']) and cpu['cpu_packet_sha256']==prepared['training_plan']['outer_provenance_receipt_sha256']
        and cpu['plan_sha256']==expected['training_plan_sha256']
        and cpu['freeze_sha256']==p['freeze_sha256'] and cpu['namespace_admission_sha256']==p['namespace_admission_sha256']
        and cpu['training_execution_qualified'] is cpu['scientific_qualified'] is False
        and all(cpu.get(k) is True for k in ('complete_custody_verified','runtime_qualified_for_this_cpu_execution','owned_process_cleanup_verified'))
        and cpu.get('prepared_sha256')==p['files']['prepared.json']==expected['prepared_sha256']
        and cpu.get('numerical_plan_file_sha256')==p['files']['numerical-plan.json']
        and cpu.get('numerical_plan_sha256')==p['numerical_plan_sha256']
        and cpu.get('exposure_sha256')==expected['exposure_sha256'],'independently accepted numerical CPU plan required')
    n=decode(data['numerical-plan.json']);d=prepared['exposure']['counts']['dpo_steps']
    need(digest(n)==p['numerical_plan_sha256'] and n['scope']=='policy_train' and n['record_roster']==list(TRAIN_ROSTER)
        and n['exposure']==prepared['exposure'] and n['exposure_sha256']==lineage['exposure_sha256']
        and n['checkpoint_steps']==[159,159+d] and n['record_pins']==prepared['exposure']['record_pins']
        and n['supervision_receipt_pins']==prepared['exposure']['supervision_receipt_pins']
        and n['raw_records_utf8']=={r['record_id']:r['raw_record_utf8'] for r in prepared['training_plan']['records']}
        and n['outer_provenance_receipt_sha256']==prepared['training_plan']['outer_provenance_receipt_sha256']
        and n['scientific_provenance_qualified'] is False and n['retry_allowed'] is False,'same complete matched numerical plan required')
    proposal=decode(data['proposal.json']);terminal=decode(data['realization-terminal.json']);accepted=decode(data['realization-acceptance.json'])
    need(sha(data['proposal.json'])==PROPOSAL_BYTE_SHA and digest(proposal)==PROPOSAL_SHA
        and proposal['source_pins']==p['protocol_source_pins'],'original proposal/protocol required')
    need(accepted.get('status')=='accepted' and type(accepted.get('required_findings')) is int and accepted['required_findings']==0
        and accepted.get('complete_custody_verified') is True
        and accepted.get('realization_terminal_sha256')==p['files']['realization-terminal.json']
        and accepted.get('freeze_sha256')==p['freeze_sha256']
        and accepted.get('namespace_admission_sha256')==p['namespace_admission_sha256'],'accepted original realization required')
    need(terminal['status']=='complete' and terminal['worlds_generated']==56 and terminal['environment_responses']==terminal['policy_updates']==0
        and terminal['retry_allowed'] is False and terminal['proposal_sha256']==PROPOSAL_SHA
        and terminal['source_pins']==p['protocol_source_pins'] and terminal['freeze_sha256']==p['freeze_sha256']
        and terminal['namespace_admission_sha256']==p['namespace_admission_sha256'],'complete realization lineage differs')
    rows=proposal['training_worlds']+proposal['development_worlds'];entries=terminal['entries']
    need(len(rows)==len(entries)==56 and len(proposal['development_worlds'])==24,'original complete descriptor roster required')
    for i,(e,r) in enumerate(zip(entries,rows)):
        need(e['world_id']==r['world_id'] and e['scope']==r['scope'] and e['status']=='complete'
            and e['path']==f'world-{i:03d}.json' and pin(e['descriptor_sha256']) and pin(e['file_sha256']),'descriptor entry differs')
    dev=proposal['development_worlds'];need(set(p['world_pins'])=={str(r['world_id']) for r in dev},'exact 24 development pins required')
    for i,r in enumerate(dev):
        e=entries[32+i];value=decode(data[f'descriptor-{i:03d}.json'])
        need(e['file_sha256']==p['files'][f'descriptor-{i:03d}.json']
            and e['descriptor_sha256']==p['world_pins'][str(r['world_id'])]==digest(value)
            and value['world_id']==r['world_id'] and value['scope']=='development','original development-only payload differs')
    return lineage


def service_contracts(root,p,data,service,diagnostic,lineage,launch):
    need(set(p['persistent_specs'])==set(p['services'])==set(p['service_identity_pins'])==set(LLM_POLICIES),'three prospective services required')
    destinations=[];contracts=[]
    for role in LLM_POLICIES:
        freeze=service.validate_freeze(decode(data[role+'.freeze.json']))
        need(freeze['scope']=='development' and freeze['model_directory']==p['model_directory'] and freeze['model_files']==p['model_files']
            and freeze['sources']=={n:p['files'][n] for n in service.SOURCE_NAMES}
            and freeze['limits']['max_requests']==576,'fixed development service model/source/request bound differs')
        if role=='frozen_llm':need(freeze['treatment']=='frozen' and freeze['adapter'] is None,'pristine frozen service required')
        else:
            role_value=lineage['roles'][role];stage='sft' if role=='sft_llm' else 'final'
            need(freeze['treatment']==role_value['treatment'] and freeze['adapter']==dict(file=stage+'.adapter.pt',
                sha256=role_value['adapter_sha256'],training_receipt=stage+'.checkpoint.json',
                training_receipt_sha256=role_value['checkpoint_sha256'],training_scope='policy_train'),'scientific service checkpoint role differs')
        identity=service.identity(freeze)
        settings=dict(identity=identity,startup_seconds=freeze['limits']['startup_seconds']+5,cleanup_seconds=5,max_requests=576)
        need(p['services'][role]==settings and p['service_identity_pins'][role]==digest(identity),'independent service identity differs')
        c=launch.validate_contract(decode(data[role+'.launch.json']))
        need(c['policy_id']==role and c['entry']==str(root/B.SERVICE) and c['entry_sha256']==p['files'][B.SERVICE]
            and c['freeze']==str(root/(role+'.freeze.json')) and c['freeze_sha256']==p['files'][role+'.freeze.json']
            and c['archive_sha256']==p['archive']['sha256'] and c['manifest_sha256']==p['manifest']['sha256']
            and c['bootstrap']==diagnostic.BOOTSTRAP,'prospective service launch differs')
        argv=[str(Path(p['runtime_source'])/'bin/python'),'-I','-B','-c',diagnostic.BOOTSTRAP,str(root/B.LAUNCH),p['files'][B.LAUNCH],
              '--contract',str(root/(role+'.launch.json')),'--contract-sha256',p['files'][role+'.launch.json']]
        need(p['persistent_specs'][role]==dict(policy_id=role,kind='persistent',command=argv,
            timeout_seconds=freeze['limits']['request_seconds']+5,identity_sha256=digest(identity),
            policy_provenance_sha256=p['files'][role+'.freeze.json']),'fixed prospective persistent command differs')
        contracts.append(c);destinations.extend((c['worker_output'],c['launch_output']))
    need(len({(c['binding_path'],c['stage_path']) for c in contracts})==1,'three services must share prospective stage binding')
    B.separate([root,p['cpu_output'],p['model_directory'],*destinations])
    for c in contracts:B.separate([root,p['cpu_output'],c['binding_path'],c['stage_path']])


def packet_shape(p,service,diagnostic):
    """Pure declaration gate for future GPU/finalizer callers; no byte admission."""
    need(type(p) is dict and set(p)==PACKET_KEYS and p['schema']==VERSION and p['scope']=='development'
        and p['source_names']==sorted(SOURCE_NAMES) and set(p['files'])==INPUT_NAMES
        and all(pin(h) for h in p['files'].values()),'exact scientific campaign packet/source roster required')
    resource_contract(p)
    need(type(p['source_commit']) is str and re.fullmatch('[0-9a-f]{40}',p['source_commit'])
        and all(p['files'].get(n)==h for n,h in HELPER_PINS.items()),'full source commit/closed helper pins required')
    need(p['runtime']==service.RUNTIME and p['runtime_source']==diagnostic.RUNTIME_SOURCE and p['model_files']==B.MODEL_FILES
        and B.absolute(p['cpu_output']) and B.absolute(p['model_directory']) and pin(p['outer_provenance_receipt_sha256']),'fixed runtime/model/output differs')
    for key,h in (('archive',diagnostic.ARCHIVE_SHA),('manifest',diagnostic.MANIFEST_SHA)):
        need(set(p[key])=={'path','sha256'} and p[key]['sha256']==h and B.absolute(p[key]['path']),'accepted runtime archive differs')
    need(pin(p['namespace_admission_sha256']) and p['files']['scientific-freeze.json']==p['freeze_sha256']
        ==p['expected_training']['scientific_freeze_sha256']
        and p['files']['cpu-preparation-acceptance.json']==p['expected_training']['cpu_preparation_acceptance_sha256']
        and p['files']['prepared.json']==p['expected_training']['prepared_sha256']
        and p['files']['proposal.json']==PROPOSAL_BYTE_SHA and pin(p['numerical_plan_sha256']),
        'original combined freeze/accepted CPU input pins required')
    need(type(p['protocol_source_pins']) is dict and len(p['protocol_source_pins'])==12,'original protocol twelve-file map required')
    for path,h in p['protocol_source_pins'].items():need(p['files'].get(Path(path).name)==h,'original protocol source closure differs')
    need(set(p['world_pins'])=={str(w) for w in range(202611070100,202611070124)}
        and all(pin(v) for v in p['world_pins'].values()),'exact independently pinned 24 development descriptors required')


def authenticate(args):
    root=Path(args.packet).absolute().parent;p=decode(capture(args.packet,args.packet_sha256,2*1024**2))
    need(type(p) is dict and set(p)==PACKET_KEYS and set(p['files'])==INPUT_NAMES,'exact flat packet roster required before capture')
    need(p['source_commit']==args.source_commit and len(p['source_commit'])==40 and all(c in '0123456789abcdef' for c in p['source_commit'])
        and Path(__file__).absolute()==root/ENTRY and not root.is_symlink(),'captured entry/commit differs')
    data={};captured_bytes=0
    for n,h in p['files'].items():
        data[n]=capture(root/n,h,16*1024**2 if n.endswith('.pt') else MAX_CAPTURE)
        captured_bytes+=len(data[n]);need(captured_bytes<=512*1024**2,'total captured input bound exceeded')
    need(all(p['files'].get(n)==h for n,h in HELPER_PINS.items()),'original helper/input bytes required')
    modules={n:B.module('_campaign_'+str(i),root/n,data[n]) for i,n in enumerate((B.SERVICE,B.CPU,B.DIAGNOSTIC,LINEAGE,B.LAUNCH))}
    service,cpu,diagnostic,lineage_api,launch=[modules[n] for n in (B.SERVICE,B.CPU,B.DIAGNOSTIC,LINEAGE,B.LAUNCH)]
    packet_shape(p,service,diagnostic)
    admission_raw=capture(args.admission,args.admission_sha256,1024**2)
    admission_contract(p,decode(admission_raw),args.packet_sha256)
    lineage=accepted_inputs(p,data,lineage_api);service_contracts(root,p,data,service,diagnostic,lineage,launch)
    for name in SOURCE_NAMES:
        for node in ast.walk(ast.parse(data[name])):
            deps=[node.module] if isinstance(node,ast.ImportFrom) and node.module else [a.name for a in node.names] if isinstance(node,ast.Import) else []
            need(all(dep+'.py' in SOURCE_NAMES for dep in deps if dep.startswith(B.PROJECT_PREFIXES)),'captured source closure incomplete')
    return dict(root=root,p=p,data=data,service=service,cpu=cpu,diagnostic=diagnostic,lineage=lineage,
        packet_path=Path(args.packet).absolute(),packet_pin=args.packet_sha256,admission_path=Path(args.admission).absolute(),
        admission_pin=args.admission_sha256,admission_raw=admission_raw)


def recheck(ctx):
    capture(ctx['packet_path'],ctx['packet_pin']);need(capture(ctx['admission_path'],ctx['admission_pin'])==ctx['admission_raw'],'admission changed')
    for n,h in ctx['p']['files'].items():need(capture(ctx['root']/n,h)==ctx['data'][n],'original input changed')


class CapturedSources(B.CapturedSources):
    def __init__(self,captured,root):
        self.root,self.loaded=Path(root),{}
        self.sources={Path(n).stem:raw for n,raw in captured.items() if n!=ENTRY}
        need(not any(n.startswith(B.PROJECT_PREFIXES) for n in sys.modules),'cached project import before capture')


def numeric_prefix(ctx,args,prefix,stage_pin,kind,wid):
    return [str(Path(prefix)/'bin/python'),'-I','-B','-c',ctx['diagnostic'].BOOTSTRAP,
        str(ctx['root']/ENTRY),ctx['p']['files'][ENTRY],'--packet',str(ctx['packet_path']),'--packet-sha256',ctx['packet_pin'],
        '--source-commit',ctx['p']['source_commit'],'--numeric-entry',kind,'--world-id',str(wid),
        '--runtime-stage',str(paths(ctx['p'])['output']/'stage.json'),'--runtime-stage-sha256',stage_pin,
        '--worker-deadline',str(args.worker_deadline),'--']


def policy_specs(p,bindings,prefixes,export,ranker_pin):
    rows={(r['world_id'],r['policy_id']):r for r in bindings}
    need(len(rows)==len(bindings)==96,'exact 96 original numerical bindings required')
    result={}
    for key in p['world_pins']:
        wid=int(key);specs=[]
        for pid in NUMERIC_POLICIES:
            row=rows[wid,pid]
            specs.append(dict(policy_id=pid,kind='task_process',command=prefixes[key]['screen']+[
                '--binding-json',canonical(row['binding']),'--binding-sha256',row['binding_sha256']],
                timeout_seconds=p['numeric_timeout_seconds'],identity_sha256=None,
                policy_provenance_sha256=digest({n:p['files'][n] for n in NUMERIC_SOURCE_NAMES['screen']})))
        specs.append(dict(policy_id='learned_ranker',kind='task_process',command=prefixes[key]['ranker']+[
            '--method','learned_ranker','--seed','109','--draws','32','--checkpoint',export['path'],
            '--checkpoint-sha256',export['sha256'],'--training-receipt-sha256',export['training_receipt_sha256']],
            timeout_seconds=p['numeric_timeout_seconds'],identity_sha256=None,policy_provenance_sha256=ranker_pin))
        specs.extend(p['persistent_specs'][pid] for pid in LLM_POLICIES);result[wid]=specs
    return result


def run_phase(ctx,loc,prefixes,event,*,builder,ranker,campaign,world_type):
    """Trusted orchestration; injected APIs allow fabricated boundary tests only."""
    p,data=ctx['p'],ctx['data'];out=loc['phase'];out.mkdir(exist_ok=False)
    try:
        save(out/'training-lineage.json',ctx['lineage'])
        proposal=decode(data['proposal.json']);plan=decode(data['numerical-plan.json'])
        need(plan['recipe']==ranker.numeric_recipe(artificial=False),'fixed numerical recipe differs')
        binding_args=dict(proposal=proposal,proposal_sha256=PROPOSAL_SHA,expected_source_pins=p['protocol_source_pins'])
        bindings=builder.numeric_policy_bindings(**binding_args)
        need(bindings==decode(data['numeric-bindings.json']) and len(bindings)==96,'original 96 numeric bindings differ')
        worlds={r['world_id']:world_type.from_private_payload(decode(data[f'descriptor-{i:03d}.json']),
            expected_sha256=p['world_pins'][str(r['world_id'])]) for i,r in enumerate(proposal['development_worlds'])}
        save(out/'numeric-training-plan.json',plan)
        event('cpu_ranker_training_start',plan_sha256=p['numerical_plan_sha256'])
        fitted=ranker.fit(plan=plan,plan_sha256=p['numerical_plan_sha256'],output=out/'ranker')
        d=plan['exposure']['counts']['dpo_steps']
        need(fitted['status']=='complete' and (fitted['sft_steps'],fitted['dpo_steps'],fitted['completed_steps'])==(160,d,160+d)
            and fitted['plan_sha256']==p['numerical_plan_sha256'] and fitted['exposure_sha256']==ctx['lineage']['exposure_sha256'],
            'matched ranker did not complete exact accepted schedule')
        final=fitted['exports']['dpo'];need(final['path']=='dpo.ranker.pt' and final['all_training_score_rows_equal'] is True
            and final['training_receipt_sha256']==fitted['training_receipt_sha256'],'final ranker export/reload differs')
        capture(out/'ranker/training_complete.json',fitted['training_receipt_sha256'])
        capture(out/'ranker'/final['path'],final['sha256'])
        export=dict(path=str(out/'ranker'/final['path']),sha256=final['sha256'],training_receipt_sha256=final['training_receipt_sha256'])
        ranker_pin=sha(read(out/'ranker/terminal.json'))
        event('cpu_ranker_training_complete',training_updates=160+d,export_sha256=final['sha256'])
        specs=policy_specs(p,bindings,prefixes,export,ranker_pin)
        bundle=builder.build_scientific_campaign(**binding_args,worlds_by_id=worlds,
            expected_world_pins={int(k):v for k,v in p['world_pins'].items()},policy_specs_by_world=specs,services=p['services'],
            expected_service_identity_pins=p['service_identity_pins'],outer_provenance_receipt_sha256=p['outer_provenance_receipt_sha256'])
        need(bundle['campaign_sha256']==digest(bundle['campaign']) and len(bundle['campaign']['sources'])==24
            and bundle['campaign']['policy_ids']==list(CPU_POLICIES+LLM_POLICIES),'all 192 campaign cells required before responses')
        save(out/'campaign-bundle.json',bundle)
        event('campaign_frozen_before_responses',campaign_sha256=bundle['campaign_sha256'],cells=192)
        binding={k:bundle[k] for k in ('campaign','campaign_sha256','source_plan_pins')};binding['output']=loc['campaign']
        campaign.initialize_campaign(**binding)
        attempts={}
        for pid in CPU_POLICIES:
            event('cpu_policy_start',policy_id=pid)
            result=campaign.run_policy(**binding,policy_id=pid,phase='cpu');attempts[pid]=result['sha256']
            save(out/(pid+'-attempt.json'),result)
            attempt,_=campaign._load_attempt(bundle['campaign'],bundle['campaign_sha256'],campaign._attempt(loc['campaign'],bundle['campaign'],pid),pid,result['sha256'])
            need(not attempt['errors'] and len(attempt['arms'])==24 and all(a['status']=='completed' for a in attempt['arms']),
                'failed/partial CPU policy retained; no next policy or GPU admission')
            event('cpu_policy_complete',policy_id=pid,attempt_sha256=result['sha256'])
        save(out/'attempt-pins.json',attempts)
        terminal=dict(schema=VERSION,status='cpu_phase_complete',campaign_sha256=bundle['campaign_sha256'],attempt_pins=attempts,failed_policies=[],
            numeric_training_updates=160+d,llm_training_updates=0,gpu_services_started=0,completed_cpu_cells=120,
            declared_cells=192,private_scores_computed=False,scientific_qualified=False,retry_allowed=False,
            response_accounting_source='original per-arm charged journals',external_acceptance_required=True)
        save(out/'terminal.json',terminal);return terminal
    except BaseException as exc:
        save(out/'failure.json',failure(exc));raise


def numeric_tail(kind,argv,p,data,wid):
    need(str(wid) in p['world_pins'],'numeric world outside fixed development roster')
    if kind=='screen':
        need(len(argv)==4 and argv[::2]==['--binding-json','--binding-sha256'],'exact screen CLI required')
        binding=decode(argv[1]);rows=decode(data['numeric-bindings.json'])
        found=[r for r in rows if r['world_id']==wid and r['policy_id']==binding.get('arm')]
        need(len(found)==1 and argv[1]==canonical(found[0]['binding']) and argv[3]==found[0]['binding_sha256']
            and digest(binding)==argv[3],'fixed world/arm binding differs')
    else:
        ranker=paths(p)['phase']/'ranker';terminal=decode(read(ranker/'terminal.json'));export=terminal['exports']['dpo']
        d=terminal['dpo_steps']
        need(type(d) is int and 1<=d<=128 and terminal['sft_steps']==160 and terminal['completed_steps']==160+d
            and terminal['status']=='complete' and terminal['plan_sha256']==p['numerical_plan_sha256']
            and terminal['exposure_sha256']==p['expected_training']['exposure_sha256']
            and export['path']=='dpo.ranker.pt' and export['all_training_score_rows_equal'] is True
            and export['training_receipt_sha256']==terminal['training_receipt_sha256'],'fixed completed ranker export required')
        expected=['--method','learned_ranker','--seed','109','--draws','32','--checkpoint',str(ranker/'dpo.ranker.pt'),
            '--checkpoint-sha256',export['sha256'],'--training-receipt-sha256',terminal['training_receipt_sha256']]
        need(argv==expected,'final-only frozen ranker command differs')
        capture(ranker/'training_complete.json',terminal['training_receipt_sha256']);capture(ranker/export['path'],export['sha256'])
    return argv


def numeric_child(args):
    """Authenticate only public binding/source inputs, not private descriptors."""
    started,cpu_started=time.monotonic(),time.process_time();root=Path(args.packet).absolute().parent
    p=decode(capture(args.packet,args.packet_sha256,2*1024**2));resource_contract(p)
    need(set(p)==PACKET_KEYS and p['schema']==VERSION and set(p['files'])==INPUT_NAMES
        and p['source_commit']==args.source_commit and Path(__file__).absolute()==root/ENTRY,'numeric captured packet/entry differs')
    names=NUMERIC_SOURCE_NAMES[args.numeric_entry]|{ENTRY,OLD,'numeric-bindings.json'}
    data={n:capture(root/n,p['files'][n],8*1024**2) for n in names}
    need(p['files'][OLD]==OLD_PIN and args.runtime_stage==str(paths(p)['output']/'stage.json'),'numeric helper/stage differs')
    stage=decode(capture(args.runtime_stage,args.runtime_stage_sha256))
    need(type(args.worker_deadline) is float and math.isfinite(args.worker_deadline)
        and 0<args.worker_deadline-started<=6000,'bounded absolute worker deadline required')
    tail=args.cli[1:] if args.cli[:1]==['--'] else args.cli
    numeric_tail(args.numeric_entry,tail,p,data,args.world_id)
    public=sys.stdin.read(2*1024**2+1);need(len(public.encode())<=2*1024**2,'bounded public envelope required')
    call_id=digest(dict(kind=args.numeric_entry,world_id=args.world_id,argv=tail,input_sha256=sha(public.encode())))
    out=paths(p)['numeric']/call_id;out.mkdir(mode=0o700,exist_ok=False)
    try:
        need(len(list(out.parent.iterdir()))<=2880,'maximum 120x24 numeric decisions exceeded')
        end=min(args.worker_deadline,started+p['numeric_work_seconds'])
        signal.setitimer(signal.ITIMER_REAL,max(.001,end-time.monotonic()))
        save(out/'before.json',B.check_numeric_runtime(p,stage,args.numeric_entry))
        with (out/'input.utf8').open('xb') as f:f.write(public.encode())
        captured={n:data[n] for n in NUMERIC_SOURCE_NAMES[args.numeric_entry]|{ENTRY}}
        save(out/'command.json',dict(entry=args.numeric_entry,world_id=args.world_id,argv=tail,packet_sha256=args.packet_sha256,
            stage_sha256=args.runtime_stage_sha256,source_sha256={n:sha(v) for n,v in captured.items()},worker_deadline=args.worker_deadline))
        loader=CapturedSources(captured,root);sys.meta_path.insert(0,loader)
        old_in,old_argv=sys.stdin,sys.argv;stdout,stderr=io.StringIO(),io.StringIO()
        public_input=io.TextIOWrapper(io.BytesIO(public.encode()),encoding='utf-8',newline='')
        try:
            sys.stdin,sys.argv=public_input,[str(root/ENTRY),*tail]
            name='llm_intervention_screen_policy' if args.numeric_entry=='screen' else 'llm_intervention_policy_cli'
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):result=importlib.import_module(name).main()
            need((type(result) is int and result==0) if args.numeric_entry=='screen' else result is None,'numeric CLI returned failure')
            after=B.check_numeric_runtime(p,stage,args.numeric_entry,loader)
        finally:
            sys.stdin,sys.argv=old_in,old_argv;public_input.close();sys.meta_path.remove(loader)
            for name,value in [('stdout.log',stdout.getvalue()),('stderr.log',stderr.getvalue())]:
                with (out/name).open('x') as f:f.write(value)
        save(out/'after.json',after);need(time.monotonic()<end,'numeric deadline exceeded')
        save(out/'terminal.json',dict(status='complete',call_id=call_id,packet_sha256=args.packet_sha256,
            input_sha256=sha(public.encode()),stdout_sha256=sha(stdout.getvalue().encode()),stderr_sha256=sha(stderr.getvalue().encode()),
            source_sha256={n:sha(v) for n,v in captured.items()},elapsed_seconds=time.monotonic()-started,
            cpu_seconds=time.process_time()-cpu_started,retry_allowed=False))
        sys.stderr.write(stderr.getvalue());sys.stdout.write(stdout.getvalue())
    except BaseException as exc:save(out/'failure.json',failure(exc));raise


def child(args,ctx):
    p,data=ctx['p'],ctx['data'];loc=paths(p)
    need(os.getpgrp()==os.getpid() and args.parent_pid==os.getppid(),'owned CPU worker handoff differs')
    os.environ['ACE_CAMPAIGN_CPU_PGID']=str(os.getpid())
    journal=ctx['service'].Journal(loc['artifacts']);captured={n:data[n] for n in SOURCE_NAMES}
    loader=CapturedSources(captured,ctx['root']);sys.meta_path.insert(0,loader)
    try:
        from llm_intervention_service_runtime import ServiceRuntime
        observer=ServiceRuntime(p,captured,loader,ctx['root']/ENTRY,globals(),args.runtime_stage,args.runtime_stage_sha256,
            journal,ctx['service'].read_pinned,ctx['service'].decode)
        authenticated=dict(packet_sha256=ctx['packet_pin'],admission_sha256=ctx['admission_pin'],files=p['files'],roles=ctx['lineage'])
        journal.save('authenticated-before.json',authenticated);observer.import_cpu_closure()
        import torch
        import llm_intervention_ranker_training as ranker
        import llm_intervention_campaign as campaign
        import llm_intervention_scientific_campaign as builder
        from llm_intervention_world import WorldDescriptor
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True,warn_only=False)
        need(time.monotonic()<args.parent_started+p['deadlines']['startup'],'absolute startup exceeded')
        save(loc['output']/'worker/work-start.json',dict(packet_sha256=ctx['packet_pin'],pid=os.getpid(),monotonic=time.monotonic()))
        signal.setitimer(signal.ITIMER_REAL,max(.001,args.worker_deadline-time.monotonic()))
        loc['numeric'].mkdir(mode=0o700,exist_ok=False)
        prefixes={wid:{kind:numeric_prefix(ctx,args,observer.prefix,args.runtime_stage_sha256,kind,int(wid))
            for kind in ('screen','ranker')} for wid in p['world_pins']}
        journal.save('numeric-prefixes.json',prefixes)
        result=run_phase(ctx,loc,prefixes,journal.event,builder=builder,ranker=ranker,campaign=campaign,world_type=WorldDescriptor)
        observer.observe('after-cpu-phase');recheck(ctx);journal.save('authenticated-after.json',authenticated)
        need(time.monotonic()<args.worker_deadline,'absolute work deadline exceeded')
        journal.save('terminal.json',dict(schema=VERSION,status='cpu_campaign_complete',packet_sha256=ctx['packet_pin'],
            admission_sha256=ctx['admission_pin'],runtime_stage_sha256=args.runtime_stage_sha256,
            phase_terminal_sha256=sha(read(loc['phase']/'terminal.json')),**{k:result[k] for k in (
                'attempt_pins','numeric_training_updates','completed_cpu_cells','declared_cells')},
            llm_training_updates=0,llm_model_or_adapter_loaded=False,tokenizer_calls=0,worlds_generated=0,
            gpu_services_started=0,private_scores_computed=False,scientific_qualified=False,retry_allowed=False,
            external_acceptance_required=True))
    except BaseException as exc:journal.save('failure.json',failure(exc));raise
    finally:sys.meta_path.remove(loader)


def validate_ranker(loc,p,data):
    plan=decode(data['numerical-plan.json']);out=loc['phase'];root=out/'ranker';d=plan['exposure']['counts']['dpo_steps'];total=160+d
    need(decode(read(out/'numeric-training-plan.json'))==plan==decode(read(root/'plan.json')),'original numerical plan changed')
    terminal_raw=read(root/'terminal.json');t=decode(terminal_raw)
    need(not (root/'failure.json').exists() and not (root/'publication-interruption.json').exists()
        and terminal_raw==read(root/'terminal-candidate.json') and t['status']=='complete' and t['scope']=='policy_train'
        and (t['completed_steps'],t['sft_steps'],t['dpo_steps'])==(total,160,d) and t['plan_sha256']==p['numerical_plan_sha256']
        and t['exposure_sha256']==plan['exposure_sha256'] and t['environment_responses']==0
        and t['retry_allowed'] is t['scientific_provenance_qualified'] is False,'complete matched ranker result differs')
    training=decode(capture(root/'training_complete.json',t['training_receipt_sha256']))
    need(training['plan_sha256']==p['numerical_plan_sha256'] and training['exposure_sha256']==plan['exposure_sha256']
        and training['completed_steps']==len(training['steps'])==total,'ranker training receipt differs')
    need({f.name for f in root.glob('step-*.complete.json')}=={f'step-{i:04d}.complete.json' for i in range(total)},'ranker step roster differs')
    for i,row in enumerate(training['steps']):
        need(decode(read(root/f'step-{i:04d}.complete.json'))==row and row['index']==i
            and all(row[k]==v for k,v in plan['exposure']['numeric_schedule'][i].items())
            and row['optimizer_started'] is row['optimizer_returned'] is True
            and math.isfinite(row['loss']) and math.isfinite(row['gradient_norm']),'once-only ranker schedule differs')
        gradients=decode(read(root/f'step-{i:04d}.gradients.json'))['parameters']
        need(gradients and all(g['present'] is True and g['finite'] is True for g in gradients),'ranker gradients missing')
    need(set(t['exports'])=={'sft','dpo'} and set(training['checkpoints'])=={'initial','sft','final'},'both ranker exports/checkpoints required')
    for role,e in t['exports'].items():
        need(e['path']==role+'.ranker.pt' and e['all_training_score_rows_equal'] is True
            and e['training_receipt_sha256']==t['training_receipt_sha256'],'ranker fresh reload differs')
        capture(root/e['path'],e['sha256'])
    for name,h in training['checkpoints'].items():capture(root/(name+'.state.pt'),h)
    ref=decode(read(root/'reference.json'))
    need(ref['completed_sft_steps']==160 and ref['parameter_signature']==training['reference_signature']
        and ref['sft_checkpoint_sha256']==training['checkpoints']['sft'],'fixed matched ranker reference differs')
    capture(root/'reference.pt',ref['state_sha256']);return t


def validate_bundle(ctx,loc,prefixes,ranker):
    p,data=ctx['p'],ctx['data'];bundle=decode(read(loc['phase']/'campaign-bundle.json'));c=bundle['campaign']
    final=ranker['exports']['dpo'];export=dict(path=str(loc['phase']/'ranker'/final['path']),sha256=final['sha256'],
        training_receipt_sha256=final['training_receipt_sha256'])
    expected=policy_specs(p,decode(data['numeric-bindings.json']),prefixes,export,sha(read(loc['phase']/'ranker/terminal.json')))
    proposal=decode(data['proposal.json']);rows=proposal['development_worlds']
    need(bundle['campaign_sha256']==digest(c) and c['scope']==bundle['scope']=='development'
        and len(c['sources'])==24 and c['policy_ids']==list(CPU_POLICIES+LLM_POLICIES)
        and c['policy_phases']=={pid:'cpu' if pid in CPU_POLICIES else 'gpu' for pid in CPU_POLICIES+LLM_POLICIES}
        and c['services']==p['services'] and bundle['proposal_sha256']==PROPOSAL_SHA
        and bundle['numeric_policy_bindings']==decode(data['numeric-bindings.json'])
        and bundle['service_identity_pins']==p['service_identity_pins'] and bundle['source_pins']==p['protocol_source_pins']
        and bundle['world_pins']==[dict(world_id=r['world_id'],sha256=p['world_pins'][str(r['world_id'])]) for r in rows]
        and bundle['worlds_generated']==bundle['responses_generated']==0
        and all(bundle[k] is False for k in ('execution_authorized','scientific_protocol_frozen','scientific_provenance_qualified',
            'source_qualified','runtime_qualified','scheduler_qualified')),'complete scientific campaign declaration differs')
    arms=list(NUMERIC_POLICIES)+['numerical_ranker','llm_frozen','llm_sft','llm_dpo']
    need(bundle['proposal_arm_mapping']==dict(zip(CPU_POLICIES+LLM_POLICIES,arms)),'original arm aliases differ')
    for i,(s,r) in enumerate(zip(c['sources'],rows)):
        plan=s['plan'];public=r['public'];wid=r['world_id']
        binding=lambda value:{k:value[k] for k in ('request_namespace','replicate_id')}
        need(plan['world']==decode(data[f'descriptor-{i:03d}.json']) and plan['world']['world_id']==wid
            and s['plan_sha256']==digest(plan) and plan['policy_specs']==expected[wid]
            and plan['initial_actions']==public['initial_actions'] and plan['acquisition_cap_units']==24
            and plan['initial_cost_units']==public['initial_cost_units'] and plan['world_sha256']==p['world_pins'][str(wid)]
            and plan['task']==public['task'] and plan['learner']==dict(graph=public['graph'],
                noise_variances=public['learner']['noise_variances'],prior_variance=public['learner']['prior_variance'])
            and [e['arm'] for e in r['development_episodes']]==arms
            and plan['initial_binding']==binding(r['initial_binding'])
            and plan['measurement_bindings']=={pid:binding(e['measurement_binding']) for pid,e in zip(CPU_POLICIES+LLM_POLICIES,r['development_episodes'])}
            and s['strata']=={k:r[k] for k in ('regime','graph_stratum','cost_stratum')}
            and plan['objective']==public['objective'] and plan['domain_notes']==public['domain_notes']
            and plan['scientific_provenance_qualified'] is plan['retry_allowed'] is False
            and plan['menu']==public['menu'] and plan['outer_provenance_receipt_sha256']==p['outer_provenance_receipt_sha256'],
            'original development descriptor/menu/commands differ')
    need(bundle['source_plan_pins']==[dict(world_id=s['plan']['world']['world_id'],sha256=s['plan_sha256']) for s in c['sources']],
        'all 24 original source plan pins required')
    return bundle


def validate_attempts(campaign_root, bundle, attempts):
    c = bundle['campaign']; policies = list(CPU_POLICIES)
    need(set(attempts)==set(policies),'exact five attempt pins required')
    need({p.name for p in campaign_root.iterdir()}=={'campaign.json','source-plan-pins.json','start.json'} |
         {f'policy-{i:02d}' for i in range(5)},'GPU attempts/private finalization/retry forbidden in CPU result')
    need(decode(read(campaign_root/'campaign.json'))==c
         and decode(read(campaign_root/'source-plan-pins.json'))==bundle['source_plan_pins'],
         'initial campaign root differs')
    processes = []
    for index,pid in enumerate(policies):
        directory = campaign_root/f'policy-{index:02d}'
        raw = capture(directory/'attempt.json',attempts[pid]); attempt = decode(raw)
        need(raw==read(directory/'attempt.candidate.json') and attempt['policy_id']==pid
             and attempt['campaign_sha256']==bundle['campaign_sha256'] and attempt['phase']=='cpu'
             and attempt['source_plan_pins']==bundle['source_plan_pins'] and attempt['errors']==[]
             and attempt['retry_allowed'] is False and attempt['private_scores_computed'] is False
             and attempt['service']==dict(status='not_started',file=None,sha256=None)
             and not (directory/'service').exists() and len(attempt['arms'])==24
             and inventory(directory,('attempt.json','attempt.candidate.json'))==attempt['evidence_sha256'],
             'CPU attempt identity/completion/custody differs')
        for i,(arm,source) in enumerate(zip(attempt['arms'],c['sources'])):
            need(arm['status']=='completed' and arm['reason'] is None and arm['path']==f'world-{i:02d}'
                 and arm['world_id']==source['plan']['world']['world_id'],'all 24 CPU world arms must complete')
            path = directory/arm['path']
            arm_raw = capture(path/'arm-receipt.json',arm['receipt_sha256']); receipt = decode(arm_raw)
            need(read(path/'arm-receipt.candidate.json')==arm_raw and receipt['status']=='raw_ready'
                 and receipt['plan_sha256']==source['plan_sha256'] and receipt['policy_id']==pid
                 and receipt['new_training_updates']==0 and receipt['private_scores_computed'] is False
                 and receipt['retry_allowed'] is False and receipt['scientific_provenance_qualified'] is False
                 and inventory(path,('arm-receipt.json','arm-receipt.candidate.json'))==receipt['source_evidence_sha256'],
                 'complete arm custody/scope differs')
            diagnostics = decode(read(path/'execution-diagnostics.json'))
            need(diagnostics['episode_returned_cleanly'] is True and diagnostics['adapters_closed_cleanly'] is True,
                 'arm execution failed')
            command = source['plan']['policy_specs'][index]['command']
            for p in sorted(path.rglob('decision-*.process.json')):
                record = decode(read(p))
                need(record.get('returncode')==0 and record['command']==command,'numerical decision failed/argv differs')
                input_path = p.with_name(p.name.replace('.process.json','.input.json'))
                need(digest(decode(read(input_path)))==record['input_sha256'],'public process input pin differs')
                processes.append(record)
    need(120<=len(processes)<=2880,'bounded complete numerical decision inventory required')
    return processes

def validate_numeric_calls(directory, processes, packet, packet_pin, stage, stage_pin, owned_pgid):
    need(directory.is_dir() and len(list(directory.iterdir()))==len(processes),'numeric call inventory differs')
    seen = set()
    for process in processes:
        command = process['command']; split = command.index('--')
        kind = command[command.index('--numeric-entry')+1]; argv = command[split+1:]
        wid = int(command[command.index('--world-id')+1])
        cid = digest(dict(kind=kind,world_id=wid,argv=argv,input_sha256=process['input_sha256']))
        need(cid not in seen,'numerical attempt repeated'); seen.add(cid)
        out = directory/cid
        need({p.name for p in out.iterdir()}=={'before.json','after.json','input.utf8','command.json',
             'stdout.log','stderr.log','terminal.json'},'incomplete numeric attempt')
        r = decode(read(out/'terminal.json')); launched = decode(read(out/'command.json'))
        expected = {n:packet['files'][n] for n in NUMERIC_SOURCE_NAMES[kind]|{ENTRY}}
        need(r['status']=='complete' and r['call_id']==cid and r['packet_sha256']==packet_pin
             and r['input_sha256']==process['input_sha256']==sha(read(out/'input.utf8'))
             and r['stdout_sha256']==sha(process['stdout'].encode())==sha(read(out/'stdout.log'))
             and r['stderr_sha256']==sha(process['stderr'].encode())==sha(read(out/'stderr.log'))
             and r['source_sha256']==expected and 0<=r['elapsed_seconds']<packet['numeric_work_seconds'] and r['cpu_seconds']>=0
             and r['retry_allowed'] is False and launched['entry']==kind and launched['argv']==argv and launched['world_id']==wid
             and launched['source_sha256']==expected and launched['packet_sha256']==packet_pin
             and launched['stage_sha256']==stage_pin,'numeric call/response/captured source differs')
        before,after = decode(read(out/'before.json')),decode(read(out/'after.json'))
        required = {'numpy'} | ({'torch'} if kind=='ranker' else set())
        need(before['versions']=={} and before['used_module_paths']=={} and before['source_sha256']=={}
             and after['versions']=={n:packet['runtime']['imported_versions'][n] for n in required}
             and after['source_sha256'] and set(after['source_sha256'])<=NUMERIC_SOURCE_NAMES[kind]
             and all(expected[n]==p for n,p in after['source_sha256'].items()),'used numeric runtime/source differs')
        for observation in (before,after):
            need(observation['prefix']==stage['staged_prefix'] and observation['python_version']=='3.10.19'
                 and type(observation['pid']) is int and observation['pid']>0
                 and observation['pgid']==owned_pgid and observation['pid']!=owned_pgid
                 and observation['archive_rescanned'] is False and observation['llm_model_or_adapter_loaded'] is False
                 and all(Path(stage['staged_prefix']).resolve() in Path(p).parents
                         for p in [*observation['used_module_paths'].values(),*observation['standard_library_paths'].values()]),
                 'numeric stage observation differs')
        need(before['pid']==after['pid'],'numeric process identity changed')


def inventory(root,exclude=()):
    root=Path(root);need(root.is_dir() and not root.is_symlink(),'ordinary evidence root required')
    result={};total=0
    for path in sorted(root.rglob('*')):
        relative=path.relative_to(root).as_posix()
        if any(relative==n or relative.startswith(n+'/') for n in exclude):continue
        need(not path.is_symlink(),'evidence symlink forbidden')
        if path.is_dir():continue
        raw=read(path);total+=len(raw);need(total<=8*1024**3 and len(result)<262144,'bounded complete custody exceeded')
        result[relative]=sha(raw)
    return result


def validate_worker(ctx,args,stage,stage_pin,provenance):
    p,data=ctx['p'],ctx['data'];loc=paths(p);out=loc['artifacts']
    def get(n):return decode(read(out/n))
    need(not list(out.rglob('failure.json')) and not list(out.rglob('publication-interruption.json')),'retained failure prevents success')
    expected=dict(packet_sha256=ctx['packet_pin'],admission_sha256=ctx['admission_pin'],files=p['files'],roles=ctx['lineage'])
    need(get('authenticated-before.json')==get('authenticated-after.json')==expected,'before/after complete input binding differs')
    checks=B.module('_scientific_runtime_checks',ctx['root']/'llm_intervention_service_runtime.py',data['llm_intervention_service_runtime.py'])
    ctx['cpu'].validate_runtime_records(lambda n:get('runtime-after-cpu-phase.json' if n=='runtime-after-cpu-tokenization.json' else n),
        stage,stage_pin,ctx['service'],ctx['diagnostic'],checks,provenance)
    need(get('runtime-stage-binding.json')['gpu_mapping_enabled'] is False,'CPU mapping exemption forbidden')
    projects_expected={Path(n).stem:p['files'][n] for n in SOURCE_NAMES if n!=ENTRY}
    for phase in ('after-service-imports','after-cpu-phase'):
        projects=get('runtime-'+phase+'.json')['captured_project_modules']
        need(set(projects_expected)<=set(projects)<=set(projects_expected)|{'__main__','__mp_main__'}
            and all(projects[n]['sha256']==h for n,h in projects_expected.items()),'actual captured runtime source closure differs')
    need(decode(read(loc['phase']/'training-lineage.json'))==ctx['lineage'],'saved training roles differ')
    prefixes=get('numeric-prefixes.json')
    expected_prefixes={wid:{kind:numeric_prefix(ctx,args,stage['staged_prefix'],stage_pin,kind,int(wid)) for kind in ('screen','ranker')}
        for wid in p['world_pins']}
    need(prefixes==expected_prefixes,'fixed numeric prefixes/deadline differ')
    ranker=validate_ranker(loc,p,data);bundle=validate_bundle(ctx,loc,prefixes,ranker)
    attempts=decode(read(loc['phase']/'attempt-pins.json'));processes=validate_attempts(loc['campaign'],bundle,attempts)
    started=decode(read(loc['output']/'worker.started.json'));validate_numeric_calls(loc['numeric'],processes,p,ctx['packet_pin'],stage,stage_pin,started['pgid'])
    phase=decode(read(loc['phase']/'terminal.json'));terminal=get('terminal.json');d=ranker['dpo_steps']
    need(phase['schema']==VERSION and phase['status']=='cpu_phase_complete' and phase['attempt_pins']==attempts
        and phase['failed_policies']==[]
        and phase['campaign_sha256']==bundle['campaign_sha256'] and phase['numeric_training_updates']==160+d
        and phase['completed_cpu_cells']==120 and phase['declared_cells']==192 and phase['external_acceptance_required'] is True
        and phase['llm_training_updates']==phase['gpu_services_started']==0
        and phase['private_scores_computed'] is phase['scientific_qualified'] is phase['retry_allowed'] is False,'CPU phase result differs')
    need(terminal==dict(schema=VERSION,status='cpu_campaign_complete',packet_sha256=ctx['packet_pin'],admission_sha256=ctx['admission_pin'],
        runtime_stage_sha256=stage_pin,phase_terminal_sha256=sha(read(loc['phase']/'terminal.json')),attempt_pins=attempts,
        numeric_training_updates=160+d,completed_cpu_cells=120,declared_cells=192,llm_training_updates=0,
        llm_model_or_adapter_loaded=False,tokenizer_calls=0,worlds_generated=0,gpu_services_started=0,private_scores_computed=False,
        scientific_qualified=False,retry_allowed=False,external_acceptance_required=True),'worker terminal differs')
    events=[get(x.name) for x in sorted(out.glob('phase-*.json'))];names=[e['phase'] for e in events]
    order=('service_cpu_import_start','service_cpu_import_complete','cpu_ranker_training_start','cpu_ranker_training_complete','campaign_frozen_before_responses')
    need(all(names.count(n)==1 for n in order) and [names.index(n) for n in order]==sorted(names.index(n) for n in order)
        and [e['policy_id'] for e in events if e['phase']=='cpu_policy_start']==list(CPU_POLICIES)
        and [e['policy_id'] for e in events if e['phase']=='cpu_policy_complete']==list(CPU_POLICIES)
        and names.index('cpu_policy_start')>names.index(order[-1]),'once-only response phase order differs')
    return dict(terminal_sha256=sha(read(out/'terminal.json')),campaign_sha256=bundle['campaign_sha256'],source_plan_pins=bundle['source_plan_pins'],
        attempt_pins=attempts,roles=ctx['lineage']['roles'],numeric_training_updates=160+d,completed_cpu_cells=120,
        numeric_decisions=len(processes),gpu_cells_executed=0,private_scores_computed=False,scientific_qualified=False)


def allocation(raw,job,p):
    need(type(job) is str and re.fullmatch('[1-9][0-9]*',job) and not os.environ.get('SLURM_JOB_GPUS')
        and not os.environ.get('SLURM_STEP_GPUS'),'CPU Slurm allocation required')
    f=dict(re.findall(r'([^\s=]+)=(\S+)',raw));r=p['resources']
    expected=dict(JobId=job,JobState='RUNNING',Account=r['account'],Partition=r['partition'],QOS=r['qos'],
        NumCPUs='2',NumTasks='1',NumNodes='1',Requeue='0',Restarts='0')
    need(all(f.get(k)==v for k,v in expected.items()) and f.get('UserId','').startswith('paco0228('),'actual CPU job identity differs')
    for key in ('ReqTRES','AllocTRES'):
        rows=[x.split('=',1) for x in f.get(key,'').split(',')];need(all(len(x)==2 for x in rows),'malformed CPU TRES')
        tres=dict(rows);need(len(tres)==len(rows) and tres.get('cpu')=='2' and tres.get('node')=='1'
            and tres.get('mem') in ('7680M','7.5G','7.50G') and not any(k.startswith('gres/') for k in tres),'CPU/memory/no-accelerator allocation differs')
    m=re.fullmatch(r'(?:(\d+)-)?(\d+):(\d+):(\d+)',f.get('TimeLimit',''))
    need(m is not None and int(m[1] or 0)*86400+int(m[2])*3600+int(m[3])*60+int(m[4])==r['wall_seconds'],'actual caller wall time differs')
    return f


def parent(args,ctx):
    p,d=ctx['p'],ctx['diagnostic'];out=paths(p)['output'];args.worker_deadline=args.started+p['deadlines']['work']
    ownership=B.module('_scientific_cpu_owned',ctx['root']/OWNED,ctx['data'][OWNED])
    args.owned_child=ownership.OwnedChild(args.started+p['deadlines']['hard'])
    job=os.environ.get('SLURM_JOB_ID','');q=subprocess.run(['scontrol','show','job','-o',job],capture_output=True,text=True,timeout=10)
    save(out/'allocation.json',dict(stdout=q.stdout,stderr=q.stderr,returncode=q.returncode))
    need(q.returncode==0,'allocation query failed');allocation(q.stdout,job,p)
    scratch=Path(os.environ.get('SLURM_SCRATCH',''));need(scratch.is_absolute(),'absolute node scratch required')
    scratch=scratch.resolve(strict=True);need(scratch.is_dir() and scratch.stat().st_uid==os.getuid(),'owned scratch required')
    local=scratch/('ace-scientific-campaign-cpu-'+job)
    B.separate([local,out,ctx['root'],p['runtime_source'],p['archive']['path'],p['manifest']['path']]);local.mkdir(mode=0o700,exist_ok=False)
    archive=B.module('_campaign_archive',ctx['root']/'llm_intervention_runtime_archive.py',ctx['data']['llm_intervention_runtime_archive.py'])
    probe=B.module('_campaign_cleanup',ctx['root']/'probe.py',ctx['data']['probe.py'])
    training=B.module('_campaign_environment',ctx['root']/B.TRAINING,ctx['data'][B.TRAINING])
    event=d.events(out,args.started,args.process_started,512*1024**2);event('stage_start')
    stage=d.stage_runtime(p,archive,local/'runtime',event);stage_pin=save(out/'stage.json',stage);event('stage_complete')
    prefix=Path(stage['staged_prefix']);provenance=d.provenance_contract(prefix);save(out/'independent-provenance-sources.json',provenance)
    command=[str(prefix/'bin/python'),'-I','-B','-c',d.BOOTSTRAP,str(ctx['root']/ENTRY),p['files'][ENTRY],
        '--packet',str(ctx['packet_path']),'--packet-sha256',ctx['packet_pin'],'--source-commit',p['source_commit'],
        '--admission',str(ctx['admission_path']),'--admission-sha256',ctx['admission_pin'],'--worker',
        '--runtime-stage',str(out/'stage.json'),'--runtime-stage-sha256',stage_pin,'--worker-deadline',str(args.worker_deadline),
        '--parent-started',str(args.started),'--parent-pid',str(os.getpid())]
    session=ownership.supervise(command,training.cpu_environment(prefix,os.environ),out,started=args.started,
        startup_deadline=args.started+p['deadlines']['startup'],work_deadline=args.worker_deadline,
        packet_sha256=ctx['packet_pin'],owned=args.owned_child,kill_owned=probe.kill_owned)
    report=validate_worker(ctx,args,stage,stage_pin,provenance);recheck(ctx)
    need(time.monotonic()<args.started+p['deadlines']['hard'],'absolute hard deadline exhausted')
    # Complete output custody includes raw failures if any; success only after validation/owned cleanup.
    tree=inventory(out,('custody.json','supervisor.json'));tree_pin=save(out/'custody.json',tree)
    save(out/'supervisor.json',dict(schema=VERSION,status='cpu_campaign_complete',job_id=job,packet_sha256=ctx['packet_pin'],
        admission_sha256=ctx['admission_pin'],stage_sha256=stage_pin,session=session,validation=report,custody_sha256=tree_pin,
        elapsed_seconds=time.monotonic()-args.started,scientific_qualified=False,retry_allowed=False,independent_acceptance_required=True))


def finish_owned(owned,out):
    # Cancellation is permitted only after independently observed retirement.
    if owned is not None:
        owned.hold_until_closed()
        if out is not None and owned.errors:
            try:save(out/'worker.lifecycle-errors.json',dict(errors=owned.errors,owned_process_retired=owned.retired))
            except BaseException as exc:
                owned.error('lifecycle_publication',exc)
                try:sys.stderr.write('Retained lifecycle errors: '+canonical(owned.errors)+'\n')
                except BaseException:pass
    signal.setitimer(signal.ITIMER_REAL,0)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','packet-sha256','source-commit'):ap.add_argument('--'+name,required=True)
    for name in ('admission','admission-sha256','runtime-stage','runtime-stage-sha256'):ap.add_argument('--'+name)
    ap.add_argument('--worker',action='store_true');ap.add_argument('--numeric-entry',choices=('screen','ranker'))
    ap.add_argument('--world-id',type=int);ap.add_argument('--worker-deadline',type=float)
    ap.add_argument('--parent-started',type=float);ap.add_argument('--parent-pid',type=int);ap.add_argument('cli',nargs=argparse.REMAINDER)
    args=ap.parse_args();args.started,args.process_started=time.monotonic(),time.process_time();args.owned_child=None
    def interrupted(sig,frame):raise B.CampaignInterrupted('scientific campaign signal '+str(sig))
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):signal.signal(sig,interrupted)
    signal.setitimer(signal.ITIMER_REAL,120)
    out=None
    try:
        if args.numeric_entry:
            need(not args.worker and args.world_id is not None,'one exact numeric mode required');numeric_child(args);return
        need(args.admission and pin(args.admission_sha256),'independent stage admission required')
        p=decode(capture(args.packet,args.packet_sha256,2*1024**2));resource_contract(p)
        need(B.absolute(p['cpu_output']) and Path(p['cpu_output'])==Path(p['cpu_output']).resolve(),'resolved output required')
        B.separate([p['cpu_output'],Path(args.packet).absolute().parent,p['runtime_source'],p['model_directory'],p['archive']['path'],p['manifest']['path']])
        if args.worker:
            need(type(args.parent_started) is float and math.isfinite(args.parent_started)
                and 0<=args.started-args.parent_started<p['deadlines']['startup'] and args.parent_pid==os.getppid()
                and args.worker_deadline==args.parent_started+p['deadlines']['work'],'owned original deadline handoff required')
            args.started=args.parent_started;claimed=paths(p)['output']/'worker'
        else:claimed=paths(p)['output']
        signal.setitimer(signal.ITIMER_REAL,max(.001,args.started+p['deadlines']['startup']-time.monotonic()))
        claimed.mkdir(mode=0o700,exist_ok=False);out=claimed
        save(out/'start.json',dict(schema=VERSION,packet_sha256=args.packet_sha256,resources=p['resources'],deadlines=p['deadlines'],retry_allowed=False))
        ctx=authenticate(args)
        if args.worker:
            need(args.runtime_stage==str(paths(p)['output']/'stage.json') and pin(args.runtime_stage_sha256),'original runtime stage pin required');child(args,ctx)
        else:parent(args,ctx)
    except BaseException as exc:
        if out is not None:
            name='publication-interruption.json' if (out/'supervisor.json').exists() else 'failure.json'
            try:
                if not os.path.lexists(out/name):
                    row=failure(exc)
                    if args.owned_child is not None:row['lifecycle_errors']=list(args.owned_child.errors)
                    save(out/name,row)
            except BaseException as publication:
                if args.owned_child is not None:args.owned_child.error('failure_publication',publication)
                try:sys.stderr.write('Failure receipt publication: '+repr(publication)+'; original: '+repr(exc)+'\n')
                except BaseException:pass
        raise
    finally:
        finish_owned(args.owned_child,out)


if __name__=='__main__':main()
