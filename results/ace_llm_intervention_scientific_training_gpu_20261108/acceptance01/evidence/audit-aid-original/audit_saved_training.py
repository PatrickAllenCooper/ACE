"""UNEXECUTED saved-artifact aids for35420219, not an autonomous acceptance issuer.

No CLI or writes. Caller must first independently establish complete terminal
custody and all runtime/scheduler/work/cleanup findings using audit-map.md.
Only captured stdlib validators are loaded; no run_training or tensor loader.
"""
from pathlib import Path
import contextlib
import hashlib
import json
import re
import sys
import types

JOB='35420219'
PACKET_COMMIT='9c54a77b3ee0ba3988a2dcdd0fe733ee0562ffe7'
RESULT_SHA='916ae493932ec509b96929ca8da06e1cde0757e0b383baf6b5252d76aa17ef32'
SOURCE_COMMIT='a9346022f145cc29439325b6f291597b2e6ec037'
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01'
TOKEN_ACCEPTANCE_SHA='f6fcac291d2f4031788db62c71cc7d87e26cc4fe427622aab7a8f6dcd3c8fb1e'
TOKEN_BASELINE_SHA='0443c128767022579041b4e303f7ff7a3616859026fb2f8772ebdcb8e16f4054'
PINS={
 'llm_intervention_screen_training_gpu_worker':'ecf9d7099d2a038fba01f2eddf93d93680fe675396a7d21d248ad81f2892c53d',
 'llm_intervention_scientific_training_gpu_worker':'ab2c999a35b247e03df87dce9f19d771c50f84747381bfbc0ed980e9c832a878',
 'llm_intervention_scientific_training_lineage':'f3b328203d5758418d0afd677513822e8d7f134acca10545aab2d3fda1d3fc35',
 'cpu_consumer':'3893a30ec2e652dac78529602564a4a295ddf9258461530ec7c74557df809025'}
GATES=('training_gpu_qualified','checkpoint_reload_inference_qualified','complete_custody_verified',
       'runtime_qualified','owned_process_cleanup_verified','scheduler_verified')


def need(ok,why):
    if not ok:raise ValueError(why)


def sha(value):return hashlib.sha256(value).hexdigest()


def encode(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def decode(value):
    def unique(items):
        result={}
        for key,v in items:
            need(key not in result,'duplicate saved key');result[key]=v
        return result
    result=json.loads(value,object_pairs_hook=unique)
    json.dumps(result,allow_nan=False)
    return result


def read_pinned(path,pin,maximum=256*1024**2):
    path=Path(path)
    need(re.fullmatch('[0-9a-f]{64}',pin or '') is not None,'independent bytepin required')
    need(path.is_file() and not path.is_symlink() and path.stat().st_nlink==1,'independent regular captured file required')
    with path.open('rb') as stream:value=stream.read(maximum+1)
    need(len(value)<=maximum and sha(value)==pin,'captured size/pin differs')
    return value


@contextlib.contextmanager
def closed_validators():
    """Explicit future call only. No existing project-module cache is admitted."""
    need(not any(name in sys.modules for name in PINS),'captured validator names already loaded')
    modules={};created=[]
    try:
        for name,pin in PINS.items():
            path=Path(__file__).resolve().parent/'captured'/(name+'.py')
            source=read_pinned(path,pin,8*1024**2)
            mod=types.ModuleType(name);mod.__file__=str(path)
            sys.modules[name]=mod;created.append(name)
            exec(compile(source,str(path),'exec'),vars(mod));modules[name]=mod
        yield modules
    finally:
        for name in reversed(created):sys.modules.pop(name,None)


def terminal_identity(*,outer_raw,outer_sha256,result_raw,disposition_raw,disposition_sha256,inventory_raw,inventory_sha256):
    """Necessary identity gate only; not a substitute for independent full custody."""
    need(sha(outer_raw)==outer_sha256 and sha(result_raw)==RESULT_SHA
         and sha(disposition_raw)==disposition_sha256 and sha(inventory_raw)==inventory_sha256,'independent terminal metadata pins differ')
    outer,result,disp,t=map(decode,(outer_raw,result_raw,disposition_raw,inventory_raw))
    need(result['job_id']==disp['job_id']==JOB and result['returncode']==0 and result['ambiguous'] is False
         and outer['remote_root']==disp['remote_root']==ROOT and outer['source_commit']==SOURCE_COMMIT
         and disp['packet_sha256']==outer_sha256 and disp['submission_result_sha256']==RESULT_SHA,
         'actual original job/source/packet/result mismatch')
    need(disp['custody_bytes_verified'] is disp['current_lineage_verified'] is True
         and disp['errors']==disp['input_pin_mismatches']==[] and disp['exact_prior_accounting_matched'] is True
         and disp['exact_prior_and_current_journals_matched'] is True
         and disp['observed_related_jobs']==45 and disp['observed_submission_journals']==92,
         'complete original45/92 custody prerequisite missing')
    need(t['returncode']==0,'original inventory transport failed');obs=decode(t['stdout'])
    need(obs['terminal_gate'] is obs['root_present'] is True and not obs.get('observation_error')
         and obs['parent_accounting'][0]==JOB and len(obs['files'])==disp['raw_files']
         and sum(v['bytes'] for v in obs['files'].values())==disp['raw_bytes'],'terminal inventory incomplete')
    for n,h in {**outer['new_input_files'],'packet.json':outer_sha256,'submission.result.json':RESULT_SHA,
                'submission.intent.json':result['intent_sha256']}.items():
        need(obs['files']['launch/'+n]['sha256']==h,'original input/journal pin mismatch')
    return outer,result,disp,obs


def expected_training(*,cpu,outer,token_acceptance_raw,freeze_raw,freeze_sha256,prepared_sha256):
    """cpu is the captured3893 module. Values come from prior authority, not new acceptance."""
    need(sha(token_acceptance_raw)==TOKEN_ACCEPTANCE_SHA and sha(freeze_raw)==freeze_sha256,'prior independent authority pin differs')
    token=decode(token_acceptance_raw);freeze=decode(freeze_raw)
    need(token['schema']=='ace-scientific-training-preparation-acceptance-v1' and token['status']=='accepted'
         and token['scope']=='policy_train' and token['required_findings']==0
         and all(token[k] is True for k in ('complete_custody_verified','runtime_qualified_for_this_cpu_execution','owned_process_cleanup_verified'))
         and token['training_execution_qualified'] is token['scientific_qualified'] is False,'actual strict TOKEN02 acceptance required')
    need(outer['baseline_sha256']==TOKEN_BASELINE_SHA and outer['cpu_acceptance_sha256']==TOKEN_ACCEPTANCE_SHA
         and outer['cpu_job_id']==token['job_id'] and token['prepared_sha256']==outer['prepared_sha256']==prepared_sha256
         and outer['plan_sha256']==token['plan_sha256'] and token['freeze_sha256']==freeze_sha256
         and outer['new_input_files']['freeze.json']==freeze_sha256,'original token/training/freeze joins differ')
    return dict(job_id=JOB,source_commit=SOURCE_COMMIT,prepared_sha256=prepared_sha256,
        training_plan_sha256=token['plan_sha256'],exposure_sha256=token['exposure_sha256'],
        scientific_freeze_sha256=freeze_sha256,collection_acceptance_sha256=outer['new_input_files']['collection-acceptance.json'],
        cpu_preparation_acceptance_sha256=TOKEN_ACCEPTANCE_SHA,model_files_sha256=cpu.digest(freeze['model_files']),
        runtime_contract_sha256=cpu.digest(freeze['runtime']))


def validate_fit(*,worker,fit_path,prepared_raw,expected):
    """Saved metadata/opaque file hashing only; do not print plan, scores or sequences."""
    need(sha(prepared_raw)==expected['prepared_sha256'],'original prepared file pin mismatch')
    return worker.validate_outputs(Path(fit_path),decode(prepared_raw),prepared_sha256=expected['prepared_sha256'],
                                   plan_sha256=expected['training_plan_sha256'])


def assemble_after_independent_audit(*,lineage,expected,prepared_raw,fit_path,raw_file_pins,checked,findings):
    """In-memory acceptance ONLY after reviewer supplies completed actual findings.

    findings is a review result, not a generated/default gate. Each required gate
    has passed=True, nonempty rationale and exact evidence-file SHA256 pins.
    This function cannot establish those findings; audit-map.md remains mandatory.
    """
    need(set(findings)==set(GATES),'all independent actual review domains required')
    for gate,item in findings.items():
        need(item['passed'] is True and type(item['rationale']) is str and item['rationale'].strip()
             and item['evidence_pins'] and all(raw_file_pins.get(n)==h for n,h in item['evidence_pins'].items()),
             'unresolved/unbound independent audit gate: '+gate)
    need(checked['status']=='validated_worker_evidence' and checked['prepared_sha256']==expected['prepared_sha256']
         and checked['training_plan_sha256']==expected['training_plan_sha256']
         and checked['runtime_and_cleanup_qualified'] is False and checked['tensor_files_deserialized'] is False,
         'closed saved-evidence validation required; it does not qualify runtime')
    fit=Path(fit_path);prefix='launch/training/session/worker/artifacts/fit/'
    def get(name):return read_pinned(fit/name,raw_file_pins[prefix+name])
    terminal_raw=get('training/terminal.json');terminal=decode(terminal_raw)
    acceptance=dict(schema='ace-scientific-training-gpu-acceptance-v1',status='scientific_training_gpu_accepted',scope='policy_train',
        required_findings=0,scientific_utility_qualified=False,**expected,**{gate:True for gate in GATES},
        completed_steps=terminal['completed_steps'],sft_steps=terminal['sft_steps'],dpo_steps=terminal['dpo_steps'],environment_responses=0,
        engine_terminal_sha256=sha(terminal_raw),independent_actual_findings=findings)
    parts={}
    for stage in ('sft','final'):
        for kind,suffix in (('checkpoint','checkpoint.json'),('adapter','adapter.pt')):
            value=get('training/'+stage+'.'+suffix);acceptance[stage+'_'+kind+'_sha256']=sha(value);parts[stage+'_'+kind+'_raw']=value
    role_check=lineage.admit_training(acceptance_raw=encode(acceptance),acceptance_sha256=sha(encode(acceptance)),expected=expected,
        prepared_raw=prepared_raw,engine_terminal_raw=terminal_raw,**parts)
    return acceptance,role_check


def accounting_after_acceptance(*,cpu,prior_raw,result,disp,obs,acceptance_raw,disposition_sha256):
    need(sha(prior_raw)==TOKEN_BASELINE_SHA,'independent accepted44/90 baseline differs')
    prior=decode(prior_raw);a=decode(acceptance_raw);rows=obs['related_accounting']
    need(a['job_id']==JOB and a['status']=='scientific_training_gpu_accepted','actual acceptance required before accounting join')
    before=[r for r in rows if r[0]!=JOB];current=[r for r in rows if r[0]==JOB]
    need(prior['terminal_jobs']==44 and prior['submission_journals']==90 and len(before)==44 and len(current)==1
         and cpu.normalized(before)==cpu.normalized(prior['related_accounting'])
         and cpu.normalized(current)==cpu.normalized([obs['parent_accounting']]),'all original rows must remain')
    row=cpu.normalized(current)[0];allocation=disp['allocation']
    need(row[2:4]==['COMPLETED','0:0'] and row[4]==4 and 0<=row[5]<=2700
         and allocation['resource_contract_matched'] is True and allocation['errors']==[] and allocation['allocated_cpus']==4
         and allocation['elapsed_seconds']==row[5] and allocation['gpu_count']==1 and allocation['gpu_slice_seconds']==row[5]
         and allocation['allocated_cpu_seconds']==4*row[5],'accepted exact allocation required')
    jp={**prior['journal_pins'],ROOT+'/submission.intent.json':result['intent_sha256'],ROOT+'/submission.result.json':RESULT_SHA}
    need(len(jp)==92 and jp==obs['journal_pins'],'all original/current journals required')
    pc,pg=cpu.gpu_cost(prior['related_accounting']);nc,ng=cpu.gpu_cost(rows)
    need((pc,pg)==(prior['closed_cpu_seconds'],prior['closed_gpu_slice_seconds']) and (nc,ng)==(pc+4*row[5],pg+row[5])
         and (nc,ng)==(disp['closed_cpu_seconds'],disp['closed_gpu_slice_seconds']),'actual charges differ')
    return dict(schema=prior['schema'],terminal_jobs=45,submission_journals=92,related_accounting=rows,journal_pins=jp,
        closed_cpu_seconds=nc,closed_gpu_slice_seconds=ng,preparation_allowance_cpu_seconds=3600,cpu_admission_seconds=nc+3600,
        outstanding_reservation=0,cpu_cap_seconds=57600,gpu_cap_seconds=7200,
        input_pins=dict(predecessor=TOKEN_BASELINE_SHA,acceptance=sha(acceptance_raw),disposition=disposition_sha256))
