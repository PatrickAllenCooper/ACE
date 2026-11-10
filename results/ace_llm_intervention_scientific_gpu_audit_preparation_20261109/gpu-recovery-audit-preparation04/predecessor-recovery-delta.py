"""Pure saved-metadata recovery joins. No filesystem, producer, model or remote API.

The caller supplies already authenticated bytes through its bounded read interface.
This module never grants acceptance. GPU01's nonzero collector remains nonzero.
"""
import hashlib
import json
from pathlib import PurePosixPath

CONTROL = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261109_02'
OLD = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108_01'
OUTPUT = OLD+'/gpu'
CAMPAIGN = 'cpu/cpu/worker/artifacts/campaign/'
ENTRY = 'qualify_llm_intervention_scientific_campaign_gpu.py'
CHECKER = 'llm_intervention_cpu_preflight_receipt.py'
HISTORY = 'llm_intervention_cpu_history_admission.py'
ORIGINAL_ENTRY = '2694093512f8ce144bce7f33d6e2306bc251011b26567642c32006a7902e7db1'
OPERATIONAL = {
    ENTRY:'696be95ff571d584f4600b5a87fe6a442eaa2da863d07c0d464ac22d4e1de1fa',
    CHECKER:'b9d0bf56e6c3174ad4ca2c21a43e3c13372c4755575a945437137dc2a1e942c8',
    HISTORY:'674d114f496c63cd9891f6c2d47b0d74a9a657090c0751215f91e2580cc941c9',
}
FAILED = {
    'failed_acceptance':'cacfe32add5d056c0870e58f4aac69ef2afcf2265171d9320bf96c2c2980bca0',
    'baseline':'cf0982dc0ac3f482c32665e710f06b98eef679d2a4b74ecdf989a7acaa3360ef',
    'failed_disposition':'545cee3c7ae4ce2eafa5f1e6fd7fd3f3566a9415e6b89d1a42a27d6640e94507',
    'failed_seal':'8e0ce58a1e093eee1e94d89025e29ac09f97f8b1c4fc3f29751a782401f5a7c0',
    'failed_before':'f9c7dc2ecf1e0c91990e2c7d787abea37187fa520675ad580a1a7f5359f8d9d3',
    'failed_after':'4303d7d4ef775ac133dde12b3d2ac875b7c31ed2382b45abd0c6cc6a55a188ac',
    'failed_remote_custody':'5224bf51110996d0a72f334dc976db7644f24d99e7500b99730ddf00eff19214',
}
RECOVERY_FILES = {'authority':'gpu-recovery-authority.json','context':'cpu-history-expected.json',
                  'receipt':'cpu-history-preflight.json','read_set':'cpu-history-readset.json'}
CONTROL_ROLES = {
    'recovery_authority':'inputs/gpu-recovery-authority.json',
    'recovery_context':'inputs/cpu-history-expected.json',
    'preflight_receipt':'inputs/cpu-history-preflight.json',
    'readset':'inputs/cpu-history-readset.json',
    'operational_manifest':'operational-source-manifest.json',
    'failed_acceptance':'gpu-failure-acceptance.json',
    'failed_disposition':'gpu-failure-disposition.json',
    'failed_seal':'gpu-failure-seal.json',
    'failed_before':'gpu-failure-inventory.json',
    'failed_after':'gpu-failure-inventory-after.json',
    'recovery_review':'gpu-recovery-review.md',
    'operational_review':'operational-source-review.md',
    'preflight_review':'preflight-review.md',
}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value):
    return sha(canonical(value))


def same(a,b):
    return canonical(a)==canonical(b)


def validate_sources(packet, closure, authority, context, manifest, freeze_pins):
    """Exactly three operational exceptions; all scientific sources stay frozen."""
    old=closure['input_source_files']
    need(ENTRY in old and old[ENTRY]==ORIGINAL_ENTRY and not {CHECKER,HISTORY}&set(old),
         'original frozen entry/closure differs')
    declared=closure['source_names']
    need(type(declared) is list and all(type(n) is str for n in declared)
         and declared==sorted(set(declared)) and ENTRY in declared and set(declared)<=set(old),
         'unique exact frozen declared source roster required')
    outside={n:h for n,h in old.items() if n not in declared}
    need(closure['helpers_outside_source_names']==outside
         and closure['entrypoint']==ENTRY and closure['entrypoint_sha256']==ORIGINAL_ENTRY
         and closure['loaded_work_source_roster']==sorted(set(declared)-{ENTRY})
         and closure['loaded_work_source_pins']=={n:old[n] for n in declared if n!=ENTRY},
         'frozen captured helpers and declared/loaded source relationships differ')
    need(packet['source_names']==sorted(set(declared)|{CHECKER,HISTORY}),
         'exact original declared roster plus two operational additions required')
    need({n for n in packet['files'] if n.endswith('.py')}==set(old)|{CHECKER,HISTORY},
         'exact original Python capture roster plus two operational additions required')
    for name,pin in old.items():
        origins=[h for n,h in freeze_pins.items() if PurePosixPath(n).name==name]
        need(origins==[pin], 'frozen scientific source basename/pin differs')
        if name!=ENTRY:
            need(packet['files'][name]==pin,'scientific source changed: '+name)
    need(all(packet['files'][n]==h for n,h in OPERATIONAL.items()),'unreviewed operational entry/helper')
    sources={n:packet['files'][n] for n in packet['source_names']}
    need(same(sources,authority['operational_source_files']) and digest(sources)==context['source_files_sha256'],
         'explicit authority/source context differs')
    need(set(manifest)=={'schema','source_commit','files'} and manifest['schema']=='ace-gpu02-operational-sources-v1'
         and set(manifest['files'])==set(OPERATIONAL), 'exact independently committed operational manifest required')
    need(type(manifest['source_commit']) is str and len(manifest['source_commit'])==40
         and all(c in '0123456789abcdef' for c in manifest['source_commit']),'full operational Git identity required')
    for name,row in manifest['files'].items():
        path=PurePosixPath(row['repository_path'])
        need(set(row)=={'repository_path','sha256'} and row['sha256']==OPERATIONAL[name]
             and not path.is_absolute() and str(path)==row['repository_path'] and '..' not in path.parts,
             'operational source manifest path/pin differs')
    return sources


def inventory_extension(previous,current):
    """Preserve complete failed47 roots; append only control, old/gpu and GPU policies."""
    need(set(current['roots'])==set(previous['roots'])|{'control'} and 'control' not in previous['roots']
         and current['roots']['control']==CONTROL and previous['roots']['launch']==OLD,
         'explicit disjoint control/new versus launch/failed roots required')
    for label,path in previous['roots'].items():
        need(current['roots'][label]==path,'preserved ancestor physical root changed')
        p=PurePosixPath(path);c=PurePosixPath(CONTROL)
        need(c!=p and c not in p.parents and p not in c.parents,'control overlaps ancestor root')
    old=previous['full_inventory'];now=current['full_inventory']
    need(all(n!='launch/gpu' and not n.startswith('launch/gpu/')
             for n in [*old['files'],*old['directories']]),'failed GPU01 output was not absent')
    need(all(now['files'].get(n)==v for n,v in old['files'].items()),'failed47 original bytes changed')
    need(set(old['directories'])<=set(now['directories']),'failed47 original directory missing')
    prefixes=('control','launch/gpu',*(CAMPAIGN+f'policy-{i:02d}' for i in (5,6,7)))
    def allowed(name):
        return any(name==p or name.startswith(p+'/') for p in prefixes)
    need(all(allowed(n) for n in set(now['files'])-set(old['files']))
         and all(allowed(n) for n in set(now['directories'])-set(old['directories'])),
         'append outside control/frozen GPU output/three policies')


def cpu_post_maps(expected,full):
    """Recompute every original CPU file AND directory from full terminal maps."""
    result={}
    for role,prefix in [('cpu_output','cpu/cpu/'),('campaign',CAMPAIGN)]:
        def keep(rel):
            if role=='cpu_output':
                return rel!='worker/artifacts/campaign' and not rel.startswith('worker/artifacts/campaign/')
            return rel.split('/')[0] not in {'policy-05','policy-06','policy-07'}
        files={n[len(prefix):]:v for n,v in full['files'].items() if n.startswith(prefix) and keep(n[len(prefix):])}
        dirs=sorted(n[len(prefix):] for n in full['directories'] if n.startswith(prefix) and keep(n[len(prefix):]))
        result[role]={'files':files,'directories':dirs}
        need(same(result[role],expected[role]),'complete original CPU postcustody map differs: '+role)
    return result


def history_extension(baseline,rows,journals,job,intent_pin,result_pin,normalize):
    prior=[r for r in rows if r[0]!=job];current=[r for r in rows if r[0]==job]
    need(job!='35498218' and len({r[0] for r in rows})==48 and len(prior)==47 and len(current)==1
         and type(baseline['terminal_jobs']) is int and baseline['terminal_jobs']==47
         and same(normalize(prior),normalize(baseline['related_accounting'])), 'exact47+1 parent rows required')
    need(type(baseline['submission_journals']) is int and baseline['submission_journals']==96
         and len(baseline['journal_pins'])==96,'exact failed47/96 journals required')
    names={CONTROL+'/submission.intent.json':intent_pin,CONTROL+'/submission.result.json':result_pin}
    need(not set(names)&set(baseline['journal_pins']) and len(journals)==98
         and journals=={**baseline['journal_pins'],**names},'exact96+2 original/new journals required')


def recovery(view,checker,decode,raw,validate_inventory):
    """Bind explicit new authority to the closed failure and independently recomputed post maps."""
    artifacts=view.c['artifacts'];p=view.p;outer=view.outer
    def pin(role):return artifacts[role]['sha256']
    def obj(role):return decode(view.artifact(role))
    for role,member in CONTROL_ROLES.items():
        view.h('control/'+member,pin(role))
        view.artifact(role)  # Independently hash actual supplied bytes, not merely map labels.
    for role,wanted in FAILED.items():
        need(pin(role)==wanted and sha(view.artifact(role))==wanted,'accepted failed47 authority changed: '+role)
    a=obj('recovery_authority');expected=obj('recovery_context');context=expected['execution_context']
    scoped=checker.validate_receipt(view.artifact('preflight_receipt'),pin('preflight_receipt'),
                                    view.artifact('recovery_context'),pin('recovery_context'))
    baseline=obj('baseline');failed=obj('failed_acceptance');fd=obj('failed_disposition');fr=obj('failed_remote_custody')
    need(failed['status']=='accepted_failed_prepolicy_attempt' and failed['required_findings']==0
         and failed['original_collector_returncode']==1 and failed['retroactive_collector_success'] is False
         and failed['runtime_qualified'] is failed['gpu_usage_qualified'] is False,
         'original failed attempt is not runtime acceptance or retroactive zero exit')
    need(baseline['input_pins']['acceptance']==pin('failed_acceptance')
         and baseline['input_pins']['disposition']==pin('failed_disposition')
         and baseline['terminal_jobs']==47 and baseline['submission_journals']==96,'closed failed baseline join')
    need(a['schema']=='ace-gpu-cpu-history-recovery-authority-v1' and a['status']=='operationally_adopted'
         and a['historical_postcustody_required'] is a['unchanged_scientific_inputs'] is True
         and a['original_entry_sha256']==ORIGINAL_ENTRY
         and a['original_source_commit']=='a9346022f145cc29439325b6f291597b2e6ec037'
         and a['original_gpu01_packet_sha256']=='436807328b4ee8ba0ec8cef3d5ff5b9c1d195b0b0165d5181a80356002f60fa7'
         and a['original_gpu01_job_id']==failed['job_id']=='35498218','explicit recovery authority required')
    need(a['control_root']==context['control_root']==CONTROL and a['failed_gpu01_root']==context['failed_gpu01_root']==OLD
         and a['allowed_new_output']==context['gpu_output']==p['paths']['gpu_output']==OUTPUT,
         'recovery root identities differ')
    need(same(a['baseline'],context['baseline']) and same(context['baseline'],
         {'sha256':pin('baseline'),'terminal_jobs':47,'submission_journals':96}), 'recovery47/96 context baseline differs')
    lineage=context['lineage'];proof=context['proof_pins']
    want_lineage={'cpu_acceptance_sha256':pin('cpu_acceptance'),'failed_gpu01_acceptance_sha256':pin('failed_acceptance'),
        'recovery_authority_sha256':pin('recovery_authority'),'source_sha256':a['historical_verifier_sha256'],
        'checker_sha256':OPERATIONAL[CHECKER],'namespace_sha256':pin('namespace_claim'),
        'original_namespace_sha256':pin('original_namespace_claim')}
    need(same(lineage,want_lineage) and a['cpu_acceptance_sha256']==pin('cpu_acceptance')
         and a['failed_gpu01_acceptance_sha256']==pin('failed_acceptance')
         and a['namespace_sha256']==pin('namespace_claim') and a['original_namespace_sha256']==pin('original_namespace_claim'),
         'explicit failure/CPU/namespace authority differs')
    need(same(proof,{'custody_sha256':pin('failed_seal'),'before_observation_sha256':pin('failed_before'),
        'after_observation_sha256':pin('failed_after'),'read_set_artifact_sha256':pin('readset')})
         and same(proof,a['historical_proof_pins']),'full failure/preflight proof join differs')
    seal=obj('failed_seal');copies=list(seal['copies'].values())
    need(len(copies)==2 and same(copies[0],copies[1])
         and copies[0]['files']['pull_terminal.py']['sha256']==a['historical_verifier_sha256'],
         'original sealed historical verifier differs')
    for member,role in [('disposition.json','failed_disposition'),('inventory_transport.json','failed_before'),
                        ('inventory_after_transport.json','failed_after'),('remote-custody.json','failed_remote_custody')]:
        need(copies[0]['files'][member]=={'sha256':pin(role),'bytes':len(view.artifact(role))},'failed seal role differs')
    validate_inventory(fr['full_inventory'])
    need(fr['full_backup_verified'] is True and fr['full_local_copy'] is False
         and fd['remote_custody_receipt_sha256']==pin('failed_remote_custody'), 'closed failure full backup join')
    for role in ('failed_before','failed_after'):
        transport=obj(role);need(transport['returncode']==0,'closed failed observation unavailable')
        obs=decode(transport['stdout'])
        need(same(obs['files'],fr['full_inventory']['files'])
             and sorted(obs['directories'])==fr['full_inventory']['directories'], 'closed full failed map differs')
    inventory_extension(fr,view.receipt)
    need(view.d['raw_root_labels']=={'control':CONTROL,'launch':OLD}
         and view.d['historical_postcustody_required'] is view.d['historical_postcustody_byte_comparison_performed'] is True,
         'collector recovery root/full-postcheck contract differs')
    before_maps=cpu_post_maps(expected['cpu_inventories'],{'files':view.before['files'],'directories':view.before['directories']})
    after_maps=cpu_post_maps(expected['cpu_inventories'],{'files':view.after['files'],'directories':view.after['directories']})
    cpu_post_maps(expected['cpu_inventories'],fr['full_inventory'])
    need(same(before_maps,after_maps),'full CPU postcustody observations differ')
    need(context['cpu_roots']=={'cpu_output':p['paths']['cpu_output'],'campaign':p['paths']['campaign']}
         and context['model_files_sha256']==digest(view.old['model_files'])
         and context['resources_sha256']==digest(p['resources']) and context['limits_sha256']==digest(p['limits']),
         'unchanged original model/resource/clock context differs')
    rs=obj('readset')
    need(set(rs)=={'schema','cpu_acceptance_sha256','campaign_root','files'}
         and rs['schema']=='ace-cpu-history-readset-v1' and rs['cpu_acceptance_sha256']==pin('cpu_acceptance')
         and rs['campaign_root']==p['paths']['campaign'] and set(rs['files'])==set(expected['consumed_read_set']),
         'complete detached read-set root/acceptance differs')
    for n,row in rs['files'].items():
        need(set(row)=={'sha256','bytes','utf8'} and type(row['utf8']) is str,'read-set member fields')
        data=row['utf8'].encode();info={'bytes':len(data),'sha256':sha(data)}
        need(type(row['bytes']) is int and info=={'bytes':row['bytes'],'sha256':row['sha256']}
             and info==expected['consumed_read_set'][n]==view.full[CAMPAIGN+n], 'captured read-set/full post map differs')
    recovery=outer['gpu_recovery'];need(view.d['gpu_recovery']==recovery,'collector/outer recovery differs')
    pairs={'authority_sha256':'recovery_authority','review_sha256':'recovery_review',
        'operational_source_review_sha256':'operational_review','preflight_review_sha256':'preflight_review',
        'failure_acceptance_sha256':'failed_acceptance','failure_seal_sha256':'failed_seal',
        'failure_disposition_sha256':'failed_disposition','preflight_receipt_sha256':'preflight_receipt'}
    need(all(recovery[k]==pin(v) for k,v in pairs.items()),'separate operational review/lineage pins differ')
    need(recovery['execution_context_sha256']==digest(context) and recovery['failed_root']==OLD
         and recovery['failed_job_id']=='35498218' and recovery['control_root']==CONTROL and recovery['output_root']==OUTPUT
         and recovery['historical_postcustody_required'] is True and recovery['original_collector_status']==fd['status'],
         'outer explicit recovery context differs')
    for role in ('recovery_review','operational_review','preflight_review'):
        need(view.artifact(role).strip(),'independent review bytes required; authority remains caller-owned')
    view.preflight=scoped
    view.facts['recovery_lineage']={'schema':'ace-gpu02-saved-recovery-lineage-evidence-v1','acceptance':False,
        'control_root':CONTROL,'failed_gpu01_root':OLD,'gpu_output':OUTPUT,'failed_gpu01_job_id':'35498218',
        'pins':{k:pin(k) for k in [*FAILED,'recovery_authority','recovery_context','preflight_receipt','readset',
                                  'operational_manifest','recovery_review','operational_review','preflight_review']},
        'original_collector_returncode':1,'retroactive_collector_success':False,
        'preflight_scope':'prelaunch consistency only; not continuous immutability',
        'full_cpu_postcustody_maps_recomputed':True,'cpu_postcustody_map_sha256':digest(after_maps),
        'all_failed47_ancestor_files_and_directories_preserved':True,'independent_postcustody_acceptance_required':True,
        'operational_sources':OPERATIONAL,'existing_gpu_acceptance_keys_changed':False}


def recovery_terminal(view):
    t=view.t;files=view.p['files'];marker=view.j('launch/gpu/controller/historical-postcustody-required.json')
    need(t['operational_recovery_authority_sha256']==files[RECOVERY_FILES['authority']]
         and t['historical_preflight_receipt_sha256']==files[RECOVERY_FILES['receipt']]
         and t['historical_postcheck_performed_on_gpu'] is False
         and t['full_external_postcustody_required'] is True,'v2 terminal cannot claim historical GPU postcheck/acceptance')
    need(same(marker,{'preflight':view.preflight,'authority_sha256':files[RECOVERY_FILES['authority']],
         'historical_postcheck_performed_on_gpu':False,'full_external_postcustody_required':True,'acceptance':False}),
         'complete separate historical postcustody marker differs')
