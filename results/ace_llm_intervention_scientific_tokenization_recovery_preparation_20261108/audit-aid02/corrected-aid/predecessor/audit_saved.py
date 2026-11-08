"""Prospective TOKEN02 saved-evidence arithmetic only; never publishes acceptance.

Use only after independent full-custody authentication. No scientific package,
model, tensor, tokenizer, subprocess or remote import/call. Source-only prepared;
not yet executed against actual terminal evidence. See HANDOFF.md for gates that
must precede/follow these checks. Exact pinned closed pure functions are reused
by AST extraction; no source module's top-level statements execute.
"""
import ast
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
ENTRY = 'prepare_llm_intervention_scientific_training_cpu.py'
INNER = 'cab514d59959dd64c080d1414bd1dea6a47da59a93c55d3d23fe3b2fc338ee1c'
OUTER = '69aaaccdf9f20b11665f4708638db9ba621df4c9881befc1d4bb4225a46f912f'
FREEZE = '15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d'
NAMESPACE = '5027bca99d1c1cdfc9cf89dd5639593308e8eb37f7f7ad2fb8aacb6ded5c0e36'
BASELINE = '7a80a2300afde1ef71f8b303815b51fc2bb347515b78c40d972b4ec10138daf5'
WORLD_IDS = tuple(range(202611070000, 202611070032))
ROSTER = tuple(f'policy_train/{w}/snapshot-{i:03d}' for w in WORLD_IDS for i in range(5))
ACCEPTANCE_KEYS = frozenset(('schema','status','scope','required_findings','cpu_packet_sha256',
    'prepared_sha256','plan_sha256','freeze_sha256','namespace_admission_sha256',
    'complete_custody_verified','runtime_qualified_for_this_cpu_execution',
    'owned_process_cleanup_verified','training_execution_qualified','scientific_qualified',
    'job_id','numerical_plan_file_sha256','numerical_plan_sha256','exposure_sha256'))
FLAGS = dict(model_weights_loaded=False, explicit_cuda_api_called=False, inference_calls=0,
    environment_responses=0, training_updates=0, training_execution_qualified=False,
    scientific_qualified=False, retry_allowed=False)


def need(ok, text):
    if not ok:
        raise ValueError(text)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(x):
    return sha(canonical(x).encode())


def equal(a, b):
    return canonical(a) == canonical(b)


def pin(x):
    return type(x) is str and re.fullmatch('[0-9a-f]{64}', x) is not None


def decode(raw):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            need(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    def reject(x):
        raise ValueError('nonfinite JSON: '+x)
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject)
    canonical(value)
    return value


def safe_relative(value):
    need(type(value) is str and value and not value.startswith('/'), 'relative path required')
    p = PurePosixPath(value)
    need(str(p) == value and all(x not in ('', '.', '..') for x in p.parts), 'unsafe path')
    return value


def closed(name, functions, constants=None, extras=None):
    """Exact reviewed source function bodies only, with explicit global bindings."""
    pins = decode((ROOT/'reference-pins.json').read_bytes())
    raw = (ROOT/'source'/name).read_bytes()
    need(sha(raw) == pins['source/'+name]['sha256'], 'closed source changed: '+name)
    tree = ast.parse(raw)
    picked = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions]
    need({n.name for n in picked} == set(functions) and len(picked) == len(functions), 'function selection differs')
    need(all(not n.decorator_list for n in picked), 'decorated function prohibited')
    need(not any(isinstance(n, (ast.Import, ast.ImportFrom)) for f in picked for n in ast.walk(f)),
         'selected saved-evidence function imports a module')
    env = dict(need=need, sha=sha, digest=digest, canonical=canonical, decode=decode,
               pin=pin, safe_relative=safe_relative, Path=Path, re=re)
    env.update(constants or {}); env.update(extras or {})
    exec(compile(ast.Module(body=picked, type_ignores=[]), name+'[pure-only]', 'exec'), env)
    return SimpleNamespace(**{n:env[n] for n in functions})


def literal(name, key):
    pins = decode((ROOT/'reference-pins.json').read_bytes())
    raw = (ROOT/'source'/name).read_bytes()
    need(sha(raw) == pins['source/'+name]['sha256'], 'literal source changed')
    nodes = [n.value for n in ast.parse(raw).body if isinstance(n, ast.Assign)
             and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id==key]
    need(len(nodes)==1, 'unique literal required')
    return ast.literal_eval(nodes[0])


def accepted_batch(packet, proposal, acceptance, get):
    """get must read only the independently pinned original collection subtree.

    Reuses complete accepted collection binding, not collection generation or
    supervision scoring. Caller first verifies get's file roster/byte pins.
    """
    api = closed(ENTRY, ['collection_binding'], dict(WORLD_IDS=WORLD_IDS, ROSTER=ROSTER,
        PROPOSAL_SHA='585536b1408054ecebc7b7bb0e80055fa153b5bc7a699d8f2927db25ee745ea8',
        MAX_RECORD_BYTES=1024**2, MAX_BATCH_BYTES=64*1024**2))
    return api.collection_binding(packet, proposal, acceptance, get)


def recompute_exposure(batch):
    """Independent stdlib selection arithmetic from original training records.

    No numeric feature creation, risk scoring, response generation or draws.
    Public feasible action ordering matches the pinned task source.
    """
    version = 'ace-matched-training-exposure-proposal-v1'
    domain = version+'/unordered-command-pair'
    command = lambda a: canonical({'stop':True} if a is None else {'action_id':a})
    details, sft, dpo, nsft, ndpo = [], [], [], [], []
    counts = {k:0 for k in ('records','retained_pairs','strict_pairs','selected_pairs','unselected_pairs',
        'tied_pairs','records_without_strict_pairs','sft_steps','dpo_steps')}
    need(set(batch['records']) == set(ROSTER), 'full ordered 160 record batch required')
    for rid in ROSTER:
        raw = batch['records'][rid]; record = decode(raw)
        need(sha(raw)==batch['record_pins'][rid] and record['scope']=='policy_train'
             and record['supervision_receipt_sha256']==batch['receipt_pins'][rid], 'original record pin/scope differs')
        state = record['envelope']['state']
        need(type(state['terminal']) is bool, 'explicit terminal state required')
        ids = sorted(a['action_id'] for a in state['menu']
                     if not state['terminal'] and a['cost_units']<=state['remaining_units']) + [None]
        need(len(ids)==len(set(ids)) and record['supervised_action_id'] in ids, 'feasible action identity differs')
        rows, strict_index, seen = [], 0, set()
        for index, pref in enumerate(record['preferences']):
            need(set(pref)=={'left','right','tie','chosen','rejected'} and type(pref['tie']) is bool,
                 'exact original preference shape required')
            pair = frozenset((pref['left'],pref['right']))
            need(len(pair)==2 and pair<=set(ids) and pair not in seen, 'complete unique feasible pair required')
            seen.add(pair); tied= pref['tie']; commands=sorted((command(pref['left']),command(pref['right'])))
            need((pref['chosen'] is pref['rejected'] is None) if tied else
                 pref['chosen']!=pref['rejected'] and frozenset((pref['chosen'],pref['rejected']))==pair,
                 'strict/tied label association differs')
            rows.append(dict(preference_index=index,pair_index=None if tied else strict_index,
                preference_sha256=digest(pref), unordered_commands=commands,
                selection_sha256=digest([domain,rid,commands]), status='tied' if tied else 'unselected',exposure_count=0,
                numeric_chosen_index=None if tied else ids.index(pref['chosen']),
                numeric_rejected_index=None if tied else ids.index(pref['rejected']),
                chosen_command=None if tied else command(pref['chosen']),rejected_command=None if tied else command(pref['rejected'])))
            strict_index += int(not tied)
        strict = [r for r in rows if r['pair_index'] is not None]
        selected = min(strict,key=lambda r:(r['selection_sha256'],tuple(r['unordered_commands']))) if strict else None
        if selected is not None:
            selected.update(status='selected',exposure_count=1)
            step=dict(stage='dpo',record_id=rid,pair_index=selected['pair_index']);dpo.append(step)
            ndpo.append(dict(step,chosen_index=selected['numeric_chosen_index'],rejected_index=selected['numeric_rejected_index']))
        step=dict(stage='sft',record_id=rid,pair_index=None);sft.append(step)
        nsft.append(dict(step,chosen_index=ids.index(record['supervised_action_id']),rejected_index=None))
        details.append(dict(record_id=rid,world_id=record['world_id'],record_sha256=batch['record_pins'][rid],
            supervision_receipt_sha256=batch['receipt_pins'][rid],envelope_sha256=digest(record['envelope']),
            task_sha256=digest(record['envelope']['task']),action_ids=ids,supervised_command=command(record['supervised_action_id']),
            numeric_supervised_index=ids.index(record['supervised_action_id']),sft_exposure_count=1,
            strict_pair_exposures=[r['exposure_count'] for r in strict],dpo_exposure_count=int(selected is not None),pairs=rows))
        for k,v in dict(records=1,retained_pairs=len(rows),strict_pairs=len(strict),selected_pairs=int(selected is not None),
            unselected_pairs=len(strict)-int(selected is not None),tied_pairs=len(rows)-len(strict),records_without_strict_pairs=int(not strict)).items():
            counts[k]+=v
    counts.update(sft_steps=len(sft),dpo_steps=len(dpo),sft_action_rows=len(sft),dpo_action_rows=2*len(dpo))
    return dict(version=version,scope='policy_train',proposal_only=True,status='proposed' if dpo else 'untrainable_no_strict_pairs',
        has_both_training_stages=bool(dpo),record_roster=list(ROSTER),roster_sha256=digest(ROSTER),record_pins=batch['record_pins'],
        supervision_receipt_pins=batch['receipt_pins'],selection_domain=domain,
        selection_key_fields=['fixed_domain','record_id','sorted_unordered_commands'],selection_tiebreak='lexical_sorted_unordered_commands',
        records=details,schedule=sft+dpo,numeric_schedule=nsft+ndpo,counts=counts,
        bounds=dict(records=256,record_bytes=1024**2,total_record_bytes=64*1024**2,pairs_per_record=528,sft_steps=256,dpo_steps=256,total_steps=512),
        source_qualification='pending_caller',study_membership_qualified=False,scientific_adoption=False,
        equal_parameter_count_claimed=False,equal_compute_claimed=False)


def audit_artifacts(packet_raw, prepared_raw, numerical_raw, exposure_raw, batch):
    need(sha(packet_raw)==INNER,'exact reviewed TOKEN02 inner packet required')
    packet=decode(packet_raw); llm=decode(prepared_raw); num=decode(numerical_raw); exposure=decode(exposure_raw)
    need(prepared_raw==(canonical(llm)+'\n').encode(),'consumer requires original canonical prepared bytes')
    expected=recompute_exposure(batch)
    need(equal(exposure,expected),'full saved exposure/selection/schedule differs from independent arithmetic')
    need(exposure['counts']['records']==160 and exposure['counts']['retained_pairs']==3584,'full 160/3584 required')
    closed(ENTRY,['validate_plans'],dict(ROSTER=ROSTER)).validate_plans(llm,num,exposure,packet,batch,INNER)
    plan=llm['training_plan']
    audit_record_alignment(plan, batch, exposure)
    lens=[len(r['prompt_ids']) for r in plan['records']]
    need(all(0<n<=4744 for n in lens),'GPU consumer measured prompt bound exceeded; no truncation permitted')
    for key,name in (('collection_terminal_sha256','terminal.json'),('collection_inventory_sha256','inventory.json'),
                     ('collection_plan_sha256','plan.json')):
        need(llm[key]==packet['collection']['files'][name],'original collection lineage differs')
    recipe=closed('llm_intervention_ranker_training.py',['numeric_recipe'],
        dict(VERSION='ace-matched-numerical-training-v1')).numeric_recipe(artificial=False)
    need(equal(num['recipe'],recipe),'complete numerical recipe differs')
    width=num['feature_width']; need(type(width)is int and 1<=width<=20000,'numerical width bound')
    need(num['parameter_count']==width*128+128+128*128+128+128+1<=3000000
         and num['encoding']=='complete-padded-numeric-state-signed-log-v1' and num['retry_allowed'] is False,
         'numerical shape/encoding/retry differs')
    need(set(num['feature_pins'])==set(ROSTER),'all numerical feature pin metadata required')
    for row in exposure['records']:
        f=num['feature_pins'][row['record_id']]
        need(set(f)=={'shape','float32_tensor_sha256'} and equal(f['shape'],[len(row['action_ids']),width])
             and pin(f['float32_tensor_sha256']),'numerical feature metadata differs')
    d=exposure['counts']['dpo_steps']
    return dict(prepared_sha256=sha(prepared_raw),plan_sha256=digest(plan),numerical_plan_file_sha256=sha(numerical_raw),
        numerical_plan_sha256=digest(num),exposure_sha256=digest(exposure),records=160,retained_pairs=3584,
        sft_steps=160,dpo_steps=d,prompt_tokens_min=min(lens),prompt_tokens_max=max(lens),stored_tokens=plan['stored_tokens'],
        checkpoint_steps=[159,159+d],checkpoint_roles=['sft_reference_after_all_160_sft','final_after_all_D_dpo'],
        numerical_feature_tensors_recomputed=False,token_ids_retokenized=False,acceptance_authorized=False)


def audit_runtime(get, stage_raw, provenance, packet):
    """get: already authenticated saved worker JSON, never remote execution.

    provenance must be independently authenticated parent capture; matching it
    to a worker copy alone does not establish captured archive/source identity.
    """
    diagnostic_name='qualify_llm_intervention_runtime_metadata.py'
    diag=closed(diagnostic_name,['inside','check_prefix','check_metadata'],
        dict(GENERATED_MODULE='_remote_module_non_scriptable'))
    diag.CACHE_PAIRS=literal(diagnostic_name,'CACHE_PAIRS')
    checks=closed('llm_intervention_service_runtime.py',['check_stage','check_metadata'])
    checks.EXTRA_PACKAGES=literal('llm_intervention_service_runtime.py','EXTRA_PACKAGES')
    service=SimpleNamespace(RUNTIME=packet['runtime'])
    api=closed('qualify_llm_intervention_service_cpu.py',['validate_runtime_records'])
    stage=decode(stage_raw)
    api.validate_runtime_records(get,stage,sha(stage_raw),service,diag,checks,provenance)
    need(get('runtime-stage-binding.json')['gpu_mapping_enabled'] is False,'CPU cannot use GPU mapping exemption')
    expected={Path(n).stem:packet['files'][n] for n in packet['source_names'] if n!=ENTRY}
    for phase in ('after-service-imports','after-cpu-tokenization'):
        projects=get('runtime-'+phase+'.json')['captured_project_modules']
        need(set(expected)<=set(projects)<=set(expected)|{'__main__','__mp_main__'} and
             all(projects[n]['sha256']==h for n,h in expected.items()),'captured source closure differs')
        for n in set(projects)-set(expected):
            need(projects[n]['sha256']==packet['files'][ENTRY],'entry alias pin differs')
    return dict(runtime_records_consistent=True,stage_sha256=sha(stage_raw),acceptance_authorized=False)


def audit_lifecycle(parent, worker, cleanup, resources, started, marker, phases, *, worker_raw, stage_raw, facts):
    """All inputs are exact saved JSON; caller authenticates paths/pins first."""
    for t in (parent,worker):
        need(t['schema']=='ace-scientific-training-preparation-cpu-v1' and t['status']=='cpu_preparation_complete'
             and t['packet_sha256']==INNER and all(equal(t[k],v) for k,v in FLAGS.items()),'terminal/zero-work flags differ')
    need(parent['worker_terminal_sha256']==sha(worker_raw) and parent['stage_sha256']==sha(stage_raw)
         and worker['runtime_stage_sha256']==sha(stage_raw) and equal(parent['cleanup'],cleanup),'parent/worker/stage/cleanup join differs')
    for k, fk in [('training_plan_sha256','plan_sha256'),('numerical_plan_sha256','numerical_plan_sha256'),
                  ('exposure_sha256','exposure_sha256'),('llm_prepared_file_sha256','prepared_sha256'),
                  ('numerical_plan_file_sha256','numerical_plan_file_sha256'),('records','records'),
                  ('retained_pairs','retained_pairs'),('sft_steps','sft_steps'),('dpo_steps','dpo_steps')]:
        need(equal(worker[k],facts[fk]),'worker terminal differs: '+k)
    need(cleanup['group_still_exists'] is False and cleanup['errors']==[] and type(cleanup['returncode'])is int
         and cleanup['returncode']==0 and cleanup['kill_sent'] is False,'natural owned leader/group cleanup required')
    need(equal(resources['cleanup'],cleanup) and resources['errors']==[] and resources['startup_observed'] is True
         and type(resources['returncode']) is int and resources['returncode']==0,'complete worker resource receipt required')
    origin=started['started_monotonic']
    need(type(origin)in(int,float) and math.isfinite(origin) and started['pid']==started['pgid']
         and type(started['pid'])is int and started['pid']>0,'original monotonic origin and owned PGID required')
    for key,offset in [('startup_deadline',300),('deadline_monotonic',550),('supervisor_deadline_monotonic',590)]:
        need(started[key]==origin+offset,'deadline origin reset/offset differs')
    need(set(marker)=={'packet_sha256','pid','monotonic'} and marker['packet_sha256']==INNER and type(marker['pid']) is int and marker['pid']==started['pid']
         and type(marker['monotonic']) in (int,float) and math.isfinite(marker['monotonic'])
         and origin<=marker['monotonic']<origin+300,'startup marker identity/time differs')
    for value in (parent['elapsed_seconds'],resources['elapsed_including_cleanup_seconds']):
        need(type(value)in(int,float) and math.isfinite(value) and 0<=value<590,'absolute completion bound differs')
    closed('prepare_llm_intervention_training_cpu.py',['validate_phases']).validate_phases(phases)
    need(all(type(r['elapsed_seconds'])in(int,float) and math.isfinite(r['elapsed_seconds']) and 0<=r['elapsed_seconds']<550
             for r in phases),'work phase outside original work deadline')
    return dict(owned_process_cleanup_consistent=True,node_local_runtime_deletion_claimed=False,acceptance_authorized=False)


def validate_acceptance(value, facts, job_id):
    """Schema and pin consistency only, NOT authorization or an emitter."""
    need(type(job_id)is str and re.fullmatch('[1-9][0-9]*',job_id),'actual independent job ID required')
    expected=dict(schema='ace-scientific-training-preparation-acceptance-v1',status='accepted',scope='policy_train',
        required_findings=0,cpu_packet_sha256=INNER,freeze_sha256=FREEZE,namespace_admission_sha256=NAMESPACE,
        complete_custody_verified=True,runtime_qualified_for_this_cpu_execution=True,owned_process_cleanup_verified=True,
        training_execution_qualified=False,scientific_qualified=False,job_id=job_id)
    for k in ('prepared_sha256','plan_sha256','numerical_plan_file_sha256','numerical_plan_sha256','exposure_sha256'):
        need(pin(facts[k]),'actual artifact pin required: '+k);expected[k]=facts[k]
    need(type(value)is dict and set(value)==ACCEPTANCE_KEYS and equal(value,expected),'strict 18-field acceptance differs')
    return None


def audit_record_alignment(plan, batch, exposure):
    engine='llm_intervention_training_engine.py'
    # These are pure recipe functions; the parent checkpoint source supplies a literal.
    numeric_keys=literal(engine,'NUMERIC_KEYS')
    recipe=literal('qualify_llm_intervention_checkpoint_gpu.py','RECIPE')
    api=closed(engine,['gpu09_numeric_recipe','screen_numeric_recipe'],
               dict(NUMERIC_KEYS=numeric_keys,GPU09_RECIPE=recipe))
    need(equal(plan['numeric_recipe'],api.screen_numeric_recipe()),'complete LLM arithmetic recipe differs')
    expected_exposures={}
    prefix=('Choose the next experimental intervention using this public SCM, paid history and prediction task. '
        'The task queries and weights describe where accurate predictions are useful. They contain no observed answers. '
        'Predict the noise-disabled structural response. Error is squared error divided by scale squared. '
        'The tolerance is a goal, not evidence that it has been achieved. Never use an unseen evaluation result to stop. '
        'Every environment response costs the listed units. Return exactly one JSON object containing an affordable '
        'action_id from state.menu, or {"stop":true}. Do not invent measurements or actions.\n')
    for rec, detail in zip(plan['records'],exposure['records']):
        raw=decode(batch['records'][rec['record_id']]);meta=rec['token_metadata']
        command=lambda a: canonical({'stop':True} if a is None else {'action_id':a})
        legal={command(a) for a in detail['action_ids']}
        strict=[[command(p['chosen']),command(p['rejected'])] for p in raw['preferences'] if not p['tie']]
        ties=[[p['left'],p['right']] for p in raw['preferences'] if p['tie']]
        need(rec['scope']=='policy_train' and rec['supervised_command']==detail['supervised_command']
             and equal(rec['strict_pairs'],strict) and equal(rec['tied_pairs'],ties)
             and set(rec['commands'])==legal and meta['legal_commands']==len(legal)
             and len({tuple(v) for v in rec['commands'].values()})==len(legal), 'token label/legal command alignment differs')
        need(meta['task_sha256']==detail['task_sha256'] and meta['envelope_sha256']==detail['envelope_sha256']
             and meta['prompt_sha256']==sha((prefix+canonical(raw['envelope'])+'\nDecision:\n').encode()),
             'raw prompt/task/envelope association differs')
        expected_exposures[rec['record_id']]=dict(sft=1,dpo_pairs=detail['strict_pair_exposures'],tied_pairs_retained=len(ties))
    need(equal(plan['exposures'],expected_exposures) and plan['one_pass_sft'] is True
         and plan['one_pass_dpo'] is all(n==1 for e in expected_exposures.values() for n in e['dpo_pairs']),
         'complete selected/unexposed/tied record exposure differs')


def audit_publication(get_raw, packet, batch):
    # Caller separately rejects failure/interruption files over the WHOLE current root.
    for n in ('llm-prepared','numerical-plan'):
        need(get_raw(n+'.json')==get_raw(n+'.candidate.json'),'candidate/publication bytes differ')
    auth=decode(get_raw('authenticated.json'))
    need(auth['packet_sha256']==INNER and auth['source_sha256']=={n:packet['files'][n] for n in packet['source_names']}
         and auth['tokenizer_sha256']==digest(packet['model_files']), 'original input authentication differs')
    need(set(auth['tokenizer_files'])==set(packet['tokenizer_files']) and
         all(auth['tokenizer_files'][n]['sha256']==h for n,h in packet['tokenizer_files'].items()), 'six tokenizer byte pins differ')
    # Complete collection tree shape is checked with the parent's observed tree, not guessed here.
    for n,raw in batch['data'].items():
        need(get_raw('originals/'+n)==raw,'retained original collection metadata differs')
    for i,rid in enumerate(ROSTER):
        need(get_raw(f'originals/record-{i:03d}.json')==batch['records'][rid]
             and get_raw(f'originals/receipt-{i:03d}.json')==batch['receipts'][rid], 'retained original record/receipt differs')
    return auth


def audit_accounting(base, old_raw, *, job_id, result, observed, disposition, pins, remote_root):
    """Reuse the exact 6227 consumer's 43/88 -> 44/90 saved transition.

    pins must come from independently authenticated raw files, not declarations
    inside base. observed is the actual final remote inventory stdout object.
    This function does not create a baseline or infer missing costs.
    """
    need(sha(old_raw)==BASELINE,'independently accepted TOKEN01 failure baseline required')
    old=decode(old_raw)
    need(pins['token-failure-baseline']==BASELINE and type(job_id)is str and job_id!='35415334',
         'actual new job identity required')
    api=closed('gpu-consumer-build_packet.py',['normalized','accepted_token_transition'],dict(CPU_ROOT=remote_root))
    api.accepted_token_transition(base,old,pins,job_id,result,observed,disposition,dict(cpus=3,wall_seconds=600))
    need(base['terminal_jobs']==44 and base['submission_journals']==90,'exact actual successor counts required')
    need(sum(int(r[4])*int(r[5]) for r in base['related_accounting'])==base['closed_cpu_seconds'],
         'whole retained allocation CPU accounting differs')
    need(base['closed_cpu_seconds']<=57600 and base['closed_gpu_slice_seconds']==1601
         and base['cpu_admission_seconds']==base['closed_cpu_seconds']+3600<=57600,'actual bounds differ')
    return dict(terminal_jobs=44,submission_journals=90,closed_cpu_seconds=base['closed_cpu_seconds'],
        closed_gpu_slice_seconds=1601,preparation_allowance_cpu_seconds=3600,outstanding_reservation=0,
        actual_delta_cpu_seconds=base['closed_cpu_seconds']-old['closed_cpu_seconds'],acceptance_authorized=False)


if __name__=='__main__':
    raise SystemExit('Preparation only: no actual audit/acceptance CLI. Await complete terminal02 custody and independent pins; see HANDOFF.md.')
