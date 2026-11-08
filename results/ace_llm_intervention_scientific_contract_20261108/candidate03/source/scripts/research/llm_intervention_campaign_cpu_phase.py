"""CPU phase coupling for the new two-world artificial campaign.

The outer launcher owns captured-source/runtime/input authentication, Slurm
resources, absolute deadlines and whole-process cleanup. This module is not a
standalone launch authority. It trains the numerical control once, freezes all
evaluation commands, tokenizes bounded shape fixtures, then runs five CPU arms.
It does not launch a GPU service or compute private evaluation scores.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

VERSION = 'ace-artificial-campaign-cpu-phase-v1'


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _pin(raw):
    return hashlib.sha256(raw).hexdigest()


def _save(path, value):
    raw = (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    with path.open('xb') as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    return _pin(raw)


def shape_envelopes(plan, *, plan_sha256):
    """Two labelled fabricated public histories, not simulator observations.

    Probe initial-history and final-affordability shapes. A Gaussian history is
    not bounded by these numbers. Every real adaptive prompt must still pass
    the model service's original token bounds without truncation or repair.
    """
    from llm_intervention_contract import VERSION as STATE_VERSION, canonical
    from llm_intervention_paired_episode import _validate
    from llm_intervention_task import make_envelope
    _, world, task, _ = _validate(plan, plan_sha256)
    rows = []
    for count in (2,5):
        summaries = []
        for node in world.graph:
            width = 1+len(node.parents)
            summaries.append({'node':node.name,'terms':['intercept',*node.parents],
                'parameter_mean':[0.]*width,
                'parameter_covariance':[[float(i==j) for j in range(width)] for i in range(width)],
                'noise_variance':.25,'eligible_rows':count})
        state = {'version':STATE_VERSION,'objective':plan['objective'],'domain_notes':plan['domain_notes'],
            'graph':plan['learner']['graph'],'menu':plan['menu'],
            'history':[{'request_id':i,'action_id':'observe_all','kind':'observe','targets':{},
                'values':{n.name:0. for n in world.graph},'cost_units':1} for i in range(count)],
            'scm':summaries,'spent_units':count,'remaining_units':6-count,
            'requests':count,'returned_responses':count,'terminal':False}
        rows.append({'fixture':'fabricated_initial_shape' if count==2 else 'fabricated_late_affordability_shape',
                     'environment_responses':0,'envelope':make_envelope(json.loads(canonical(state)),task)})
    return rows


def policy_specs(*, numeric_bindings, numeric_prefix, ranker_prefix, ranker_export,
                 persistent_specs, numeric_provenance, ranker_provenance):
    """Bind five CPU commands to fixed source prefixes and the final ranker only."""
    from llm_intervention_artificial_campaign import WORLD_IDS, NUMERIC_POLICIES, LLM_POLICIES
    from llm_intervention_contract import canonical
    _need(set(persistent_specs)==set(LLM_POLICIES), 'all three fixed persistent specs required')
    for prefix in (numeric_prefix,ranker_prefix):
        _need(type(prefix) in (list,tuple) and prefix and all(type(v) is str and v for v in prefix),
              'captured CPU command prefixes required')
    _need(set(numeric_provenance)==set(NUMERIC_POLICIES), 'four numerical source provenance pins required')
    _need(set(ranker_export)=={'path','sha256','training_receipt_sha256'},'fixed ranker export binding required')
    rows={(r['world_id'],r['policy_id']):r for r in numeric_bindings}
    result={}
    for wid in WORLD_IDS:
        specs=[]
        for pid in NUMERIC_POLICIES:
            row=rows[wid,pid]
            specs.append({'policy_id':pid,'kind':'task_process','command':list(numeric_prefix)+[
                '--binding-json',canonical(row['binding']),'--binding-sha256',row['binding_sha256']],
                'timeout_seconds':30.,'identity_sha256':None,'policy_provenance_sha256':numeric_provenance[pid]})
        specs.append({'policy_id':'learned_ranker','kind':'task_process','command':list(ranker_prefix)+[
            '--method','learned_ranker','--seed','109','--draws','32',
            '--checkpoint',ranker_export['path'],'--checkpoint-sha256',ranker_export['sha256'],
            '--training-receipt-sha256',ranker_export['training_receipt_sha256']],
            'timeout_seconds':30.,'identity_sha256':None,'policy_provenance_sha256':ranker_provenance})
        specs.extend(persistent_specs[pid] for pid in LLM_POLICIES)
        result[wid]=specs
    return result


def run_cpu_phase(*, output, campaign_output, accepted_training_inputs, collection_inventory,
                  records, numeric_prefix, ranker_prefix, persistent_specs, services,
                  numeric_provenance, outer_provenance_receipt_sha256, tokenizer, event):
    """Run once on a separately authenticated CPU allocation, preserving failure.

    accepted_training_inputs is admit_training's exact captured-byte arguments.
    records maps the original four record IDs to their unchanged captured bytes.
    Prefixes must execute the captured screen-policy and numerical-policy CLIs.
    Neither the API nor metadata alone proves those outer execution properties.
    """
    _need(not os.environ.get('CUDA_VISIBLE_DEVICES') and not os.environ.get('SLURM_JOB_GPUS')
          and not os.environ.get('SLURM_STEP_GPUS'),'CPU phase cannot own a GPU allocation')
    out=Path(output).absolute(); out.mkdir(exist_ok=False)
    started=time.monotonic()
    try:
        from llm_intervention_campaign_training_lineage import admit_training
        from llm_intervention_artificial_campaign import build_artificial_campaign,numeric_policy_bindings,POLICY_IDS
        from llm_intervention_contract import canonical,digest
        import llm_intervention_campaign as campaign
        import llm_intervention_ranker_training as ranker
        from llm_intervention_language_policy import prepare_task_tokens
        import torch
        event('cpu_training_lineage_start')
        lineage=admit_training(**accepted_training_inputs)
        _save(out/'training-lineage.json',lineage)
        prepared=json.loads(accepted_training_inputs['prepared_raw'])
        inventory=collection_inventory
        _need(digest(prepared['exposure'])==lineage['exposure_sha256'], 'accepted exposure differs')
        roster=tuple(inventory['record_roster'])
        _need(roster==('artificial/17/snapshot-000','artificial/17/snapshot-001',
                      'artificial/4/snapshot-000','artificial/4/snapshot-001')
              and set(records)==set(roster)
              and all(_pin(records[rid])==inventory['record_pins'][rid] for rid in roster),
              'complete unchanged accepted numerical training batch required')
        # Authenticating inventory bytes against the original collection pin is
        # an outer input obligation. Store the exact supplied object for review.
        _save(out/'collection-inventory.json',inventory)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True,warn_only=False)
        plan=ranker.prepare_plan(records=records,record_pins=inventory['record_pins'],
            supervision_receipt_pins=inventory['supervision_receipt_pins'],record_roster=roster,
            scope='artificial',exposure=prepared['exposure'],exposure_sha256=lineage['exposure_sha256'],
            recipe=ranker.numeric_recipe(artificial=False),checkpoint_steps=(3,5),
            outer_provenance_receipt_sha256=outer_provenance_receipt_sha256)
        plan_pin=digest(plan); _save(out/'numeric-training-plan.json',plan)
        event('cpu_ranker_training_start',plan_sha256=plan_pin)
        fitted=ranker.fit(plan=plan,plan_sha256=plan_pin,output=out/'ranker')
        _need(fitted['status']=='complete' and fitted['completed_steps']==6
              and fitted['sft_steps']==4 and fitted['dpo_steps']==2,'matched numerical training failed')
        final=fitted['exports']['dpo']
        export={'path':str(out/'ranker'/final['path']),'sha256':final['sha256'],
                'training_receipt_sha256':final['training_receipt_sha256']}
        _need(_pin(Path(export['path']).read_bytes())==export['sha256']
              and final['all_training_score_rows_equal'] is True,'fixed final export or reload differs')
        event('cpu_ranker_training_complete',training_updates=6,export_sha256=export['sha256'])
        specs=policy_specs(numeric_bindings=numeric_policy_bindings(),numeric_prefix=numeric_prefix,
            ranker_prefix=ranker_prefix,ranker_export=export,persistent_specs=persistent_specs,
            numeric_provenance=numeric_provenance,ranker_provenance=_pin((out/'ranker/terminal.json').read_bytes()))
        bundle=build_artificial_campaign(policy_specs_by_world=specs,services=services,
            outer_provenance_receipt_sha256=outer_provenance_receipt_sha256)
        _save(out/'campaign-bundle.json',bundle)
        event('campaign_frozen_before_responses',campaign_sha256=bundle['campaign_sha256'])
        token_rows=[]
        event('campaign_tokenization_start')
        for source in bundle['campaign']['sources']:
            for probe in shape_envelopes(source['plan'],plan_sha256=source['plan_sha256']):
                grammar,metadata=prepare_task_tokens(probe['envelope'],tokenizer,max_tokens=8192,max_new_tokens=32)
                token_rows.append({'world_id':source['plan']['world']['world_id'],**probe,**metadata,
                    'prompt_ids':list(grammar.prompt_ids),
                    'completion_ids':[list(ids) for ids in grammar.completion_ids]})
        _save(out/'tokenization.json',{'scope':'fabricated shape probes only','rows':token_rows,
            'environment_responses':0,'no_all_adaptive_prompt_bound_claim':True})
        event('campaign_tokenization_complete',fixture_count=len(token_rows))
        binding={k:bundle[k] for k in ('campaign','campaign_sha256','source_plan_pins')}
        binding['output']=Path(campaign_output).absolute()
        campaign.initialize_campaign(**binding)
        attempts={}
        for pid in POLICY_IDS[:5]:
            event('cpu_policy_start',policy_id=pid)
            result=campaign.run_policy(**binding,policy_id=pid,phase='cpu')
            attempts[pid]=result['sha256']
            event('cpu_policy_complete',policy_id=pid,attempt_sha256=result['sha256'])
        _save(out/'attempt-pins.json',attempts)
        # Operational failures are preserved in the campaign. They prevent GPU
        # admission, while the complete five-arm evidence remains available.
        failed=[]
        for pid,pin in attempts.items():
            attempt,_=campaign._load_attempt(bundle['campaign'],bundle['campaign_sha256'],
                campaign._attempt(binding['output'],bundle['campaign'],pid),pid,pin)
            if attempt['errors'] or any(a['status']!='completed' for a in attempt['arms']): failed.append(pid)
        terminal={'schema':VERSION,'status':'cpu_phase_complete' if not failed else 'cpu_phase_contains_failures',
            'campaign_sha256':bundle['campaign_sha256'],'attempt_pins':attempts,'failed_policies':failed,
            'numeric_training_updates':6,'llm_training_updates':0,'gpu_services_started':0,
            'private_scores_computed':False,'scientific_qualified':False,'retry_allowed':False,
            'response_accounting_source':'original per-arm charged journals; not zero-response preparation',
            'elapsed_seconds':time.monotonic()-started,'external_acceptance_required':True}
        _save(out/'terminal.json',terminal)
        return terminal
    except BaseException as exc:
        _save(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),
            'traceback':traceback.format_exc(),'retry_allowed':False,'scientific_qualified':False})
        raise
