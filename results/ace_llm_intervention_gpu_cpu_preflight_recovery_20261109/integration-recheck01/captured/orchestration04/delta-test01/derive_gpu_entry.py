"""Derive an explicitly separate operational host entry, never edit originals.

Draft integration generator. Generated code needs helper and implementation
review, actual failure/preflight acceptance, and a concrete reviewed packet.
No source is executed by this module.
"""
import ast
import hashlib

ORIGINAL_SHA = '2694093512f8ce144bce7f33d6e2306bc251011b26567642c32006a7902e7db1'
PREFLIGHT = 'llm_intervention_cpu_preflight_receipt.py'
HISTORY = 'llm_intervention_cpu_history_admission.py'


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('original source anchor differs: ' + before[:80])
    return source.replace(before, after, 1)


def derive_entry(original, preflight_source, history_source):
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA:
        raise ValueError('independent original entry digest differs')
    source = original.decode('utf-8')
    source = replace_once(source, "VERSION = 'ace-scientific-paired-campaign-gpu-v1'",
        "VERSION = 'ace-scientific-paired-campaign-gpu-cpu-history-v2'")
    constants = "\n".join([
        "PREFLIGHT = " + repr(PREFLIGHT),
        "PREFLIGHT_SHA = " + repr(hashlib.sha256(preflight_source).hexdigest()),
        "HISTORY = " + repr(HISTORY),
        "HISTORY_SHA = " + repr(hashlib.sha256(history_source).hexdigest()),
        "RECOVERY_FILES = {'authority':'gpu-recovery-authority.json',",
        "    'context':'cpu-history-expected.json', 'receipt':'cpu-history-preflight.json',",
        "    'read_set':'cpu-history-readset.json'}",
    ])
    source = replace_once(source,
        "    'llm_intervention_scientific_gpu_phase.py', 'llm_intervention_scientific_service_evidence.py'}",
        "    'llm_intervention_scientific_gpu_phase.py', 'llm_intervention_scientific_service_evidence.py',\n"
        "    " + repr(PREFLIGHT) + ", " + repr(HISTORY) + "}")
    source = replace_once(source, "\nPOLICIES = (", "\n" + constants + "\nPOLICIES = (")
    source = replace_once(source,
        "              'freeze_sha256','namespace_admission_sha256'}",
        "              'freeze_sha256','namespace_admission_sha256','recovery_files'}")
    source = replace_once(source,
        "    expected_files = set(old['files'])|EXTRA_SOURCES|{'cpu-packet.json','cpu-acceptance.json'}",
        "    need(packet['recovery_files']==RECOVERY_FILES, 'exact operational recovery inputs required')\n"
        "    need(files.get(PREFLIGHT)==PREFLIGHT_SHA and files.get(HISTORY)==HISTORY_SHA,\n"
        "         'independently reviewed recovery helper sources differ')\n"
        "    expected_files = set(old['files'])|EXTRA_SOURCES|{'cpu-packet.json','cpu-acceptance.json'}|set(RECOVERY_FILES.values())")
    source = replace_once(source,
        "    bundle = admit_cpu(acceptance,packet,old,decode(data['prepared.json']))",
        "    recovery = recovery_contract(packet,data,root,acceptance,old,admission)\n"
        "    bundle = admit_cpu(acceptance,packet,old,decode(data['prepared.json']))")
    source = replace_once(source,
        "    need(inventory(paths['cpu_output'],exclude=excluded)==acceptance['cpu_output_files'],\n"
        "         'independent CPU output inventory differs')\n"
        "    need(inventory(paths['campaign'])==acceptance['campaign_files'],'independent CPU campaign inventory differs')",
        "    # Full historical byte/membership checks are bound by recovery_contract\n"
        "    # before this call and must be repeated in external terminal custody.\n"
        "    # Current consumed metadata below still uses independently pinned reads.")
    source = replace_once(source,
        "            'lineage':lineage,'launch':launch,'freezes':freezes,'contracts':contracts,'admission':admission,",
        "            'lineage':lineage,'launch':launch,'freezes':freezes,'contracts':contracts,'admission':admission,\n"
        "            'recovery':recovery,")
    source = replace_once(source, "\ndef recheck_inputs(ctx):", RECOVERY_FUNCTION + "\n\ndef recheck_inputs(ctx):")
    source = replace_once(source,
        "        bundle = ctx['bundle']\n        c,campaign_root = campaign._root(",
        "        bundle = ctx['bundle']\n"
        "        from llm_intervention_cpu_history_admission import capture_history\n"
        "        historical = capture_history(\n"
        "            acceptance_raw=data['cpu-acceptance.json'], acceptance_sha256=packet['cpu_acceptance_sha256'],\n"
        "            bundle_raw=read(Path(packet['paths']['cpu_output'])/ctx['acceptance']['bundle_path'],\n"
        "                            ctx['acceptance']['bundle_sha256']), bundle_sha256=ctx['acceptance']['bundle_sha256'],\n"
        "            readset_raw=data[RECOVERY_FILES['read_set']], readset_sha256=packet['files'][RECOVERY_FILES['read_set']],\n"
        "            campaign_source_raw=data['llm_intervention_campaign.py'],\n"
        "            campaign_source_sha256=packet['files']['llm_intervention_campaign.py'])\n"
        "        c,campaign_root = campaign._root(")
    source = replace_once(source,
        "        need(inventory(packet['paths']['cpu_output'],exclude='worker/artifacts/campaign')==ctx['acceptance']['cpu_output_files'],\n"
        "             'CPU output changed during GPU phase')\n"
        "        final_campaign=inventory(packet['paths']['campaign'])\n"
        "        need({n:h for n,h in final_campaign.items() if n.split('/')[0] not in {'policy-05','policy-06','policy-07'}}\n"
        "             ==ctx['acceptance']['campaign_files'],'campaign changed outside three GPU attempts')",
        "        save(directory/'historical-postcustody-required.json', {\n"
        "            'preflight':ctx['recovery']['preflight'],\n"
        "            'authority_sha256':packet['files'][RECOVERY_FILES['authority']],\n"
        "            'historical_postcheck_performed_on_gpu':False,\n"
        "            'full_external_postcustody_required':True, 'acceptance':False})")
    source = replace_once(source,
        "            'token_decoding_replayed':False,'cpu_finalization_and_external_acceptance_required':True,",
        "            'token_decoding_replayed':False,'cpu_finalization_and_external_acceptance_required':True,\n"
        "            'operational_recovery_authority_sha256':packet['files'][RECOVERY_FILES['authority']],\n"
        "            'historical_preflight_receipt_sha256':packet['files'][RECOVERY_FILES['receipt']],\n"
        "            'historical_postcheck_performed_on_gpu':False,'full_external_postcustody_required':True,")
    # A language-level with registers cleanup before entering its body.
    begin = source.index('        c,campaign_root = campaign._root(', source.index('def run_controller'))
    end = source.index('    finally:\n        sys.meta_path.remove(loader)', begin)
    body = source[begin:end]
    source = (source[:begin] + '        with historical.install(campaign):\n' +
              ''.join('    '+line if line.strip() else line for line in body.splitlines(keepends=True)) + source[end:])
    ast.parse(source)
    return source.encode('utf-8')


RECOVERY_FUNCTION = '''
def recovery_contract(packet,data,root,acceptance,old,admission):
    """Pinned historical preparation, not continuous immutability or acceptance."""
    names=RECOVERY_FILES
    checker=module('pinned_cpu_history_receipt',root/PREFLIGHT,data[PREFLIGHT])
    preflight=checker.validate_receipt(data[names['receipt']],packet['files'][names['receipt']],
        data[names['context']],packet['files'][names['context']])
    expected=decode(data[names['context']]);context=expected['execution_context']
    authority=decode(data[names['authority']])
    need(authority.get('schema')=='ace-gpu-cpu-history-recovery-authority-v1'
         and authority.get('status')=='operationally_adopted'
         and authority.get('historical_postcustody_required') is True
         and authority.get('unchanged_scientific_inputs') is True
         and authority.get('original_entry_sha256')=='2694093512f8ce144bce7f33d6e2306bc251011b26567642c32006a7902e7db1'
         and authority.get('original_source_commit')=='a9346022f145cc29439325b6f291597b2e6ec037'
         and authority.get('original_gpu01_packet_sha256')=='436807328b4ee8ba0ec8cef3d5ff5b9c1d195b0b0165d5181a80356002f60fa7'
         and authority.get('original_gpu01_job_id')=='35498218',
         'independently adopted operational recovery authority required')
    need(admission['accounting_sha256']==context['baseline']['sha256'],
         'stage admission differs from independently accepted recovery baseline')
    paths=packet['paths'];lineage=context['lineage']
    need(lineage['cpu_acceptance_sha256']==packet['cpu_acceptance_sha256']
         and lineage['recovery_authority_sha256']==packet['files'][names['authority']]
         and lineage['checker_sha256']==PREFLIGHT_SHA,
         'original CPU/recovery/checker authority differs')
    need(context['control_root']==str(root.parent)==authority['control_root']
         and context['gpu_output']==paths['gpu_output']==authority['allowed_new_output']
         and context['failed_gpu01_root']==authority['failed_gpu01_root']
         and context['cpu_roots']=={'cpu_output':paths['cpu_output'],'campaign':paths['campaign']},
         'explicit new control and frozen output roots differ')
    need(context['baseline']==authority['baseline']
         and lineage['failed_gpu01_acceptance_sha256']==authority['failed_gpu01_acceptance_sha256']
         and lineage['namespace_sha256']==authority['namespace_sha256']
         and lineage['original_namespace_sha256']==authority['original_namespace_sha256']
         and lineage['source_sha256']==authority['historical_verifier_sha256']
         and context['proof_pins']==authority['historical_proof_pins']
         and packet['cpu_acceptance_sha256']==authority['cpu_acceptance_sha256'],
         'failure/accounting/namespace lineage differs')
    digest=lambda value:sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    sources={n:packet['files'][n] for n in packet['source_names']}
    need(sources==authority['operational_source_files']
         and context['source_files_sha256']==digest(sources)
         and context['model_files_sha256']==digest(old['model_files'])
         and context['resources_sha256']==digest(packet['resources'])
         and context['limits_sha256']==digest(packet['limits']),
         'exact execution source/model/resource/clock binding differs')
    need(context['proof_pins']['read_set_artifact_sha256']==packet['files'][names['read_set']],
         'independent consumed read-set artifact differs')
    inventories=expected['cpu_inventories']
    for role,key in (('cpu_output','cpu_output_files'),('campaign','campaign_files')):
        need({n:v['sha256'] for n,v in inventories[role]['files'].items()}==acceptance[key],
             'complete preflight files differ from independent CPU acceptance')
    return {'preflight':preflight,'authority_sha256':packet['files'][names['authority']]}
'''
