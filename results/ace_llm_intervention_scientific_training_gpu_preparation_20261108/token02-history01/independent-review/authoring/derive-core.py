from pathlib import Path
import re,hashlib,json,shutil
root=Path(Path('/tmp/ace-scientific-training-gpu-token02-current.txt').read_text().strip())
old=(root/'predecessors/core/build_packet.py').read_text();src=old
producer=Path('/tmp/ace-scientific-token02-authoring-8tbddys1/recovery_contract.py.txt')
shutil.copyfile(producer,root/'predecessors/token02-recovery-contract-draft.txt')
contract=producer.read_text().split('\ndef admit(supplied):',1)[0]
# Adapt the already explicit producer metadata joins, retaining the GPU's original collection baseline name.
contract=contract.replace("m['baseline']","m['previous-baseline']").replace("pins['baseline']","pins['previous-baseline']")
contract=contract.replace("m['failure-disposition']","m['collection-failure-disposition']")
contract=contract.replace("('failed01','failure-disposition')","('failed01','collection-failure-disposition')")
contract=contract.replace("all(relative(n) and pin(h)","all(relative('launch/'+n) and pin(h)")
for a,b in [('RESOURCES','TOKEN_RESOURCES'),('ORIGINAL_DEADLINES','TOKEN_ORIGINAL_DEADLINES'),('DEADLINES','TOKEN_DEADLINES'),('ENTRY_SHA','TOKEN_ENTRY_SHA'),('ENTRY','TOKEN_ENTRY'),('ROOT','CPU_ROOT')]:
 contract=re.sub(r'\b'+a+r'\b',b,contract)
contract=contract.replace("len(stage['source_names'])==32","stage['source_names']==m['freeze']['stage_closure']['stages']['token']['source_names']")
# Complete transport metadata root/job identity, not invented disposition fields.
needle="    need(all(first[k]==last[k]"
contract=contract.replace(needle,"    need(all(literal(m[n]['python_source'],'ROOT')==FAILED_TOKEN_ROOT and literal(m[n]['python_source'],'JOB')==FAILED_TOKEN_JOB\n             for n in ('token-failure-inventory','token-failure-readback')), 'failed token captured inventory root/job differs')\n"+needle)
const='''
# Explicit TOKEN02 history only; all training source/resources and TRAIN01 root stay frozen.
FREEZE_SHA=FIXED['freeze']
FAILED_TOKEN_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01'
FAILED_TOKEN_JOB='35415334'
FAILED_COLLECTION_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_01'
TOKEN_ENTRY='prepare_llm_intervention_scientific_training_cpu.py'
TOKEN_ENTRY_SHA='47bbf574269b9187f1237e4a9731115228b06bbb5ab65937df3e0f14bd37b8f3'
TOKEN_RESOURCES=dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=3,memory_mib=11520,wall_seconds=600,gpus=0,requeue=False)
TOKEN_ORIGINAL_DEADLINES=dict(startup=120,work=550,cleanup=570,hard=590)
TOKEN_DEADLINES=dict(startup=300,work=550,cleanup=570,hard=590)
FAILED_TOKEN_PACKET_SHA='a831737c6bc24a3b38269e36fa347ed3934c7803e7e18b0738881ac0b8fc323d'
FAILED_TOKEN_STAGE_SHA='101d88f3ff15ed322273d63ad281de05afd921754102a907f7232ac000801e7b'
'''
result=Path('/Users/pat/code/ACE/results/ace_llm_intervention_scientific_tokenization_20261108/launch01/submission.result.json').read_bytes()
const+="FAILED_TOKEN_RESULT_SHA="+repr(hashlib.sha256(result).hexdigest())+'\n'
const+="FIXED['collection-failure-disposition']='fd3a52be3054c1dd8d0d2b324dea3e64bdf8bc3676c28e7367cc0bbf06a3c685'\n"
const+="TEXT_METADATA |= {'token-operational-review','token-operational-adoption-review','token-operational-guide'}\n"
newnames=['collection-failure-disposition','failed-token-packet','failed-token-stage','failed-token-result','token-failure-acceptance','token-no-learnedwork-proof','token-failure-disposition','token-failure-inventory','token-failure-readback','token-failure-custody-inventory','token-failure-baseline','token-operational-amendment','token-operational-review','token-operational-adoption-review','token-operational-guide','token-diagnosis']
const+='TOKEN_METADATA='+repr(tuple(newnames))+'\nMETADATA+=TOKEN_METADATA\n'
src=src.replace("CPU_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01'","CPU_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_02'",1)
src=src.replace('\n\ndef need(ok,message):','\n'+const+'\n\ndef need(ok,message):',1)
src=src.replace("('baseline_sha256','previous-baseline')","('baseline_sha256','token-failure-baseline')",1)
src=src.replace('\ndef authenticate(supplied,prepared_raw', '\n'+contract+'\n\ndef authenticate(supplied,prepared_raw',1)
src=src.replace("base=m['baseline'];old=m['previous-baseline']","base=m['baseline'];old=m['token-failure-baseline']",1)
src=src.replace("stages['tokenize']['remote_root']==CPU_ROOT","stages['tokenize']['remote_root']==FAILED_TOKEN_ROOT",1)
src=src.replace("    collection_originals=collection_lineage(m,pins)\n","""    collection_originals=collection_lineage(m,pins)
    failed_token=failed_token_transition(m,pins)
    token_recovery=token_operational_binding(supplied,m,pins)
    token_science_equality(m['failed-token-stage'],cpu)
    token_outer=m['cpu-launch-packet']
    for name in TOKEN_METADATA:
        if name=='collection-failure-disposition':continue  # Producer uses original failure-disposition.json.
        need(token_outer['new_input_files'][metadata_name(name)]==pins[name], 'TOKEN02 original recovery member differs: '+name)
    need(token_outer['baseline_sha256']==pins['token-failure-baseline']
         and token_outer['token_operational_amendment_sha256']==pins['token-operational-amendment']
         and token_outer['token_operational_adoption_review_sha256']==pins['token-operational-adoption-review']
         and token_outer['token_failure_acceptance_sha256']==pins['token-failure-acceptance']
         and token_outer['token_no_learnedwork_proof_sha256']==pins['token-no-learnedwork-proof']
         and token_outer['resources']==TOKEN_RESOURCES and token_outer['deadlines']==TOKEN_DEADLINES,
         'TOKEN02 exact adopted startup/root-only recovery lineage differs')
""",1)
src=src.replace("old['terminal_jobs']==42","old['terminal_jobs']==43",1).replace("old['submission_journals']==86","old['submission_journals']==88",1)
# Only authenticate's accepted transition predecessor: collection function remains the original42 baseline.
start=src.index('def authenticate(')
head,tail=src[:start],src[start:]
tail=tail.replace("==pins['previous-baseline']","==pins['token-failure-baseline']",1)
src=head+tail
src=src.replace("upstream_pins={**collection_originals,**{CPU_ROOT", "failed_token=failed_token,token_recovery=token_recovery,\n        upstream_pins={**collection_originals,**{FAILED_TOKEN_ROOT+'/'+n:h for n,h in failed_token['preserved_roots'][FAILED_TOKEN_ROOT].items()},**{CPU_ROOT",1)
src=src.replace("        operational_amendment_sha256=ctx['pins']['operational-amendment'],", "        operational_amendment_sha256=ctx['pins']['operational-amendment'],\n        token_recovery=ctx['token_recovery'],failed_token_root=FAILED_TOKEN_ROOT,failed_token_job_id=FAILED_TOKEN_JOB,",1)
(root/'build_packet.py').write_text(src)
# Launcher already independently authenticates exact metadata and checks all historical rows/journals twice.
launcher=(root/'predecessors/core/launch_once.py').read_text()
launcher=launcher.replace("        claim_pin=ctx['pins']['namespace-claim']", "        b.need(packet['token_recovery']==ctx['token_recovery'] and packet['failed_token_root']==b.FAILED_TOKEN_ROOT\n               and packet['failed_token_job_id']==b.FAILED_TOKEN_JOB,'exact TOKEN02 recovery history differs')\n        claim_pin=ctx['pins']['namespace-claim']",1)
launcher=launcher.replace("            cpu_job_id=m['cpu-acceptance']['job_id'],existing_namespace_claim_sha256=claim_pin,", "            cpu_job_id=m['cpu-acceptance']['job_id'],existing_namespace_claim_sha256=claim_pin,\n            token_recovery=ctx['token_recovery'],failed_token_job_id=b.FAILED_TOKEN_JOB,failed_token_root=b.FAILED_TOKEN_ROOT,",1)
(root/'launch_once.py').write_text(launcher)
shutil.copyfile(__file__,root/'derive-core.py')
print(root)
