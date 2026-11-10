"""Text-only narrow derivation. Never execute the predecessor or audit aid."""
import ast
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent
old=(ROOT/'predecessor/batch03/batch_gpu_audit.py').read_bytes()
assert hashlib.sha256(old).hexdigest()=='555b571486500043118b730eb1a46f67886c01ef7af5734313da2b5c3df7abfd'
text=old.decode()

def replace(a,b,count=1):
    global text
    assert text.count(a)==count,(a,text.count(a),count)
    text=text.replace(a,b)

replace("CAMPAIGN =", """CONTROL_ROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261109_02'
FAILED_GPU_ROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108_01'
GPU_OUTPUT = FAILED_GPU_ROOT+'/gpu'
RECOVERY_FILES = {'authority':'gpu-recovery-authority.json','context':'cpu-history-expected.json',
                  'receipt':'cpu-history-preflight.json','read_set':'cpu-history-readset.json'}
SERVICES = {p:p+'.freeze.json' for p in ('frozen_llm','sft_llm','dpo_llm')}
CAMPAIGN =""")
replace('def select_members(full_inventory):','def select_members(full_inventory, *, control_binding, control_binding_sha256):')
replace("members = {name:full['files'][name] for name in audit_names(full)}", """control = _control_binding(b, full, control_binding, control_binding_sha256)
    names = set(audit_names(full)) | set(control['members'])
    members = {name:full['files'][name] for name in sorted(names)}""")
replace("def _selection(b, request):", """def _control_binding(b, full, binding, binding_sha256):
    \"\"\"Pure exact-byte joins for current control. No audit or authority execution.\"\"\"
    _pinned_object(b,binding,binding_sha256,'GPU02 control binding')
    b.need(type(binding) is dict and set(binding)=={'schema','outer','outer_sha256','inner','inner_sha256','result','result_sha256'}
           and binding['schema']=='ace-gpu02-audit-batch-control-v1', 'explicit recovery control binding required')
    for role,member in [('outer','control/packet.json'),('inner','control/inputs/packet.json'),
                        ('result','control/submission.result.json')]:
        payload=_pinned_object(b,binding[role],binding[role+'_sha256'],role+' control artifact')
        b.need(full['files'].get(member)==dict(bytes=len(payload),sha256=binding[role+'_sha256']),
               'current control artifact absent or different: '+member)
    outer,inner,result=(binding[k] for k in ('outer','inner','result'))
    b.need(outer['schema']=='ace-scientific-campaign-gpu-recovery-launch-v2'
           and inner['schema']=='ace-scientific-paired-campaign-gpu-cpu-history-v2'
           and outer['remote_root']==CONTROL_ROOT and inner['scope']=='development'
           and outer['gpu_packet_sha256']==binding['inner_sha256']
           and outer['cpu_acceptance_sha256']==inner['cpu_acceptance_sha256']
           and outer['cpu_packet_sha256']==inner['cpu_packet_sha256'], 'current recovery packet/CPU identities differ')
    b.need(type(outer['prior_jobs']) is int and outer['prior_jobs']==47
           and type(outer['prior_journals']) is int and outer['prior_journals']==96,
           'current control must descend from failed47/96')
    paths=inner['paths']
    b.need(outer['paths']==paths and paths['gpu_inputs']==CONTROL_ROOT+'/inputs'
           and paths['gpu_output']==GPU_OUTPUT and paths['campaign']==paths['cpu_output']+'/worker/artifacts/campaign'
           and inner['services']==SERVICES and inner['recovery_files']==RECOVERY_FILES,
           'frozen output and exact three control freezes required')
    def captured(member,pin):
        b.need(b.pin(pin) and full['files'].get(member,{}).get('sha256')==pin,
               'current control captured pin differs: '+member)
    for name,pin in inner['files'].items():
        b.name_ok(name);captured('control/inputs/'+name,pin)
    for name,pin in outer['new_input_files'].items():
        b.name_ok(name);captured('control/'+name,pin)
    b.need(outer['new_input_files']['inputs/packet.json']==binding['inner_sha256'], 'outer inner-member pin differs')
    captured('control/inputs/cpu-acceptance.json',inner['cpu_acceptance_sha256'])
    captured('control/inputs/cpu-packet.json',inner['cpu_packet_sha256'])
    captured('control/gpu-failure-baseline.json',outer['baseline_sha256'])
    lineage=outer['gpu_recovery']
    b.need(lineage['control_root']==CONTROL_ROOT and lineage['failed_root']==FAILED_GPU_ROOT
           and lineage['output_root']==GPU_OUTPUT and lineage['failed_job_id']=='35498218'
           and lineage['historical_postcustody_required'] is True
           and lineage['authority_sha256']==inner['files'][RECOVERY_FILES['authority']]
           and lineage['preflight_receipt_sha256']==inner['files'][RECOVERY_FILES['receipt']],
           'explicit old-failure/new-control recovery binding differs')
    for field,name in [('failure_acceptance_sha256','gpu-failure-acceptance.json'),
                       ('failure_seal_sha256','gpu-failure-seal.json'),
                       ('failure_disposition_sha256','gpu-failure-disposition.json')]:
        captured('control/'+name,lineage[field])
    job=result['job_id']
    b.need(type(job) is str and re.fullmatch('[1-9][0-9]*',job) is not None and job!='35498218'
           and type(result['returncode']) is int and result['returncode']==0 and result['ambiguous'] is False,
           'one unambiguous new submission identity required')
    captured('control/submission.intent.json',result['intent_sha256'])
    logs={'control/logs/'+job+suffix for suffix in ('.out','.err')}
    b.need({n for n in full['files'] if n.startswith('control/logs/')}==logs,
           'exact current-job stdout/stderr roster required')
    members={'control/inputs/cpu-packet.json',*('control/inputs/'+n for n in SERVICES.values()),*logs}
    b.need(members<=set(full['files']), 'required exact current control metadata/log member missing')
    return dict(job_id=job,members=tuple(sorted(members)),cpu_acceptance_sha256=inner['cpu_acceptance_sha256'],
                cpu_output=paths['cpu_output'],campaign_root=paths['campaign'])


def _selection(b, request):""")
replace("'batch_gpu72_audit'", "'batch_gpu72_recovery_audit'",count=2)
replace("    cpu=request['cpu_acceptance'];cpu_raw=", """    b.need(receipt['roots'].get('control')==CONTROL_ROOT and receipt['roots'].get('launch')==FAILED_GPU_ROOT,
           'receipt must distinguish new control from frozen old launch')
    control=_control_binding(b,full,request['control_binding'],request['control_binding_sha256'])
    b.need(control['cpu_output']==receipt['roots']['cpu']+'/cpu', 'control packet CPU physical root differs')
    cpu=request['cpu_acceptance'];cpu_raw=""")
replace("full['files'].get('launch/inputs/cpu-acceptance.json')", "full['files'].get('control/inputs/cpu-acceptance.json')")
replace("'GPU captured accepted CPU bytes differ')", "'current GPU02 control captured accepted CPU bytes differ')\n    b.need(control['cpu_acceptance_sha256']==request['cpu_acceptance_sha256'], 'control/independent CPU acceptance pin differs')")
replace("members=request['members'];expected=select_members(full)", "members=request['members'];expected=select_members(full,control_binding=request['control_binding'],control_binding_sha256=request['control_binding_sha256'])")
replace("schema='ace-gpu72-audit-batch-transfer-v1'", """schema='ace-gpu72-recovery-audit-batch-transfer-v2',
        control_binding_sha256=request['control_binding_sha256'],
        job_id=request['control_binding']['result']['job_id'],
        inner_packet_sha256=request['control_binding']['inner_sha256'],outer_packet_sha256=request['control_binding']['outer_sha256'],
        control_root=CONTROL_ROOT,failed_gpu01_root=FAILED_GPU_ROOT,gpu_output=GPU_OUTPUT""")
replace("                destination, helper_path, log_dir):", "                control_binding, control_binding_sha256, destination, helper_path, log_dir):")
replace("            destination=destination,helper_path=helper_path,log_dir=log_dir)", "            control_binding=control_binding,control_binding_sha256=control_binding_sha256,\n            destination=destination,helper_path=helper_path,log_dir=log_dir)")
replace("        b.write_new(logs/'accepted-cpu.json', b.raw(request['cpu_acceptance']))", "        b.write_new(logs/'accepted-cpu.json', b.raw(request['cpu_acceptance']))\n        b.write_new(logs/'recovery-control-binding.json', b.raw(request['control_binding']))")
replace("schema='ace-gpu72-audit-batch-retrieval-v1'", "schema='ace-gpu72-recovery-audit-batch-retrieval-v2'")
ast.parse(text)
with (ROOT/'batch_gpu_audit.py').open('x') as f:f.write(text)
