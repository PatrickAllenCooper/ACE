"""Source derivation only. Does not import or execute any captured program."""
from pathlib import Path
import ast, hashlib, difflib

HERE=Path(__file__).resolve().parent
OLD=HERE/'predecessor'
NEW=HERE/'implementation'
GPU=HERE.parent/'coordinator02/authoring/build_packet.py'
ENTRY='finalize_llm_intervention_scientific_campaign_cpu.py'
VERSION='ace-scientific-campaign-finalizer-gpu-history-v2'
GPU_VERSION='ace-scientific-paired-campaign-gpu-cpu-history-v2'

def change(text, old, new, count=1):
    assert text.count(old)==count, (old[:100],text.count(old),count)
    return text.replace(old,new)

def function(text,name):
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return '\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n'

def emit(name,text):
    ast.parse(text)
    (NEW/name).write_text(text)
    (HERE/'diffs'/(name+'.diff')).write_text(''.join(difflib.unified_diff(
        (OLD/name).read_text().splitlines(True),text.splitlines(True),fromfile='predecessor/'+name,tofile='implementation/'+name)))

s=(OLD/ENTRY).read_text()
s=change(s,"VERSION = 'ace-scientific-campaign-finalizer-cpu-v1'",'VERSION = '+repr(VERSION))
s=change(s,"    'source_commit', 'freeze_sha256', 'namespace_admission_sha256'}",
    "    'source_commit', 'freeze_sha256', 'namespace_admission_sha256',\n    'operational_source_commit', 'gpu_recovery_acceptance_sha256'}")
s=change(s,"and re.fullmatch('[0-9a-f]{40}', p['source_commit']),", "and re.fullmatch('[0-9a-f]{40}', p['source_commit'])\n         and type(p['operational_source_commit']) is str\n         and re.fullmatch('[0-9a-f]{40}', p['operational_source_commit']),")
s=change(s,"gp['schema'] == 'ace-scientific-paired-campaign-gpu-v1'",'gp[\'schema\'] == '+repr(GPU_VERSION))
s=change(s,"{ENTRY, GPU_ENTRY, 'gpu-packet.json', 'gpu-acceptance.json'}", "{ENTRY, GPU_ENTRY, 'gpu-packet.json', 'gpu-acceptance.json'} | RECOVERY_ADDITIONS")
s=change(s,"    for name in ('cpu-packet', 'gpu-packet', 'cpu-acceptance', 'gpu-acceptance'):","    for name in ('cpu-packet', 'gpu-packet', 'cpu-acceptance', 'gpu-acceptance', 'gpu-recovery-acceptance'):")
s=change(s,"    api = types.SimpleNamespace(inventory=inventory)", "    checker = module('finalizer_preflight_checker', root/PREFLIGHT, data[PREFLIGHT])\n    api = types.SimpleNamespace(inventory=inventory, check_preflight=checker.validate_receipt)")
s=change(s,"    # First private bundle decode:","    admit_recovery(packet, gpu_packet, data, gpu, gt, api)\n    # First private bundle decode:")
s=change(s,"def expected_numeric_steps(data):",(HERE/'child-gate.txt').read_text()+"\n\ndef expected_numeric_steps(data):")
s=change(s,"GPU_ACCEPTANCE_KEYS =",(HERE/'child-constants.txt').read_text()+"\nGPU_ACCEPTANCE_KEYS =")
emit(ENTRY,s)
entry_sha=hashlib.sha256(s.encode()).hexdigest()

s=(OLD/'build_packet.py').read_text()
s=change(s,"GPU_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108_01'",
    "GPU_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261109_02'\nGPU_OUTPUT_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108_01'")
s=change(s,"ENTRY_SHA='28c5fc6b426cac70cf4bae4895350c560cd43080937a32e41d936f072005b45a'",
    "FROZEN_ENTRY_SHA='28c5fc6b426cac70cf4bae4895350c560cd43080937a32e41d936f072005b45a'\nENTRY_SHA="+repr(entry_sha))
s=change(s,"VERSION='ace-scientific-campaign-finalizer-cpu-v1'",'VERSION='+repr(VERSION))
s=change(s,"OUTER_VERSION='ace-scientific-finalization-launch-v1'","OUTER_VERSION='ace-scientific-finalization-gpu-recovery-launch-v2'")
s=change(s,'RAW_ROOTS=dict(launch=GPU_ROOT,','RAW_ROOTS=dict(control=GPU_ROOT,launch=GPU_OUTPUT_ROOT,')
s=change(s,"('gpu',GPU_ROOT,47,96,4,1,2880)","('gpu',GPU_ROOT,48,98,4,1,2880)")
s=change(s,"s['entry_sha256']==ENTRY_SHA","s['entry_sha256']==FROZEN_ENTRY_SHA")
s=change(s,"gpu='ace-scientific-campaign-gpu-launch-v1'","gpu='ace-scientific-campaign-gpu-recovery-launch-v2'")
s=change(s,"        a,p,r,d,t,base=(m[prefix+'-'+kind]", "        if prefix=='gpu':\n            observations['failed-gpu-root']=failed_gpu_history(m,pins)\n            previous=m['gpu-failure-baseline'];previous_pin=pins['gpu-failure-baseline']\n            jobs.append(FAILED_GPU_JOB)\n        a,p,r,d,t,base=(m[prefix+'-'+kind]")
s=change(s,"obs['files']['launch/'+n]['sha256']==h", "obs['files'][('control/' if prefix=='gpu' else 'launch/')+n]['sha256']==h")
s=change(s,"gp['schema']=='ace-scientific-paired-campaign-gpu-v1'",'gp[\'schema\']=='+repr(GPU_VERSION))
s=change(s,"gpu_inputs=GPU_ROOT+'/inputs',gpu_output=GPU_ROOT+'/gpu'", "gpu_inputs=GPU_ROOT+'/inputs',gpu_output=GPU_OUTPUT_ROOT+'/gpu'")
s=change(s,"    gpu_authority(m,pins)\n", "    gpu_authority(m,pins)\n    recovery=recovery_authority(supplied,m,pins,observations)\n")
s=change(s,"return dict(metadata=m,pins=pins,observations=observations,token_recovery=token_recovery)",
    "return dict(metadata=m,pins=pins,observations=observations,token_recovery=token_recovery,gpu_recovery=recovery)")
s=change(s,"files['launch/inputs/packet.json']", "files['control/inputs/packet.json']")
s=change(s,"files['launch/inputs/'+n]", "files['control/inputs/'+n]")
s=change(s,"gate.capture('launch/inputs/'+n", "gate.capture('control/inputs/'+n")
s=change(s,"        need(after[k]==obs[k], 'remote custody changed before/after: '+k)",
    "        if k=='directories':\n            need(type(after[k]) is list and type(obs[k]) is list\n                 and len(set(after[k]))==len(after[k]) and len(set(obs[k]))==len(obs[k])\n                 and sorted(after[k])==sorted(obs[k]), 'remote directories changed before/after')\n        else:need(after[k]==obs[k], 'remote custody changed before/after: '+k)")
old_func=function(s,'make_packet')
s=change(s,old_func,(HERE/'make-packet.txt').read_text()+'\n')
s=change(s,"base['terminal_jobs']==47 and base['submission_journals']==96", "base['terminal_jobs']==48 and base['submission_journals']==98")
s=change(s,"'actual released47/96 baseline required'", "'actual released48/98 baseline required'")
s=change(s,"len(normalized(base['related_accounting']))==47 and len(base['journal_pins'])==96", "len(normalized(base['related_accounting']))==48 and len(base['journal_pins'])==98")
s=change(s,"prior_jobs=47,prior_journals=96", "prior_jobs=48,prior_journals=98")
s=change(s,"all47 prior rows/96journals and extend48/98", "all48 prior rows/98journals and extend49/100")
s=change(s,"gpu_job_id=ctx['metadata']['gpu-acceptance']['job_id'],token_recovery=ctx['token_recovery'],",
    "gpu_job_id=ctx['metadata']['gpu-acceptance']['job_id'],token_recovery=ctx['token_recovery'],\n            gpu_recovery=ctx['gpu_recovery'],operational_source_commit=stage['operational_source_commit'],\n            gpu_recovery_acceptance_sha256=ctx['pins']['gpu-recovery-acceptance'],")
inherited=GPU.read_text()
inherited_funcs=function(inherited,'exact_inventory')+'\n'+function(inherited,'failed_gpu_history')
inherited_funcs=inherited_funcs.replace('==ENTRY_SHA','==ORIGINAL_GPU_ENTRY_SHA').replace("p['resources']==RESOURCES and p['limits']==LIMITS", "p['resources']==GPU_RESOURCES and p['limits']==GPU_LIMITS")
s=change(s,"FULL_MAP_METADATA =",(HERE/'builder-constants.txt').read_text()+"\n\nFULL_MAP_METADATA =")
s=change(s,"def metadata_maximum(name):", "FULL_MAP_METADATA |= frozenset(('gpu-failure-acceptance','gpu-failure-disposition',\n    'gpu-failure-inventory','gpu-failure-inventory-after','gpu-failure-seal','cpu-history-expected'))\n\ndef metadata_maximum(name):")
s=change(s,"def main():", inherited_funcs+'\n'+(HERE/'builder-gate.txt').read_text()+"\n\ndef main():")
emit('build_packet.py',s)

s=(OLD/'launch_once.py').read_text()
s=change(s,"packet['prior_jobs']==47 and packet['prior_journals']==96", "packet['prior_jobs']==48 and packet['prior_journals']==98")
s=change(s,"        claim_pin=ctx['pins']['namespace-claim']", "        b.need(packet['gpu_recovery']==ctx['gpu_recovery']\n               and packet['gpu_recovery_acceptance_sha256']==ctx['pins']['gpu-recovery-acceptance']\n               and packet['operational_source_commit']==m['finalizer-source-manifest']['source_commit'],\n               'exact GPU02/finalizer operational source lineage required')\n        claim_pin=ctx['pins']['namespace-claim']")
s=change(s,"gpu_job_id=m['gpu-acceptance']['job_id'],token_recovery=ctx['token_recovery'],", "gpu_job_id=m['gpu-acceptance']['job_id'],token_recovery=ctx['token_recovery'],\n            gpu_recovery=ctx['gpu_recovery'],gpu_recovery_acceptance_sha256=ctx['pins']['gpu-recovery-acceptance'],\n            operational_source_commit=m['finalizer-source-manifest']['source_commit'],")
emit('launch_once.py',s)

s=(OLD/'pull_terminal.py').read_text()
for old,new in [("baseline['terminal_jobs']==47 and baseline['submission_journals']==96 and len(prior)==47", "baseline['terminal_jobs']==48 and baseline['submission_journals']==98 and len(prior)==48"),
    ('exact47 predecessor rows plus actual48th','exact48 predecessor rows plus actual49th'),
    ("len(baseline['journal_pins'])==96 and len(expected)==98", "len(baseline['journal_pins'])==98 and len(expected)==100"),
    ('exact96 prior and98 final journal pins','exact98 prior and100 final journal pins'),
    ("packet['prior_jobs']==47 and packet['prior_journals']==96", "packet['prior_jobs']==48 and packet['prior_journals']==98"),
    ("('gpu' if label=='launch' else label)","('gpu' if label=='launch' else 'gpu-control' if label=='control' else label)")]:
    s=change(s,old,new)
s=change(s,"            need(a.job_id not in", "            need(packet['gpu_recovery']==intent['gpu_recovery']==ctx['gpu_recovery']\n                 and packet['gpu_recovery_acceptance_sha256']==intent['gpu_recovery_acceptance_sha256']==ctx['pins']['gpu-recovery-acceptance']\n                 and packet['operational_source_commit']==intent['operational_source_commit']==m['finalizer-source-manifest']['source_commit'],\n                 'full GPU02/finalizer operational lineage differs')\n            need(a.job_id not in")
emit('pull_terminal.py',s)
emit('monitor_once.py',(OLD/'monitor_once.py').read_text())
s=(OLD/'compact_inputs.py').read_text()
s=change(s,"name.startswith('launch/inputs/')", "name.startswith('control/inputs/')")
s=change(s,"        self.full=r['full_inventory'];self.selected=r['local_inventory']", "        need(expected_roots.get('control')=='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261109_02'\n             and expected_roots.get('launch')=='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108_01',\n             'explicit GPU02 control / original output roots required')\n        self.full=r['full_inventory'];self.selected=r['local_inventory']")
s=change(s,"sorted(observation['directories'])", "observed_directories(observation['directories'])")
s=change(s,"        need(after[key]==obs[key],'original remote before/after differs: '+key)",
    "        if key=='directories':\n            need(observed_directories(after[key])==observed_directories(obs[key]),'original remote directory sets differ')\n        else:need(after[key]==obs[key],'original remote before/after differs: '+key)")
helper="""def observed_directories(values):
    need(type(values) is list and all(type(n) is str for n in values)
         and len(set(values))==len(values),'original unique directory list required')
    for n in values:relative(n)
    return sorted(values)


"""
s=change(s,'class Inputs:',helper+'class Inputs:')
emit('compact_inputs.py',s)
adapter_sha=hashlib.sha256(s.encode()).hexdigest()
b=(NEW/'build_packet.py').read_text()
b=change(b,"COMPACT_ADAPTER_SHA='a0e601a7be0ec2144faf0956de003dbe19b1a0f538b22a36a3506d316b698245'",'COMPACT_ADAPTER_SHA='+repr(adapter_sha))
emit('build_packet.py',b)
print('Derived sources only; no producer imported or invoked.')
