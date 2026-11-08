from pathlib import Path
import ast,shutil
out=Path('/tmp/ace-scientific-finalizer-token02-8wf1foat');g=Path('/private/tmp/ace-scientific-training-gpu-token02-mcu8h32f/collector');c=Path('/tmp/ace-scientific-cpu-campaign-token02-3syhqil0/pull_terminal.py')
gs=(g/'pull_terminal.py').read_text();cs=c.read_text()
def fun(source,name):return ast.get_source_segment(source,next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name))
r=next(ast.literal_eval(n.value) for n in ast.parse(cs).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE_OBSERVATION' for t in n.targets))
a=r.index('v=dict(');z=r.index('commands=',a)
r=r[:a]+'''v=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),results={},files={},directories=[],
 root_present=root.exists(),root_presence={label:Path(path).exists() for label,path in ROOTS.items()},
 namespace_present=os.path.lexists(namespace),original_namespace_present=os.path.lexists(original_namespace),
 original_hardlink_groups=[],upstream_pins={},upstream_errors={})
'''+r[z:]
a=r.index(' for label,tree,present in ');z=r.index('\n  if not present',a)
r=r[:a]+" for label,tree,present in ((label,Path(path),v['root_presence'][label]) for label,path in ROOTS.items()):"+r[z:]
r=r.replace('import os,', 'import os,')
r="import signal\ndef deadline(signum,frame):raise TimeoutError('bounded full terminal observation')\nsignal.signal(signal.SIGALRM,deadline);signal.alarm(300)\n"+r
header='''"""Once-only complete scientific finalizer custody. No private decode or acceptance.
Standalone coordinator, authenticated builder and cleared transport helper only.
"""
from pathlib import Path
import argparse,ast,datetime,fcntl,hashlib,json,os,re,shutil,stat,subprocess,time,types,shlex
REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
LAUNCH=REPO/'results/ace_llm_intervention_scientific_finalization_20261108/launch01'
LAUNCH_EXT=BASE/'ACE-LLM-intervention-scientific-finalization-20261108-01'
LOCAL=LAUNCH.parent/'terminal01'
EXT=BASE/'ACE-LLM-intervention-scientific-finalization-terminal-20261108-01'
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_finalization_20261108_01'
HELPER='/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
RESOURCES=dict(account='ucb736_asc2',partition='acpu',qos='cpu-normal',cpus=2,memory_mib=7680,wall_seconds=1200,gpus=0,requeue=False)
STREAM_HELPER_SHA='fab01ad840e4e73ef0fc2136fd2eeb12b1033256e80a8f96a392afb2d50b758a'
STREAM_REVIEW_SHA='5058426e9dccaaf210eb2969a8623a31123f13dfefaa43c57e975e5800819371'

'''
names=['need','sha','raw','utc','pin','decode','fingerprint','read_metadata','inventory','write','normalize','tres','gpu_count','cost','validate_remote_intent','load_stream_helper','mirror_attempt','stream_transfer','authenticate_stream_files']
text=header+'\n\n'.join(fun(gs,n) for n in names)+'\n\nREMOTE_OBSERVATION='+repr(r)+'\n\n'
text+=fun(cs,'cpu_tres')+'\n\n'
resources=fun(cs,'resources').replace("'100'","'20'").replace("'launch/cpu/allocation.json'","'launch/finalization/allocation.json'").replace('ace_llm_scientific_campaign_cpu01','ace_llm_scientific_finalization01').replace('<=6000','<=1200').replace('CPU campaign','CPU finalizer')
text+=resources+'\n\n'
(out/'collector').mkdir();(out/'collector/pull_terminal.py').write_text(text)
for n in ('stream_custody.py','stream_custody_review.md'):shutil.copyfile(g/n,out/'collector'/n)
shutil.copyfile('/tmp/derive-finalizer-collector.py',out/'derive-collector.py')
