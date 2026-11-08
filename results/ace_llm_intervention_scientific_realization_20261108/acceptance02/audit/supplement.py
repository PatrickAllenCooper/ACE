"""Additional saved metadata checks only, after the retained main audit passed."""
from pathlib import Path
import ast, hashlib, json, math, subprocess
E=Path(__file__).parent
R=Path('/Users/pat/code/ACE')
T=R/'results/ace_llm_intervention_scientific_realization_20261108/terminal02'
L=T/'remote/launch'; O=L/'realization'; W=O/'session/worker'
def read(p):return json.loads(p.read_bytes())
def sha(b):return hashlib.sha256(b).hexdigest()
def need(x,c):
    if not x:raise ValueError(c)
report=read(E/'attempt02/stdout')
need(report['required_findings']==0,'prior_audit')
review='docs/development/guidance/reviews/ace_llm_intervention_scientific_realization02_packet_review_2026-11-08.md'
b=subprocess.check_output(['git','--no-replace-objects','show',report['packet_commit']+':'+review],cwd=R)
intent=read(L/'submission.intent.json')
need(sha(b)==intent['review_sha256']=='24530af2f3ecca35d651b6b909a0a86fd4ec66cb76bb4f506c55c5cfd4ab3773','packet_review_commit')
for label in ('local-launch','external-launch'):
    c=T/'coordinator'/label
    need(not (c/'launcher_failure.json').exists(),'launcher_failure')
    held=read(c/'controller.json');released=read(c/'controller_released.json')
    need(released['released'] is True and released['pid']==held['pid'],'released_locks')
    transport=read(c/'submission_transport.json')
    need(transport['returncode']==0 and json.loads(transport['stdout'])==read(L/'submission.result.json'),'saved_sbatch_transport')
phases=[read(p) for p in sorted(O.glob('phase-*.json'))]
need(len(phases)==102 and [p.name for p in sorted(O.glob('phase-*.json'))]==[f'phase-{i:04d}.json' for i in range(102)],'phase_roster')
need(phases[0]['phase']=='stage_start' and phases[-1]['phase']=='stage_complete','stage_bookends')
need(all(x['phase']=='stage_progress' and x['detail']['phase']=='stage' and x['detail']['objects']==49966 for x in phases[1:-1]),'stage_progress')
need([x['detail']['completed_objects'] for x in phases[1:-1]]==list(range(1,49966,500)),'stage_progress_order')
need(all(a['elapsed_seconds']<=b['elapsed_seconds'] and a['process_cpu_seconds']<=b['process_cpu_seconds'] and a['utc']<=b['utc'] for a,b in zip(phases,phases[1:])),'monotonic_phase_progress')
need(all(0<=p['elapsed_seconds']<120 and 0<p['peak_self_rss_bytes']<=512*1024**2 for p in phases),'bounded_staging')
handoff=read(O/'session/handoff.json');marker=read(W/'work-start.json')
need(phases[-1]['elapsed_seconds']<marker['monotonic']-handoff['started_monotonic']<120,'stage_before_draw')
# Read literal-only provenance constants from authenticated source; no imports.
env={}
def literal(n):
    if isinstance(n,ast.Name):return env[n.id]
    if isinstance(n,ast.Dict):return {literal(k):literal(v) for k,v in zip(n.keys,n.values)}
    if isinstance(n,(ast.Tuple,ast.List)):return [literal(v) for v in n.elts]
    return ast.literal_eval(n)
for n in ast.parse((L/'inputs/qualify_llm_intervention_runtime_metadata.py').read_bytes()).body:
    if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
        try:env[n.targets[0].id]=literal(n.value)
        except (ValueError,KeyError,TypeError):pass
stage=read(O/'stage.json');prefix=stage['staged_prefix'];provenance=read(O/'independent-provenance-sources.json')
need(set(provenance['sources'])==set(env['PROVENANCE_SOURCES']),'provenance_sources')
for name,(relative,pin) in env['PROVENANCE_SOURCES'].items():
    row=provenance['sources'][name]
    need(row['sha256']==pin and row['path']==prefix+'/lib/python3.10/site-packages/'+relative,'provenance_identity')
need(sha(provenance['expected_source'].encode())==provenance['expected_source_sha256'],'provenance_source_bytes')
need(set(stage['cache_source_sha256'])=={v for pair in env['CACHE_PAIRS'] for v in pair},'cache_pair_roster')
need(all(len(v)==64 and all(c in '0123456789abcdef' for c in v) for v in stage['cache_source_sha256'].values()),'cache_pair_hashes')
task=W/'task';proposal=read(L/'inputs/proposal.json');taskintent=read(task/'intent.json');terminal=read(task/'terminal.json')
need(len(taskintent['entries'])==56 and taskintent['started_utc']<=terminal['completed_utc'],'task_time_roster')
for i,row in enumerate(proposal['training_worlds']+proposal['development_worlds']):
    value=read(task/f'world-{i:03d}.json')
    need([m['noise_variance'] for m in value['mechanisms']]==row['truth_law']['noise_variances'],'fixed_noise_law')
    need(all(type(c) is list and len(c)==2 for m in value['mechanisms'] for c in m['coefficients']),'coefficient_shape')
obs=json.loads(read(T/'inventory_transport.json')['stdout'])
steps=[l.split('|') for l in obs['results']['steps']['stdout'].splitlines() if l.strip()]
need(all(r[2]=='COMPLETED' and r[3]=='0:0' for r in steps),'all_step_success')
batch=next(r for r in steps if r[0]==report['job_id']+'.batch')
need(batch[8]=='7862112K','actual_maxrss')
result={'status':'supplement_passed','required_findings':0,'packet_review_sha256':sha(b),'stage_phases':len(phases),'stage_start_elapsed_seconds':phases[0]['elapsed_seconds'],'stage_complete_elapsed_seconds':phases[-1]['elapsed_seconds'],'stage_process_cpu_seconds':phases[-1]['process_cpu_seconds'],'stage_peak_self_rss_bytes':max(p['peak_self_rss_bytes'] for p in phases),'work_start_elapsed_seconds':marker['monotonic']-handoff['started_monotonic'],'task_started_utc':taskintent['started_utc'],'task_completed_utc':terminal['completed_utc'],'scheduler_maxrss_kib':7862112,'scheduler_maxrss_mib':7862112/1024,'memory_headroom_claimed':False,'review_path':review,'scientific_qualified':False}
print(json.dumps(result,sort_keys=True,indent=2))
