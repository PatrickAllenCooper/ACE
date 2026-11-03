from pathlib import Path
import ast,copy,datetime,hashlib,json,shutil,subprocess
repo=Path('/Users/pat/code/ACE');base=Path(__file__).resolve().parent;dest=base/'input';dest.mkdir()
old=repo/'results/ace_llm_intervention_task_runtime_20261102/gpu05/input'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 with p.open('x') as f:f.write(json.dumps(x,indent=2,sort_keys=True)+'\n')
source=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
text=(repo/'scripts/research/qualify_llm_intervention_checkpoint_gpu.py').read_text();tree=ast.parse(text)
values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
for name in values['SOURCES']:
 raw=subprocess.check_output(['git','-C',str(repo),'show',source+':scripts/research/'+name]);assert raw==(repo/'scripts/research'/name).read_bytes();(dest/name).write_bytes(raw)
for name in values['ASSETS']:
 if name=='gpu06_terminal.json':p=repo/'results/ace_llm_intervention_task_runtime_20261102/gpu06/terminal/gpu06/worker/terminal.json'
 elif name=='gpu08_failure_disposition.json':p=repo/'results/ace_llm_intervention_task_runtime_20261102/gpu08/failure_disposition.json'
 elif name=='gpu07_failure_disposition.json':p=repo/'results/ace_llm_intervention_task_runtime_20261102/gpu07/failure_disposition.json'
 elif name=='gpu06_acceptance.json':p=repo/'results/ace_llm_intervention_task_runtime_20261102/gpu06/acceptance.json'
 else:p=old/name
 shutil.copy2(p,dest/name)
freeze=json.loads((old/'freeze.json').read_text());freeze.update(schema='ace-checkpoint-task-gpu-v1',source_revision=source,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),recipe=values['RECIPE'],sources={n:sha(dest/n) for n in values['SOURCES']},assets={n:sha(dest/n) for n in values['ASSETS']},startup_seconds=180,worker_seconds=540,conservative_cpu_reservation_seconds=4800,conservative_gpu_reservation_seconds=600,cycle_reserved_cpu_seconds=54000,cycle_reserved_gpu_seconds=5700,scope='new checkpointed canonical artificial qualification, no science or environment response',resource_basis='GPU08 canonical gate passed and4341token update16.566GB peak;7343token forward OOM on19.62GiB. Same smallest20GB slice with explicit decoder recomputation, actual savings unqualified, bounded failure preserved')
freeze['resources']['wall_seconds']=600;write(dest/'freeze.json',freeze)
shutil.copy2(repo/'docs/development/guidance/reviews/ace_llm_intervention_checkpoint_implementation_review_2026-11-02.md',dest/'implementation_review.md')
root='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_checkpoint_gpu_20261102_09'
prior=(repo/'results/ace_llm_intervention_task_runtime_20261102/gpu06/input/gpu06.sbatch').read_text()
prior=prior.replace('ace_llm_gdiag06','ace_llm_check09').replace('ace_llm_intervention_gradient_gpu_20261102_06','ace_llm_intervention_checkpoint_gpu_20261102_09').replace('00:05:00','00:10:00').replace('supervise_llm_intervention_gradients.py','supervise_llm_intervention_checkpoint_gpu.py').replace('0ff56c88449d9ae053b97bb83a6bc7799369294358d69f2700cfb70294daa6c3',sha(dest/'freeze.json')).replace('/gpu06','/gpu09')
(dest/'gpu09.sbatch').write_text(prior)
(dest/'source-pins.sha256').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in sorted(dest.iterdir())))
manifest={'files':{p.name:sha(p) for p in sorted(dest.iterdir())},'freeze_sha256':sha(dest/'freeze.json'),'remote_root':root,'source_revision':source};write(base/'input_manifest.json',manifest)
print(source,manifest['freeze_sha256'],len(manifest['files']),len(freeze['sources']),len(freeze['assets']))
