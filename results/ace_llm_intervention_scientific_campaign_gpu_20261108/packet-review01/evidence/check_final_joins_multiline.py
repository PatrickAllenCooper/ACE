from pathlib import Path
import ast,hashlib,json,shlex
W=Path(__file__).parent;R=Path('/Users/pat/code/ACE');P=R/'results/ace_llm_intervention_scientific_campaign_gpu_20261108/launch01';ST=P.parent
OPS=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-campaign-gpu-20261108-01-storage-build-preparation/records')
def j(p):return json.loads(p.read_bytes())
def h(b):return hashlib.sha256(b).hexdigest()
def ck(n,b):checks.append({'check':n,'passed':bool(b)})
checks=[];facts={};receipt=j(P/'storage-receipt.json');inner=j(P/'inputs/packet.json');base=j(P/'cpu-baseline.json');outer=j(P/'packet.json')
for n in ['verify-000','verify-001','verify-002']:
 q=OPS/'operations'/n;v=j(q/'receipt.json');t=j(q/'transport.json')
 ck(n+' complete archive readback equals accepted full map',v['full_inventory']==receipt['full_inventory'] and v['full_backup_verified'] is True and v['backup_archive_sha256']==receipt['backup_archive_sha256'] and v['receipt_sha256']==h((P/'storage-receipt.json').read_bytes()) and v['helper_sha256']==receipt['helper_sha256'] and v['acceptance'] is False)
 ck(n+' isolated pinned source and successful unambiguous transport',h((q/'helper.py').read_bytes())==receipt['helper_sha256'] and t['returncode']==0 and t['remote_outcome_ambiguous'] is t['retry_allowed'] is False and 'python3 -I -B' in t['command'][-1] and receipt['helper_sha256'] in t['command'][-1])
 facts[n]={'receipt_sha256':h((q/'receipt.json').read_bytes()),'transport_sha256':h((q/'transport.json').read_bytes()),'elapsed_seconds':t['wall_seconds'],'full_files':len(v['full_inventory']['files']),'full_directories':len(v['full_inventory']['directories'])}
for n in ['fetch-log-028','fetch-log-091']:
 q=OPS/'operations'/n;v=j(q/'receipt.json');t=j(q/'transport.json');name=Path(v['member']).name
 ck(n+' opaque requested member authenticated without decode',v['verified'] is True and v['acceptance'] is False and v['backup_archive_sha256']==receipt['backup_archive_sha256'] and v['helper_sha256']==receipt['helper_sha256'] and v['sha256']==inner['files'][name]==receipt['full_inventory']['files'][v['member']]['sha256'] and v['bytes']==receipt['full_inventory']['files'][v['member']]['bytes'] and t['returncode']==0 and t['remote_outcome_ambiguous'] is False)
 facts[n]={k:v[k] for k in ['member','sha256','bytes','verified']}
cmd=j(ST/'build-command01/command.json');mismatches=[]
for key,v in cmd['metadata'].items():
 candidates=[P/(key+ext) for ext in ('.json','.md') if (P/(key+ext)).exists()]
 if len(candidates)!=1 or h(candidates[0].read_bytes())!=v['sha256']:mismatches.append(key)
ck('all56 explicit metadata argument pins match captured packet roles',not mismatches and len(cmd['metadata'])==56)
readiness=j(P/'site-readiness.md') if (P/'site-readiness.md').read_bytes().startswith(b'{') else json.loads((P/'site-readiness.md').read_text().split('```json\n',1)[1].split('\n```',1)[0])
commands=readiness['commands'];facts['readiness_utc']=readiness['utc'];facts['readiness_command_names']=list(commands)
ck('saved readiness commands completed and scratch probe succeeded',all(v['returncode']==0 for v in commands.values()) and readiness['scratch_write_read_cleanup'] is True)
def normalized(rows):
 return sorted(tuple(r[:7]+[','.join(sorted(r[7].split(',')))]) for r in rows)
rows=[s.split('|')[:8] for s in commands['accounting']['stdout'].splitlines() if '|ace_llm' in s]
ck('saved readiness exact46 accepted rows retained',normalized(rows)==normalized(base['related_accounting']))
facts['readiness_small_outputs']={k:v['stdout'] for k,v in commands.items() if k not in ('accounting',) and len(v['stdout'])<3000}
shell=(P/'job.sbatch').read_text();tokens=shlex.split(shell.split('\nexec ',1)[1]);bootstrap=tokens[4]
freeze=j(P/'freeze.json');root=outer['remote_root'];entry='qualify_llm_intervention_scientific_campaign_gpu.py'
expected=[freeze['runtime_source']+'/bin/python','-I','-B','-c',bootstrap,root+'/inputs/'+entry,inner['files'][entry],'--packet',root+'/inputs/packet.json','--packet-sha256',h((P/'inputs/packet.json').read_bytes()),'--admission',root+'/inputs/admission.json','--admission-sha256',h((P/'inputs/admission.json').read_bytes()),'--output',root+'/gpu']
ck('exact sbatch isolated runtime source pin and argument joins',tokens==expected and 'hashlib.sha256(data).hexdigest()!=expected' in bootstrap and 'raise ValueError' in bootstrap and len(bootstrap)<4096)
ck('exact sbatch resources and no requeue',all(x in shell.splitlines() for x in ['#SBATCH --account=ucb736_asc2','#SBATCH --partition=aa100','#SBATCH --qos=gpu-testing','#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=4','#SBATCH --mem=15360M','#SBATCH --gres=gpu:a100_3g.20gb:1','#SBATCH --time=00:48:00','#SBATCH --no-requeue']))
# Authenticate reused review pin map without executing a reviewed source.
reviewdir=R/'results/ace_llm_intervention_remote_custody_20261109/downstream-fullmap128-independent-review01';pp=j(reviewdir/'source-pins.json')['source_pins'];facts['prior_source_pin_keys']=list(pp)
ck('captured GPU coordinators in independently reviewed pin map',all(pp['gpu/'+n]==h((P/n).read_bytes()) for n in ['build_packet.py','launch_once.py','monitor_once.py','cpu_history_contract.py','compact_inputs.py']))
ck('no submission or build failure records present at review',not any((P/n).exists() for n in ['controller.json','submission.intent.json','submission.result.json','launcher_failure.json','build-failure.json']))
# Compress the accidentally large reviewer fact capture without changing original bytes.
old=j(W/'checks02.json');summary={**old,'facts':{k:v for k,v in old['facts'].items() if not k.startswith('verify-')}}
summary['original_complete_capture']={'path':str(W/'checks02.json'),'bytes':(W/'checks02.json').stat().st_size,'sha256':h((W/'checks02.json').read_bytes()),'note':'Redundant full inventories are preserved in original reviewer capture and pinned accepted storage receipts; omitted from compact summary.'}
(W/'checks02-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
result={'checks':checks,'failed_checks':[x for x in checks if not x['passed']],'facts':facts,'metadata_only':True,'remote_actions':0,'scientific_execution':False}
(W/'checks03.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'checks':len(checks),'failed_checks':result['failed_checks'],'readiness':facts['readiness_small_outputs']},indent=2))
