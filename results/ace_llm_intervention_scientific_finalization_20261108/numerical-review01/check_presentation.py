"""Saved canvas/prose/accounting associations; no UI, producer or model execution."""
from pathlib import Path
import hashlib,json,math,re,resource,time,traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPO=Path('/Users/pat/code/ACE')
CANVAS=ROOT/'reporting01/ACE-24-system-comparison.canvas.tsx'
MANAGED=Path('/Users/pat/.cursor/projects/Users-pat-code-ACE/canvases/ACE-24-system-comparison.canvas.tsx')
DRAFT=REPO/'docs/development/guidance/ace_llm_intervention_24_system_results_2026-11-10.md'
report={'checks':0,'errors':[],'notes':[],'source_execution':False,'canvas_rendered':False,'tests_rerun':0,'remote_actions':0}
def sha(b):return hashlib.sha256(b).hexdigest()
def j(p):return json.loads(p.read_bytes())
def check(ok,label):
    report['checks']+=1
    if not ok:report['errors'].append(label)
def main():
    first=j(HERE/'cells-checks.json');second=j(HERE/'aggregate-checks.json');assert not first['errors'] and not second['errors']
    code=CANVAS.read_bytes();check(code==MANAGED.read_bytes(),'exact repository/managed canvas bytes')
    source=code.decode();start=source.index('const DATA = ')+len('const DATA = ')
    data,end=json.JSONDecoder().raw_decode(source[start:]);logic=source[start+end:]
    expected=j(ROOT/'reporting01/data01/display-data.json')
    expected['curves']=[{k:p[k] for k in ('world_id','policy_id','total_cost_units','normalized_mse','action_id')} for p in expected['curves']]
    check(data==expected,'all canvas data exact; only five-field curve projection')
    check(len(data['groups'])==10 and len(data['cells'])==192 and len(data['curves'])==4540,'canvas complete data counts')
    cells={(c['world_id'],c['policy_id']):c for c in first['cells']}
    for group in data['groups']:
        for pair in group['pairs']:
            for left,right in ((pair['left'],pair['right']),(pair['right'],pair['left'])):
                sign=1 if pair['left']==left else -1
                displayed=[v*sign for v in pair['differences']]
                independent=[cells[w,left]['capped_total_cost_units']-cells[w,right]['capped_total_cost_units'] for w in pair['world_ids']]
                check(displayed==independent,'canvas pair orientation '+group['id']+'/'+left+'/'+right)
                check(math.isclose(pair['mean_difference']*sign,math.fsum(independent)/len(independent),rel_tol=1e-12,abs_tol=1e-14),'canvas pair mean sign '+group['id']+'/'+left+'/'+right)
    required_literals=['useState("all")','useState(String(DATA.worlds[0].id))','const sign=pair.left===left?1:-1;',
      'const diff=pair.differences.map(v=>v*sign);','yMin={0} yMax={32}','yMin={-24} yMax={24}',
      'Object.keys(ACTIONS).map(a=>','?.accepted_acquisitions??0','across all 24 systems',
      'const x=(v:number)=>70+(v-8)/24*650;','Math.max(.05,...sets.flatMap(s=>s.map(p=>p.normalized_mse)))*1.08',
      'const y=(v:number)=>260-v/maximum*220;','Points are not interpolated','Axes rescale to include every displayed error.',
      'if(v===right)setRight(left)','if(v===left)setLeft(right)']
    for phrase in required_literals:check(phrase in logic,'static canvas representation '+phrase)
    check('<polyline' not in logic and '<path' not in logic and '<circle' in logic and '<rect' in logic,'individual saved curve markers without connecting paths')
    check(all(k+':' in logic for k in ('observe_all','set_x_neg','set_x_zero','set_x_pos','set_y_neg','set_y_zero','set_y_pos','set_xy')),'all8 action labels including zero-count alternatives')
    draft=DRAFT.read_bytes();text=draft.decode()
    rows=[line.split('|')[1:-1] for line in text.splitlines() if line.startswith('| ') and not line.startswith('| ---')][1:]
    check(len(rows)==8,'writeup exactly8 policy table rows')
    for row,p in zip(rows,second['policy_table']):
        vals=[v.strip() for v in row]
        check(vals[1:]==[format(p['mean_capped_total_cost'],'.3f'),str(p['ever_attained']),str(p['final_attained']),format(p['mean_final_error'],'.5f'),format(p['actual_total_cost']/p['world_count'],'.3f')],'writeup formatted values '+p['policy_id'])
    if 'required 15.167 units' in text:report['notes'].append('Replace required with had a mean capped total cost of; capped nonattainment is not attainment.')
    if '85 GB local download requirement' in text:report['notes'].append('Replace with former85GiB Mac free-space planning target; retained source is a capacity estimate, not download volume.')
    ret=ROOT/'report-retrieval01';hand=j(ret/'handoff.json')
    receipt_raw=(ret/'inputs/remote-custody.json').read_bytes();check(sha(receipt_raw)==hand['original_backend_receipt_sha256'],'retained archive receipt pin')
    receipt=json.loads(receipt_raw);selected=receipt['local_inventory']['files']
    claims={}
    for n in ('training-acceptance.json','collection-acceptance.json'):
        p=ROOT/'terminal01/remote/launch'/n;b=p.read_bytes();check(sha(b)==selected['launch/'+n]['sha256'],'upstream accepted claim bytes '+n);x=json.loads(b)
        if n.startswith('training'):
            claims['training']={k:x[k] for k in ('status','scope','required_findings','sft_steps','dpo_steps','completed_steps','collection_acceptance_sha256')}
            check((x['sft_steps'],x['dpo_steps'],x['completed_steps'])==(160,119,279),'actual training update counts')
        else:
            claims['collection']={k:x[k] for k in ('status','scope','required_findings','records','retained_pairs','collection_accounting')}
            check(x['records']==160 and x['retained_pairs']==3584 and x['collection_accounting']==dict(executor_accepted_responses=9145,known_returns=9145,reserved_response_attempts=9145,total_reserved_cost_units=10240,unknown_return_attempts=0),'actual collection counts/full charging')
        claims[n]={'sha256':sha(b),'bytes':len(b),'path':str(p)}
    check(len(selected)==379 and sum(v['bytes'] for v in selected.values())==306334884,'selected finalizer custody size')
    check(hand['raw_archive_file_count']==107284 and hand['raw_archive_bytes']==3308319856 and hand['archive']['bytes']==3390535680 and hand['member_count']==242 and hand['member_bytes']==4403819,'storage/report byte counts')
    base_raw=(ret/'inputs/baseline.json').read_bytes();check(sha(base_raw)==hand['inputs']['baseline.json']['sha256'],'accepted accounting baseline pin')
    base=json.loads(base_raw)
    check((base['terminal_jobs'],base['submission_journals'],base['closed_cpu_seconds'],base['closed_gpu_slice_seconds'],base['outstanding_reservation'],base['preparation_allowance_cpu_seconds'])==(49,100,40265,4458,0,3600),'actual closed accounting counts')
    row=next(r for r in base['related_accounting'] if r[0]=='35533604')
    check(row[2:7]==['COMPLETED','0:0','2','1596','ucb736_asc2'] and int(row[4])*int(row[5])==3192,'actual finalizer charge')
    check(round(40265/3600,4)==11.1847 and round(4458/3600,4)==1.2383 and 40265+3600==43865,'hours and single allowance arithmetic')
    collect=ROOT/'monitor-collection-control01/collector01/run.json';cost=j(collect)
    check(cost['returncode']==0 and cost['child_cpu_seconds']==97.013593 and cost['elapsed_seconds']==2765.272798125,'separate observed collector costs')
    claims['accounting']={k:base[k] for k in ('terminal_jobs','submission_journals','closed_cpu_seconds','closed_gpu_slice_seconds','outstanding_reservation','preparation_allowance_cpu_seconds')}
    claims['collector_run']={'sha256':sha(collect.read_bytes()),'path':str(collect),**cost}
    claims['storage']={k:hand[k] for k in ('archive','raw_archive_file_count','raw_archive_bytes','member_count','member_bytes')}
    report.update(canvas_sha256=sha(code),draft_sha256=sha(draft),canvas_paths=[str(CANVAS),str(MANAGED)],draft_path=str(DRAFT),supporting_claims=claims,
      limitations=['Static canvas data/logic inspection only; no pixel/interaction/runtime rendering qualification.',
      'Reported errors are trusted saved evaluator outputs; no private world re-evaluation.',
      'The report scopes this observed recipe and development set; no universal efficacy or real-system transfer claim.'])
    for name,b in [('canvas.tsx',code),('results-draft.md',draft),('canvas-logic.txt',logic.encode())]:
        with (HERE/name).open('xb') as f:f.write(b)

start=time.monotonic();cpu=time.process_time()
try:main()
except BaseException as e:
    report['errors'].append({'exception':type(e).__name__,'message':str(e)})
    (HERE/'presentation-failure.traceback.txt').write_text(traceback.format_exc())
finally:
    report.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=sha(Path(__file__).read_bytes()))
    with (HERE/'presentation-checks.json').open('x') as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:report.get(k) for k in ('checks','errors','notes','canvas_sha256','draft_sha256')}))
raise SystemExit(bool(report['errors']))
