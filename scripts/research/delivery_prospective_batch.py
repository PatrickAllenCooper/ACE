"""Frozen CPU-only prospective study with collection/fit/evaluation barriers.

Preparation and tests expose no selected-system responses. Freeze requires a
completed accepted target-runtime pilot and attribution gate. Slurm submission
is a separate operation; this worker never creates allocations or calls APIs.
"""
import argparse
import math
import os
from pathlib import Path
import shutil
import sys
import time

from runner_delivery_confirmation import read,write,sha,utc,journal_count,no_network,supervise
from delivery_prospective_design import validate_world,world_spec,seed_for,action_block,reserved
from delivery_prospective_models import dependencies
from delivery_prospective_io import collect_history,fit_cell,sealed_fits,evaluate_states

HISTORIES={'balanced_varied_value':'balanced','matched_random':'random'}
CELLS=[('delivery',0),('online',0),('simpler',0),('ablation',0),
       ('delivery',1),('simpler',1),('delivery',2),('simpler',2)]
WORKERS=('delivery_prospective_batch.py','delivery_prospective_models.py','delivery_prospective_io.py',
         'delivery_prospective_design.py','delivery_prospective_analysis.py','delivery_theory.py',
         'runner_delivery_confirmation.py','audit_delivery_prospective_pilot.py',
         'delivery_prospective_slurm.py','test_delivery_prospective_models.py',
         'test_delivery_prospective_io.py','test_delivery_prospective_batch.py',
         'test_delivery_theory.py','test_delivery_prospective_analysis.py')


def expected_world_ids():
    return [f'{size}:{i:02d}' for size in (5,30) for i in range(20)]


def validate_descriptors(manifest,descriptors):
    """All mechanisms and action blocks fixed; never evaluate a response here."""
    if set(manifest['worlds'])!=set(expected_world_ids()) or manifest['structural_responses_evaluated']!=0:
        raise ValueError('forty preregistered worlds required; no seed replacement')
    seeds=[]
    for case in expected_world_ids():
        descriptor=manifest['worlds'][case];spec,actions=descriptors[case]
        size=int(case.split(':')[0]);seed=seed_for('world:'+case);seeds.append(seed)
        roots=validate_world(spec)
        if spec['size']!=size or spec['seed']!=seed or descriptor['seed']!=seed or descriptor['roots']!=roots:
            raise ValueError('frozen system identity/graph changed')
        if set(actions)!={'balanced','random','evaluation'}:
            raise ValueError('history/evaluation menus incomplete')
        blocks={}
        for strategy,menu in actions.items():
            if len(menu)!=400 or any(set(a)!=set(roots) for a in menu):
                raise ValueError('exact400root-action menu required')
            blocks[strategy]={action_block([a[n] for n in roots]) for a in menu}
            if any(reserved(b)!=(strategy=='evaluation') for b in blocks[strategy]):
                raise ValueError('reserved action partition changed')
            if descriptor['rows_by_strategy'][strategy]!=len(menu) or descriptor['blocks_by_strategy'][strategy]!=len(blocks[strategy]):
                raise ValueError('frozen action accounting changed')
        if blocks['evaluation']&(blocks['balanced']|blocks['random']):
            raise ValueError('heldout joint block leaked')
    if len(set(seeds))!=40:raise ValueError('seed collision; no replacement')


def allocation_plan(projection):
    """Requested CPU time, including rounding/guards, must also fit the cap."""
    wall={}
    for size in ('5','30'):
        per_world=projection['strata'][size]['raw_full_fit_cpu_seconds']/20
        wall[size]=max(900,math.ceil((3*per_world+16*5+60)/900)*900)
        if wall[size]>86400:raise ValueError('world wall request exceeds cpu-normal limit')
    total=(20*sum(wall.values())+300+900+3600+900)/3600
    if total>150 or projection['estimated_full_cpu_core_hours']>150:
        raise ValueError('projection or requested allocation exceeds150core-hours')
    return {'world_wall_seconds':wall,'qualification_wall_seconds':300,
        'collection_wall_seconds':900,'evaluation_wall_seconds':3600,
        'pilot_reserved_seconds':900,'total_requested_cpu_core_hours':total,'threads':1,'rss_bytes':3*2**30,
        'max_simultaneous_worlds':4,'account':'ucb736_asc1','partition':'acpu','qos':'cpu-normal'}


def load_descriptors(root,manifest):
    descriptors={}
    for case,d in manifest['worlds'].items():
        directory=Path(root)/case.replace(':','-')
        for file,key in (('world.json','world_sha256'),('actions.json','actions_sha256')):
            if sha(directory/file)!=d[key]:raise ValueError('world/action artifact changed')
        descriptors[case]=(read(directory/'world.json'),read(directory/'actions.json'))
    validate_descriptors(manifest,descriptors)
    return descriptors


def freeze(draft,project,source,gate,pilot,out,source_revision):
    from audit_delivery_prospective_pilot import audit
    draft,project,source,pilot,out=map(lambda p:Path(p).resolve(),(draft,project,source,pilot,out))
    g=read(gate)
    if (not g.get('custody_audit',{}).get('full_acceptance') or g['selected_delivery']!='scm' or
            g['strongest_simpler']!='flat' or g['decisive_ablation']!='optimization_ablation'):
        raise ValueError('complete frozen Stage A recipe gate required')
    if not isinstance(source_revision,str) or len(source_revision)!=40 or any(c not in '0123456789abcdef' for c in source_revision):
        raise ValueError('full committed source revision required')
    commit_receipt=read(project/'source_commit_receipt.json')
    if commit_receipt['source_revision']!=source_revision:
        raise ValueError('source revision does not match verified Git bundle')
    required={'scripts/research/'+name for name in WORKERS}|{'baselines.py','experiments/large_scale_scm.py'}
    if not required<=set(commit_receipt['files']):raise ValueError('committed worker coverage incomplete')
    for name,h in commit_receipt['files'].items():
        if sha(project/name)!=h:raise ValueError('verified committed file changed')
    accepted=audit(pilot,source,pilot/'acceptance.json');projection=read(pilot/'projection.json')
    if dependencies()!=accepted['dependencies']:raise ValueError('target runtime differs from measured pilot')
    for file in ('delivery_prospective_models.py','delivery_prospective_design.py','runner_delivery_confirmation.py'):
        if sha(Path(__file__).with_name(file))!=accepted['worker_hashes'][file]:
            raise ValueError('timed kernel/generator/guard changed')
    manifest=read(draft/'manifest.json');descriptors=load_descriptors(draft,manifest)
    for file,h in manifest['source_hashes'].items():
        if sha(project/file)!=h or accepted['generator_hashes'][file]!=h:
            raise ValueError('audited generator source changed')
    # Outcome-independent generator qualification for every descriptor, not
    # selected outcomes. Ignore only machine-specific provenance path strings.
    for case,(spec,_) in descriptors.items():
        regenerated=world_spec(spec['size'],spec['seed'],project)
        if {k:v for k,v in spec.items() if k!='source'}!={k:v for k,v in regenerated.items() if k!='source'}:
            raise ValueError('generator descriptor does not reproduce: '+case)
    resources=allocation_plan(projection)
    out.mkdir(parents=True,exist_ok=False)
    for case in expected_world_ids():
        directory=out/'descriptors'/case.replace(':','-');directory.mkdir(parents=True)
        for file in ('world.json','actions.json'):
            shutil.copyfile(draft/case.replace(':','-')/file,directory/file)
    shutil.copyfile(draft/'manifest.json',out/'descriptor_manifest.json')
    shutil.copyfile(gate,out/'attribution_gate.json')
    shutil.copyfile(pilot/'acceptance.json',out/'pilot_acceptance.json')
    shutil.copyfile(pilot/'projection.json',out/'pilot_projection.json')
    write(out/'registration.json',{'at':utc(),'stage':'B frozen prospective protocol before collection',
        'source_revision':source_revision,'source':str(source),'project':str(project),'output':str(out),'worlds':expected_world_ids(),
        'worker_hashes':{n:sha(Path(__file__).with_name(n)) for n in WORKERS},'source_hashes':accepted['source_hashes'],
        'generator_hashes':accepted['generator_hashes'],
        'source_commit_receipt_sha256':sha(project/'source_commit_receipt.json'),
        'dependencies':accepted['dependencies'],'descriptor_manifest_sha256':sha(out/'descriptor_manifest.json'),
        'gate_sha256':sha(out/'attribution_gate.json'),'pilot_acceptance_sha256':sha(out/'pilot_acceptance.json'),
        'pilot_projection_sha256':sha(out/'pilot_projection.json'),'histories':HISTORIES,'cells':CELLS,
        'matrix_fits':640,'primary_cells':240,'resources':resources,'new_response_ceiling':48000,
        'training_responses':32000,'shared_evaluation_responses':16000,
        'estimand':'noise-disabled deterministic structural target for every arm; not stochastic interventional expectation',
        'primary_targets':{'5':'X3','30':'X30'},'metric':'continuous heldout MSE/training target variance',
        'calibration':'first50paid training rows available before retrospective online updates; same SCM scaling',
        'init':'fixed0; init1/2 delivery/flat sensitivity separately, never deployment selection',
        'contrasts':'two controls per stratum;20systems/stratum; average two history log ratios within each world; four-test Holm',
        'superiority':{'ratio_at_most':.8,'upper_marginal95_interval_below':1,'holm_p_below':.05},
        'masking':'only joint roots intervened; every nonroot mechanism row eligible',
        'scope':'delivery recipe against root-target flat refit; supervision and compute differ, not pure architecture attribution',
        'stop':'retain every failure; stop new fits and block evaluation; no replacement worlds, histories or scored initializations',
        'initial_response_count':0,'submission':'separate validated Slurm submission; no allocations from worker'})


def validate(out):
    out=Path(out);p=read(out/'registration.json')
    if (p['worlds']!=expected_world_ids() or p['histories']!=HISTORIES or p['cells']!=[list(c) for c in CELLS] or
            p['matrix_fits']!=640 or p['new_response_ceiling']!=48000 or p['resources']['total_requested_cpu_core_hours']>150):
        raise ValueError('protocol matrix/resource drift')
    for file,key in (('descriptor_manifest.json','descriptor_manifest_sha256'),('attribution_gate.json','gate_sha256'),
                     ('pilot_acceptance.json','pilot_acceptance_sha256'),('pilot_projection.json','pilot_projection_sha256')):
        if sha(out/file)!=p[key]:raise ValueError('protocol receipt changed')
    for name,h in p['worker_hashes'].items():
        if sha(Path(__file__).with_name(name))!=h:raise ValueError('frozen worker changed')
    for name,h in p['source_hashes'].items():
        if sha(Path(p['source'])/name)!=h:raise ValueError('original learner source changed')
    for name,h in p['generator_hashes'].items():
        if sha(Path(p['project'])/name)!=h:raise ValueError('generator source changed')
    receipt=Path(p['project'])/'source_commit_receipt.json'
    if sha(receipt)!=p['source_commit_receipt_sha256'] or read(receipt)['source_revision']!=p['source_revision']:
        raise ValueError('source commit custody changed')
    if dependencies()!=p['dependencies']:raise ValueError('dependency drift')
    return p


def claim(path,value):
    import json
    with Path(path).open('x') as f:json.dump(value,f)


def intended_cells(out,p,input_hashes):
    out=Path(out);cells=[]
    for case in p['worlds']:
        for history in HISTORIES:
            key=case+'/'+history
            for arm,init in CELLS:
                cells.append({'index':len(cells),'case':case,'graph_size':int(case.split(':')[0]),
                    'history':history,'arm':arm,'init':init,
                    'input_dir':str(out/'training'/case.replace(':','-')/history),
                    'out':str(out/'fits'/case.replace(':','-')/history/(arm+'-i'+str(init))),
                    'binding':{'protocol_sha256':sha(out/'registration.json'),'input_sha256':input_hashes[key],
                        'kernel_sha256':p['worker_hashes']['delivery_prospective_models.py'],
                        'source_hashes':p['source_hashes'],'dependencies':p['dependencies']}})
    return cells


def collect(out):
    out=Path(out);p=validate(out);sys.addaudithook(no_network)
    qualification=read(out/'qualification_complete.json')
    if (qualification['registration_sha256']!=sha(out/'registration.json') or
            qualification['dependencies']!=p['dependencies'] or
            qualification['execution_sha256']!=sha(out/'qualification_execution.json') or
            qualification['log_sha256']!=sha(out/'qualification.log')):
        raise ValueError('target-runtime qualification changed or missing')
    descriptors=load_descriptors(out/'descriptors',read(out/'descriptor_manifest.json'))
    claim(out/'collection_started.json',{'at':utc(),'registration_sha256':sha(out/'registration.json')})
    inputs={};artifacts={};charged=0
    for case in p['worlds']:
        spec,actions=descriptors[case]
        for history,strategy in HISTORIES.items():
            destination=out/'training'/case.replace(':','-')/history
            data=collect_history(spec,actions[strategy],destination)
            training_variance(data)
            key=case+'/'+history;inputs[key]=sha(destination/'input.json')
            artifacts[key]={file:sha(destination/file) for file in ('input.json','queries.ndjson','receipt.json')}
            charged+=journal_count(destination/'queries.ndjson')
    if charged!=32000:raise ValueError('training response count changed')
    matrix={'registration_sha256':sha(out/'registration.json'),'cells':intended_cells(out,p,inputs)}
    write(out/'matrix.json',matrix)
    write(out/'collection_complete.json',{'at':utc(),'charged_responses':charged,'artifacts':artifacts,
        'registration_sha256':sha(out/'registration.json'),'matrix_sha256':sha(out/'matrix.json')})


def training_variance(data):
    import numpy as np
    values=np.array([r['node_values'][data['target']] for r in data['rows']])
    if not np.isfinite(values).all():
        raise ValueError('nonfinite training normalizer; retain world without replacement')
    variance=float(values.var())
    if not math.isfinite(variance) or variance<=0:
        raise ValueError('invalid training normalizer; retain world without replacement')
    return variance


def validate_collection(out,p):
    out=Path(out);done=read(out/'collection_complete.json');matrix=read(out/'matrix.json')
    if (done['registration_sha256']!=sha(out/'registration.json') or done['matrix_sha256']!=sha(out/'matrix.json') or
            matrix['registration_sha256']!=sha(out/'registration.json') or done['charged_responses']!=32000):
        raise ValueError('collection/matrix seal changed')
    inputs={}
    if set(done['artifacts'])!={case+'/'+h for case in p['worlds'] for h in HISTORIES}:
        raise ValueError('shared paid histories incomplete')
    for key,files in done['artifacts'].items():
        case,history=key.split('/');directory=out/'training'/case.replace(':','-')/history
        if set(files)!={'input.json','queries.ndjson','receipt.json'}:raise ValueError('history seal incomplete')
        for file,h in files.items():
            if sha(directory/file)!=h:raise ValueError('training artifact changed')
        if journal_count(directory/'queries.ndjson')!=400:raise ValueError('charged history count changed')
        inputs[key]=files['input.json']
    return validate_matrix(matrix['cells'],out,p,inputs)


def validate_matrix(cells,out,p,inputs):
    if cells!=intended_cells(out,p,inputs):raise ValueError('full640cell membership/binding changed')
    return cells


def run_cell(out,index):
    out=Path(out);p=validate(out)
    # The fit child validates membership but never opens coefficient/test files.
    cells=read(out/'matrix.json')['cells']
    done=read(out/'collection_complete.json')
    if (done['matrix_sha256']!=sha(out/'matrix.json') or done['registration_sha256']!=sha(out/'registration.json') or
            done['charged_responses']!=32000):raise ValueError('collection/matrix binding changed')
    if type(index) is not int or not 0<=index<640 or len(cells)!=640:raise ValueError('unregistered fit index')
    inputs={key:files['input.json'] for key,files in done['artifacts'].items()}
    validate_matrix(cells,out,p,inputs)
    cell=cells[index]
    if cell['index']!=index:raise ValueError('cell order changed')
    # Full response custody is validated by its parent before starting this child.
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    fit_cell(cell['input_dir'],p['source'],cell['out'],cell['arm'],cell['init'],cell['binding'])


def fit_world(out,case):
    out=Path(out);p=validate(out);cells=validate_collection(out,p)
    if case not in p['worlds']:raise ValueError('unregistered world')
    directory=out/'world_execution'/case.replace(':','-');directory.mkdir(parents=True,exist_ok=False)
    claim(directory/'started.json',{'at':utc(),'job_id':os.environ.get('SLURM_JOB_ID'),'protocol_sha256':sha(out/'registration.json')})
    deadline=time.monotonic()+p['resources']['world_wall_seconds'][case.split(':')[0]]
    attempts=[]
    for cell in (c for c in cells if c['case']==case):
        if (out/'stop_new_fits.json').exists():raise RuntimeError('another fit failed; preserve completed cells and stop')
        Path(cell['out']).parent.mkdir(parents=True,exist_ok=True)
        r=supervise([sys.executable,__file__,'fit-cell','--out',str(out),'--index',str(cell['index'])],
            deadline,3*2**30,directory/('cell-'+str(cell['index'])+'.log'),interval=2,
            env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
        samples=r.pop('samples',[]);r['peak_tree_rss_bytes']=max((s['rss_bytes'] for s in samples),default=0)
        attempts.append({'cell_index':cell['index'],**r});write(directory/'execution.json',{'attempts':attempts})
        if r['status']!='complete':
            try:claim(out/'stop_new_fits.json',{'at':utc(),'case':case,'cell_index':cell['index'],**r})
            except FileExistsError:pass
            raise RuntimeError('fit failure retained; evaluation blocked')
    write(directory/'complete.json',{'at':utc(),'n_fits':16,'protocol_sha256':sha(out/'registration.json')})


def seal(out):
    out=Path(out);p=validate(out);cells=validate_collection(out,p)
    if (out/'stop_new_fits.json').exists():raise ValueError('failed study cannot proceed to evaluation')
    artifacts=sealed_fits(cells)
    for cell in cells:
        receipt=read(Path(cell['out'])/'receipt.json')
        updates=48000 if cell['arm']=='online' else 100 if cell['arm']=='ablation' else 30000
        heads=1 if cell['arm']=='simpler' else 3 if cell['graph_size']==5 else 25
        if (receipt['unique_paid_rows']!=400 or receipt['calibration_rows']!=50 or len(receipt['heads'])!=heads or
                not math.isfinite(receipt['cpu_seconds']) or receipt['cpu_seconds']<=0 or
                any(h['updates']!=updates or h['eligible_rows']!=400 for h in receipt['heads'].values())):
            raise ValueError('fit accounting/head/optimizer matrix changed')
    for case in p['worlds']:
        directory=out/'world_execution'/case.replace(':','-');done=read(directory/'complete.json')
        if done['n_fits']!=16 or done['protocol_sha256']!=sha(out/'registration.json'):
            raise ValueError('world execution incomplete')
        attempts=read(directory/'execution.json')['attempts']
        if [a['cell_index'] for a in attempts]!=[c['index'] for c in cells if c['case']==case] or any(a['status']!='complete' for a in attempts):
            raise ValueError('world attempt matrix incomplete')
    target=out/'fit_seal.json'
    value={'registration_sha256':sha(out/'registration.json'),'matrix_sha256':sha(out/'matrix.json'),'artifacts':artifacts}
    if target.exists():
        if read(target)!=value:raise ValueError('existing full fit seal differs')
    else:write(target,value)
    return cells


def evaluate(out):
    out=Path(out);p=validate(out);cells=seal(out);sys.addaudithook(no_network)
    import numpy as np
    import torch
    from delivery_prospective_analysis import analyze
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    claim(out/'evaluation_started.json',{'at':utc(),'fit_seal_sha256':sha(out/'fit_seal.json')})
    descriptors=load_descriptors(out/'descriptors',read(out/'descriptor_manifest.json'))
    rows=[];scores={};charged=0;cpu0=time.process_time();start=time.monotonic()
    for case in p['worlds']:
        spec,actions=descriptors[case];test_dir=out/'evaluation'/case.replace(':','-')
        evaluation=collect_history(spec,actions['evaluation'],test_dir,evaluation=True)
        charged+=journal_count(test_dir/'queries.ndjson')
        for cell in (c for c in cells if c['case']==case):
            states=torch.load(Path(cell['out'])/'models.pt',map_location='cpu',weights_only=True)
            training=read(Path(cell['input_dir'])/'input.json')
            metric,predicted=evaluate_states(training,evaluation,states,p['source'],cell['arm'])
            np.savez(test_dir/('cell-'+str(cell['index'])+'-predictions.npz'),**predicted)
            scores[str(cell['index'])]={'cell':cell,'metric':metric,'evaluation_input_sha256':sha(test_dir/'input.json'),
                'predictions_sha256':sha(test_dir/('cell-'+str(cell['index'])+'-predictions.npz'))}
            if cell['init']==0 and cell['arm'] in ('delivery','online','simpler'):
                rows.append({'graph_size':cell['graph_size'],'system_id':case,'history':cell['history'],
                    'arm':cell['arm'],'init':0,'nmse':metric['nmse'],'status':'complete'})
    if charged!=16000 or len(scores)!=640 or len(rows)!=240:raise ValueError('evaluation response/score matrix incomplete')
    result=analyze(rows,{size:[c for c in p['worlds'] if c.startswith(str(size)+':')] for size in (5,30)})
    write(out/'scores.json',{'cells':scores,'primary_analysis':result,'primary_rows':rows,
        'scope':p['scope'],'new_training_responses':32000,'new_shared_evaluation_responses':16000,
        'fit_seal_sha256':sha(out/'fit_seal.json'),'evaluation_cpu_seconds':time.process_time()-cpu0,
        'evaluation_wall_seconds':time.monotonic()-start})
    fit_cpu=sum(read(Path(c['out'])/'receipt.json')['cpu_seconds'] for c in cells)
    write(out/'complete.json',{'at':utc(),'scores_sha256':sha(out/'scores.json'),'fit_seal_sha256':sha(out/'fit_seal.json'),
        'registration_sha256':sha(out/'registration.json'),'n_fits':640,'n_primary_cells':240,'charged_responses':48000,
        'fit_cpu_core_hours':fit_cpu/3600,'evaluation_cpu_core_hours':(time.process_time()-cpu0)/3600,
        'account':p['resources']['account'],'job_id':os.environ.get('SLURM_JOB_ID')})


def supervise_phase(out,phase):
    """Bound only a newly spawned ACE child; preserve failed phase custody."""
    out=Path(out);p=validate(out)
    if phase not in ('qualification','collect','evaluate'):raise ValueError('unregistered phase')
    wall=p['resources'][{'collect':'collection','evaluate':'evaluation'}.get(phase,phase)+'_wall_seconds']
    claim(out/(phase+'_supervisor_started.json'),{'at':utc(),'job_id':os.environ.get('SLURM_JOB_ID'),
        'registration_sha256':sha(out/'registration.json'),'wall_seconds':wall})
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
        'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(Path(__file__).resolve().parent),
        'ACE_DELIVERY_RUNNER_SOURCE':p['source'],'ACE_DELIVERY_DRAFT':str(out/'descriptors'),
        'ACE_DELIVERY_DRAFT_MANIFEST':str(out/'descriptor_manifest.json')}
    if phase=='qualification':
        command=[sys.executable,'-m','unittest','test_delivery_prospective_models',
            'test_delivery_prospective_io','test_delivery_prospective_batch',
            'test_delivery_theory','test_delivery_prospective_analysis']
    else:command=[sys.executable,__file__,phase,'--out',str(out)]
    start=time.monotonic();r=supervise(command,start+wall-20,p['resources']['rss_bytes'],
        out/(phase+'.log'),interval=1,env=env)
    samples=r.pop('samples',[]);r['peak_tree_rss_bytes']=max((s['rss_bytes'] for s in samples),default=0)
    r.update({'elapsed_seconds':time.monotonic()-start,'registration_sha256':sha(out/'registration.json'),
        'job_id':os.environ.get('SLURM_JOB_ID'),'account':p['resources']['account']})
    write(out/(phase+'_execution.json'),r)
    if r['status']!='complete':
        try:claim(out/'stop_new_fits.json',{'at':utc(),'phase':phase,**r})
        except FileExistsError:pass
        raise RuntimeError('phase failed; no replacement, preserve charged attempts')
    if phase=='qualification':
        write(out/'qualification_complete.json',{'at':utc(),'registration_sha256':sha(out/'registration.json'),
            'dependencies':p['dependencies'],'execution_sha256':sha(out/'qualification_execution.json'),
            'log_sha256':sha(out/'qualification.log'),'confirmation_responses_evaluated':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','collect','fit-world','fit-cell','seal','evaluate','supervise-phase'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--case');parser.add_argument('--index',type=int)
    for name in ('draft','project','source','gate','pilot'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--source-revision');parser.add_argument('--phase');a=parser.parse_args()
    if a.command=='freeze':freeze(a.draft,a.project,a.source,a.gate,a.pilot,a.out,a.source_revision)
    elif a.command=='fit-world':fit_world(a.out,a.case)
    elif a.command=='fit-cell':run_cell(a.out,a.index)
    elif a.command=='supervise-phase':supervise_phase(a.out,a.phase)
    else:globals()[a.command](a.out)
