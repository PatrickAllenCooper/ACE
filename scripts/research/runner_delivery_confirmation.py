"""Outcome-blind preparation and guarded execution of the delivery registration.

Default CLI validates only. Launch requires a separately authorized approval receipt.
No confirmation workload is run by the validation suite.
"""
from __future__ import annotations
import argparse
import dataclasses
from collections import Counter
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'protocols/runner_delivery_confirmation_20261004'
REPO = Path('/Users/pat/code/ACE-Runner')
GRID = REPO / 'studies/2026-09-full-set/grid/tlam_mission1_full_grid.npz'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def validate_protocol(protocol=PROTOCOL, source=REPO, check_git=True):
    protocol, source = Path(protocol), Path(source)
    reg = read(protocol/'registration.json')
    cfg = read(protocol/'acquisition_config.json')
    audit = read(protocol/'seed_audit.json')
    if sha(protocol/'acquisition_config.json') != reg['acquisition_config_sha256'] or sha(protocol/'seed_audit.json') != reg['seed_audit_sha256']:
        raise ValueError('configuration/seed audit changed')
    document = protocol/'protocol.md'
    if not document.exists():
        document = ROOT/'docs/development/guidance/runner_delivery_confirmation_2026-10-04.md'
    if sha(document) != reg['protocol_sha256']:
        raise ValueError('protocol document changed')
    if len(reg['seeds']) != reg['n_required'] or len(set(reg['seeds'])) != reg['n_required'] or reg['seeds'] != audit['selected_seeds']:
        raise ValueError('seed count/order differs from registration')
    if set(reg['seeds']) & set(audit['known_seeds']) or audit['read_failures']:
        raise ValueError('seed exclusion audit failed')
    if cfg['proposer'] != 'random' or cfg['policy_update'] != 'none' or cfg['pretrain_steps'] or cfg['pretrain_interval']:
        raise ValueError('model calls forbidden')
    for path, expected in reg['source_hashes'].items():
        if sha(source/path) != expected:
            raise ValueError('source changed: '+path)
    if check_git and subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip() != reg['source_revision']:
        raise ValueError('source revision changed')
    return reg, cfg


class Journal:
    """Charge BEFORE invoking the emulator; failures still consume attempts."""
    def __init__(self, path, limit):
        if type(limit) is not int or limit<1:
            raise ValueError('positive integer attempt ceiling required')
        self.path, self.limit = Path(path), limit
        if self.path.exists():
            raise ValueError('no replacement/resumption of a started case')
        self.path.touch()
        self.count = 0

    def reserve(self):
        if self.count >= self.limit:
            raise RuntimeError('charged-call ceiling')
        self.count += 1
        with self.path.open('a') as f:
            f.write(json.dumps({'attempt':self.count,'at':utc()})+'\n')
            f.flush()


def journal_count(path):
    entries = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if [e['attempt'] for e in entries] != list(range(1,len(entries)+1)):
        raise ValueError('attempt journal sequence changed')
    return len(entries)


def process_rss(pid):
    """Sum RSS for the campaign child's process tree (KiB -> bytes)."""
    lines = subprocess.check_output(['ps','-axo','pid=,ppid=,rss='],text=True,timeout=1).splitlines()
    rows = [tuple(map(int,line.split())) for line in lines if len(line.split())==3]
    tree = {pid}
    for _ in rows:
        children = {p for p,parent,_ in rows if parent in tree}
        if children <= tree:
            break
        tree |= children
    return sum(rss*1024 for p,_,rss in rows if p in tree)


def supervise(command, deadline, rss_limit, log, sampler=process_rss, interval=.1, env=None):
    """Kill ONLY this newly created process group on wall/RSS limits.

    An independent watchdog enforces the deadline even while sampling blocks.
    OS scheduling latency is recorded rather than claimed to be zero.
    """
    if time.monotonic() >= deadline:
        return {'status':'time_limit','exit_code':None,'samples':[]}
    samples, reason, lock = [], [], threading.Lock()
    with Path(log).open('w') as stream:
        child = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                                 start_new_session=True,env=env)
        def stop(why):
            with lock:
                if child.poll() is None:
                    reason.append(why)
                    try:
                        os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        watchdog = threading.Timer(max(0,deadline-time.monotonic()),stop,args=('time_limit',))
        watchdog.start()
        try:
            while child.poll() is None:
                try:
                    rss = sampler(child.pid)
                    samples.append({'at':utc(),'rss_bytes':rss})
                    if rss > rss_limit:
                        stop('memory_limit')
                except (subprocess.SubprocessError,OSError):
                    stop('telemetry_failure')
                time.sleep(interval)
            code = child.wait()
        finally:
            watchdog.cancel()
            watchdog.join()
            if child.poll() is None:
                stop('supervisor_failure')
                child.wait()
    return {'status':reason[0] if reason else ('complete' if code==0 else 'failed'),
            'exit_code':code,'samples':samples,'finished_at':utc()}


def timing_gate(first_seconds, reg):
    return 1.2*reg['n_required']*first_seconds+300 <= reg['resource_proposal']['elapsed_ceiling_seconds']


def approval_check(path, protocol=PROTOCOL):
    receipt = read(path)
    if receipt.get('approved') is not True or receipt.get('registry_reconciled') is not True or not receipt.get('authorized_by') or receipt.get('registration_sha256') != sha(Path(protocol)/'registration.json'):
        raise ValueError('launch approval and registry reconciliation required')
    if receipt.get('adapter_sha256') != sha(__file__):
        raise ValueError('approval must bind the frozen adapter')
    expected=receipt.get('dependency_versions')
    actual={d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions()}
    if not expected or expected!=actual:
        raise ValueError('approved dependency environment changed')
    return receipt


def inventory(folder):
    folder=Path(folder)
    paths = sorted(folder.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('symlinks forbidden in custody inventory')
    return {str(p.relative_to(folder)):sha(p) for p in paths if p.is_file()}


def seal_cases(out, reg):
    out=Path(out)
    expected={str(s) for s in reg['seeds']}
    case_root=out/'cases'
    if {p.name for p in case_root.iterdir()} != expected:
        raise ValueError('case replacement/missing case')
    for seed in reg['seeds']:
        case=case_root/str(seed)
        done=read(case/'complete.json')
        if done['seed']!=seed or done['inits']!=reg['delivery']['inits'] or done['calls']!=journal_count(case/'calls.jsonl'):
            raise ValueError('incomplete or unpaired case')
        meta=read(case/'online/meta.json')
        rows=[json.loads(line) for line in (case/'online/observations.ndjson').read_text().splitlines()]
        digest=hashlib.sha256(json.dumps(rows,sort_keys=True,allow_nan=False).encode()).hexdigest()
        roles=dict(Counter(r['role'] for r in rows))
        expected_roles=meta['query_counts']['ace']
        if (digest!=done['rows_sha256'] or len(rows)!=done['calls']
                or any(r['method']!='ace' for r in rows)
                or len({r['query_index'] for r in rows})!=len(rows)
                or expected_roles['total']!=len(rows)
                or any(expected_roles.get(role)!=count for role,count in roles.items())
                or sum(v for k,v in expected_roles.items() if k!='total')!=len(rows)
                or done['calls']>reg['resource_proposal']['call_cap_per_case']):
            raise ValueError('persisted history/count custody failed')
        cfg=read(out/'protocol/acquisition_config.json');cfg['seed']=seed
        if done['config']!=cfg:
            raise ValueError('acquisition configuration changed')
        if meta['total_steps']!=reg['iterations'] or meta['has_baseline']:
            raise ValueError('step count/baseline differs from registration')
        nonroots={n for n,parents in meta['causal_dag'].items() if parents}
        eligible={'flat':sum(not(set(r['interventions']) & nonroots) for r in rows)}
        eligible.update({n:sum(n not in r['interventions'] for r in rows) for n in nonroots})
        for node in nonroots:
            if not (case/'online/mlps'/f'{node}.pt').is_file():
                raise ValueError('missing original node weights')
        for init in reg['delivery']['inits']:
            receipt=read(case/f'init-{init}'/'receipt.json')
            if (receipt['source_rows_sha256']!=done['rows_sha256'] or receipt['charged_rows']!=done['calls']
                    or receipt['custody_metadata_sha256']!=sha(case/'online/meta.json')
                    or receipt['eligible_rows']!=eligible or receipt['epochs']!=reg['delivery']['epochs']
                    or receipt['seed']!=init or receipt['evaluation'] is not None):
                raise ValueError('paired observation parity failed')
        for path in ['online/mlp.pt','online/meta.json','online/observations.ndjson']:
            if not (case/path).is_file():
                raise ValueError('missing original artifacts')
        for init in reg['delivery']['inits']:
            if not (case/f'init-{init}'/'models.pt').is_file():
                raise ValueError('missing fitted weights')
    if sum(journal_count(case_root/str(s)/'calls.jsonl') for s in reg['seeds'])>reg['resource_proposal']['total_call_cap']:
        raise ValueError('total call ceiling')
    return {'seeds':reg['seeds'],'case_hashes':inventory(case_root),'sealed_at':utc()}


def verify_seal(out, reg):
    out=Path(out)
    manifest=read(out/'sealed.json')
    if manifest['seeds']!=reg['seeds'] or manifest['case_hashes']!=inventory(out/'cases'):
        raise ValueError('seal changed; evaluator refuses access')
    # Recheck semantic parity as well as byte custody.
    seal_cases(out,reg)
    return manifest


def no_network(event,args):
    if event=='socket.connect':
        raise PermissionError('network forbidden in campaign worker')


def no_grid_access(event,args):
    no_network(event,args)
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        if Path(os.fsdecode(args[0])).suffix.lower() in ('.npz','.npy'):
            raise PermissionError('evaluation arrays forbidden in acquisition/fitting process')


def case_worker(out,seed,limit):
    out=Path(out)
    campaign=read(out/'campaign.json')
    protocol=out/'protocol'
    reg,cfg=validate_protocol(protocol,out/'source',check_git=False)
    if campaign['registration_sha256']!=sha(protocol/'registration.json') or seed not in reg['seeds']:
        raise ValueError('unregistered case')
    approval_check(out/'approval.json',protocol)
    if campaign['source_inventory']!=inventory(out/'source'):
        raise ValueError('source snapshot changed')
    sys.addaudithook(no_grid_access)
    sys.path.insert(0,str(out/'source'))
    import torch
    from ace.config import ACEConfig
    from ace.oracle import ACEOracle
    from ace.delivery import refit_delivery
    from environments.tlam_mission1 import TLAMMission1Environment
    torch.set_num_threads(reg['resource_proposal']['threads'])
    torch.set_num_interop_threads(1)
    case=out/'cases'/str(seed)
    case.mkdir()
    journal=Journal(case/'calls.jsonl',min(limit,reg['resource_proposal']['call_cap_per_case']))
    base=TLAMMission1Environment()
    class Counted:
        def __getattr__(self,name): return getattr(base,name)
        def generate(self,n_samples=1,interventions=None):
            if n_samples!=1: raise ValueError('only one-row charged calls registered')
            journal.reserve()
            return base.generate(n_samples=n_samples,interventions=interventions)
        def evaluate(self,*args,**kwargs):
            journal.reserve()
            return base.evaluate(*args,**kwargs)
    cfg['seed']=seed
    with torch.device('cpu'):
        oracle=ACEOracle.from_env(Counted(),config=ACEConfig(**cfg))
        if oracle.device.type!='cpu': raise ValueError('CPU only')
        result=oracle.run_sweep(reg['iterations'],run_random_baseline=False)
        result.run_config={'config':dataclasses.asdict(oracle.config),'confirmation_source':reg['source_revision']}
        result.save(case/'online')
        rows=result.observations.records
        if any(r['method']!='ace' for r in rows) or len(rows)!=journal.count:
            raise ValueError('paid history accounting failed')
        rows_sha=hashlib.sha256(json.dumps(rows,sort_keys=True,allow_nan=False).encode()).hexdigest()
        timings=[]
        for init in reg['delivery']['inits']:
            start=time.monotonic()
            receipt=refit_delivery(result,case/f'init-{init}',custody_meta=case/'online/meta.json',epochs=reg['delivery']['epochs'],seed=init)
            if receipt['source_rows_sha256']!=rows_sha: raise ValueError('history changed')
            timings.append({'init':init,'seconds':time.monotonic()-start,'ended_at':utc()})
    validate_protocol(protocol,out/'source',check_git=False)
    write(case/'complete.json',{'seed':seed,'inits':reg['delivery']['inits'],'calls':journal.count,
                              'rows_sha256':rows_sha,'config':cfg,'refit_timings':timings,'completed_at':utc()})


def evaluate_worker(out):
    out=Path(out); protocol=out/'protocol'
    reg,_=validate_protocol(protocol,out/'source',check_git=False)
    approval_check(out/'approval.json',protocol)
    campaign=read(out/'campaign.json')
    if campaign['source_inventory']!=inventory(out/'source'):
        raise ValueError('source snapshot changed')
    sys.addaudithook(no_network)
    verify_seal(out,reg) # BEFORE opening grid or importing scoring code.
    campaign=read(out/'campaign.json')
    if campaign['registration_sha256']!=sha(protocol/'registration.json') or sha(GRID)!=reg['evaluation']['file_sha256']:
        raise ValueError('registration/grid changed')
    sys.path.insert(0,str(out/'source'))
    import numpy as np
    import torch
    from ace.grid_eval import GridTruth,chain_predict,flat_predict,score
    from ace.oracle import MLPSurrogate
    torch.set_num_threads(reg['resource_proposal']['threads'])
    torch.set_num_interop_threads(1)
    truth=GridTruth.load(GRID,target='engagement_rate',env_id=reg['environment'])
    if truth.sha256!=reg['evaluation']['array_sha256'] or truth.n_points!=reg['evaluation']['points']:
        raise ValueError('grid identity mismatch')
    def predictor(state):
        model=MLPSurrogate.from_state_dict(state).eval()
        if any(t.device.type!='cpu' for t in model.parameters()): raise ValueError('CPU only')
        def predict(x):
            with torch.inference_mode():
                k=reg['evaluation']['inference_chunk_rows']
                p=np.concatenate([model(torch.tensor(x[i:i+k],dtype=torch.float32)).numpy() for i in range(0,len(x),k)])
            if not np.isfinite(p).all(): raise ValueError('nonfinite prediction')
            return p
        return predict
    pairs=[]
    with torch.device('cpu'):
        for seed in reg['seeds']:
            case=out/'cases'/str(seed); meta=read(case/'online/meta.json'); dag=meta['causal_dag']
            if meta['feature_names']!=list(truth.feature_names) or meta['target_name']!=truth.target: raise ValueError('schema mismatch')
            def metrics(states):
                return {'chain':score(chain_predict({n:predictor(s) for n,s in states.items() if n!='flat'},dag,truth),truth.y,truth.levels()),
                        'flat':score(flat_predict(predictor(states['flat']),truth),truth.y,truth.levels())}
            original={'flat':torch.load(case/'online/mlp.pt',weights_only=True,map_location='cpu')}
            original.update({p.stem:torch.load(p,weights_only=True,map_location='cpu') for p in (case/'online/mlps').glob('*.pt')})
            online=metrics(original)
            fits=[{'init':i,**metrics(torch.load(case/f'init-{i}'/'models.pt',weights_only=True,map_location='cpu'))} for i in reg['delivery']['inits']]
            pairs.append({'seed':seed,'online':online,'delivery':fits})
    spec=importlib.util.spec_from_file_location('frozen_stats',out/'source/scripts/analysis/study_stats.py')
    stats=importlib.util.module_from_spec(spec);sys.modules[spec.name]=stats;spec.loader.exec_module(stats)
    floor=reg['evaluation']['error_floor']
    a={r['seed']:statistics.median(max(1-f['chain']['exact'],floor) for f in r['delivery']) for r in pairs}
    b={r['seed']:max(1-r['online']['chain']['exact'],floor) for r in pairs}
    comparison=stats.paired_log_ratio(a,b,floor=floor,level=reg['analysis']['confidence_level'])
    confirmed=comparison['r']<=reg['analysis']['required_ratio_max'] and comparison['ci_hi']<1 and comparison['perm_p']<reg['analysis']['alpha']
    verify_seal(out,reg)
    write(out/'scores.json',{'pairs':pairs,'comparison':comparison,'confirmed':confirmed,'scored_at':utc(),'scope':reg['claim']})


def _launch(out,approval,python,protocol=PROTOCOL):
    """Only future authorized execution calls this; never called by validation."""
    started=time.monotonic(); reg,_=validate_protocol(protocol)
    approval_check(approval,protocol)
    deadline=started+reg['resource_proposal']['elapsed_ceiling_seconds']
    out=Path(out);out.mkdir(exist_ok=False);(out/'cases').mkdir()
    import shutil
    shutil.copytree(protocol,out/'protocol');shutil.copy2(approval,out/'approval.json')
    shutil.copy2(ROOT/'docs/development/guidance/runner_delivery_confirmation_2026-10-04.md',out/'protocol/protocol.md')
    archive=out/'source.tar'
    with archive.open('wb') as f: subprocess.run(['git','archive',reg['source_revision']],cwd=REPO,stdout=f,check=True)
    (out/'source').mkdir()
    with tarfile.open(archive) as f: f.extractall(out/'source',filter='data')
    validate_protocol(out/'protocol',out/'source',check_git=False)
    shutil.copy2(__file__,out/'adapter.py')
    write(out/'campaign.json',{'registration_sha256':sha(protocol/'registration.json'),'adapter_sha256':sha(__file__),'started_at':utc(),'phase':'authorized_acquisition','source_inventory':inventory(out/'source')})
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','USE_LLM':'false','ACE_OBS_LOG':'on','OMP_NUM_THREADS':str(reg['resource_proposal']['threads']),'MKL_NUM_THREADS':str(reg['resource_proposal']['threads']),'PYTHONDONTWRITEBYTECODE':'1'}
    total=0;reports=[];status='incomplete'
    for index,seed in enumerate(reg['seeds']):
        before=time.monotonic()
        report=supervise([python,str(out/'adapter.py'),'--case-worker',str(out),str(seed),str(reg['resource_proposal']['total_call_cap']-total)],deadline,reg['resource_proposal']['rss_ceiling_bytes'],out/f'case-{seed}.log',env=env,interval=.5)
        reports.append({'seed':seed,**report});write(out/'runtime.json',reports)
        journal=out/'cases'/str(seed)/'calls.jsonl'
        total+=journal_count(journal) if journal.exists() else 0
        if report['status']!='complete': status=report['status'];break
        if index==0 and not timing_gate(time.monotonic()-before,reg): status='timing_gate_failed';break
    else:
        write(out/'sealed.json',seal_cases(out,reg))
        report=supervise([python,str(out/'adapter.py'),'--evaluate-worker',str(out)],deadline,reg['resource_proposal']['rss_ceiling_bytes'],out/'evaluation.log',env=env,interval=.5)
        reports.append({'stage':'evaluation',**report});status=report['status']
    write(out/'execution.json',{'status':status,'calls':total,'elapsed_seconds':time.monotonic()-started,'completed_at':utc(),'reports':reports})
    if status!='complete' and (out/'scores.json').exists(): (out/'scores.json').rename(out/'INCOMPLETE-scores.json')
    return status


def launch(out,approval,python,protocol=PROTOCOL):
    """Preserve incomplete artifacts and a failure receipt; never resume/retry."""
    out=Path(out)
    if out.exists():
        raise ValueError('output already exists; no resumption')
    try:
        return _launch(out,approval,python,protocol)
    except Exception as error:
        if out.exists():
            if (out/'scores.json').exists():
                (out/'scores.json').rename(out/'INCOMPLETE-scores.json')
            write(out/'execution.json',{'status':'incomplete','error':str(error),
                                       'finished_at':utc(),'replacements':0})
        raise


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--case-worker':
        case_worker(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]));return
    if len(sys.argv)>1 and sys.argv[1]=='--evaluate-worker':
        evaluate_worker(sys.argv[2]);return
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--approval');p.add_argument('--out');p.add_argument('--python');args=p.parse_args()
    if not args.launch:
        reg,_=validate_protocol();print(json.dumps({'validated':True,'cases':len(reg['seeds']),'launched':False}));return
    if not all([args.approval,args.out,args.python]): p.error('launch requires approval, output and interpreter')
    print(launch(args.out,args.approval,args.python))

if __name__=='__main__': main()
