"""Frozen, finite CPU normalization qualification; never acquires or scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import runner_delivery_confirmation as guard

SOURCE=Path('/Users/pat/code/ACE-Runner')
CASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/delivery-confirmation-20261005/cases/27424209')
ORDER=[(0,'baseline'),(0,'candidate'),(1,'candidate'),(1,'baseline'),(2,'baseline'),(2,'candidate')]
CHECKPOINTS={1,10,100,1000,10000,30000}


def worker(out):
    out=Path(out);frozen=guard.read(out/'frozen.json')
    assert guard.sha(__file__)==frozen['script_sha256']
    assert guard.inventory(CASE)==frozen['case_hashes']
    guard.validate_protocol()
    guard.approval_check(out/'approval.json')
    sys.addaudithook(guard.no_grid_access)
    sys.path.insert(0,str(SOURCE))
    import torch
    import numpy as np
    from ace.oracle import MLPSurrogate
    from ace.result import ACEResult
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    def digest(value):
        h=hashlib.sha256()
        def visit(v):
            if torch.is_tensor(v):
                if v.device.type!='cpu':raise ValueError('CPU only')
                h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.detach().contiguous().numpy().tobytes())
            elif isinstance(v,dict):
                for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
            elif isinstance(v,(list,tuple)):
                for item in v:visit(item)
            else:h.update(json.dumps(v,allow_nan=False).encode())
        visit(value);return h.hexdigest()
    def trace(model,opt,loss):
        return digest({'weights':model.state_dict(),'optimizer':opt.state_dict(),
                       'gradients':[p.grad for p in model.parameters()],'loss':loss})
    # Deterministic short fixture, no emulator/grid or external data.
    with torch.device('cpu'),torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(71)
        a=MLPSurrogate(3);a.set_input_range([-1,-2,-3],[2,3,4])
        b=MLPSurrogate.from_state_dict(a.state_dict());x=torch.arange(24,dtype=torch.float32).reshape(8,3)/10;y=torch.arange(8,dtype=torch.float32)/10
        cached=(x-b.in_lo)/torch.clamp(b.in_hi-b.in_lo,min=1e-9)
        assert torch.equal(cached,(x-a.in_lo)/torch.clamp(a.in_hi-a.in_lo,min=1e-9))
        oa=torch.optim.Adam(a.parameters(),lr=.002);ob=torch.optim.Adam(b.parameters(),lr=.002)
        for _ in range(20):
            oa.zero_grad();ob.zero_grad();la=torch.nn.functional.mse_loss(a(x),y);lb=torch.nn.functional.mse_loss(b.net(cached).squeeze(-1),y)
            assert torch.equal(la,lb);la.backward();lb.backward();oa.step();ob.step()
            if trace(a,oa,la)!=trace(b,ob,lb):raise ValueError('fixture exact-equivalence failure')
    guard.write(out/'fixture.json',{'passed':True,'steps':20,'at':guard.utc()})
    result=ACEResult.load(CASE/'online')
    original=(SOURCE/'ace/delivery.py').read_text()
    source=original.replace('for _ in range(epochs):','for epoch_index in range(epochs):')
    source=source.replace('            states[name] = model.state_dict()', '            states[name] = model.state_dict()')
    source=source.replace('                optimizer.step()','                optimizer.step()\n                if epoch_index+1 in CHECKPOINTS:\n                    checkpoint(name,epoch_index+1,model,optimizer,loss)')
    reports=[]
    for init,kind in ORDER:
        def checkpoint(name,epoch,model,opt,loss):
            folder=out/'traces'/str(init)/kind;folder.mkdir(parents=True,exist_ok=True)
            key=f'{name}-{epoch}.json';value={'digest':trace(model,opt,loss),'epoch':epoch,'name':name}
            guard.write(folder/key,value)
            other=out/'traces'/str(init)/('candidate' if kind=='baseline' else 'baseline')/key
            if other.exists() and guard.read(other)!=value:raise ValueError(f'bitwise mismatch: {init}/{name}/{epoch}')
        text=source
        if kind=='candidate':
            text=text.replace('            for epoch_index in range(epochs):','            cached_x=(x-model.in_lo)/torch.clamp(model.in_hi-model.in_lo,min=1e-9) if result.normalize_inputs else x\n            for epoch_index in range(epochs):')
            text=text.replace('mse_loss(model(x), y)','mse_loss(model.net(cached_x).squeeze(-1), y)')
        ns={'checkpoint':checkpoint,'CHECKPOINTS':CHECKPOINTS};exec(compile(text,f'<frozen-{kind}>','exec'),ns)
        guard.write(out/'progress.json',{'stage':'fit','init':init,'kind':kind,'at':guard.utc(),'completed_fits':len(reports)})
        start=time.monotonic()
        receipt=ns['refit_delivery'](result,out/f'{init}-{kind}',custody_meta=CASE/'online/meta.json',epochs=30000,seed=init)
        elapsed=time.monotonic()-start
        assert receipt['source_rows_sha256']==frozen['rows_sha256']
        reports.append({'init':init,'kind':kind,'seconds':elapsed,'at':guard.utc()});guard.write(out/'fits.json',reports)
    for init in [0,1,2]:
        if guard.inventory(out/'traces'/str(init)/'baseline')!=guard.inventory(out/'traces'/str(init)/'candidate'):raise ValueError('trace inventory mismatch')
    assert guard.inventory(CASE)==frozen['case_hashes']
    totals={kind:sum(row['seconds'] for row in reports if row['kind']==kind) for kind in ['baseline','candidate']}
    saving=totals['baseline']-totals['candidate']
    guard.write(out/'result.json',{'exact_equivalence':True,'totals_seconds':totals,'paired_saving_seconds':saving,'runtime_target_passed':saving>=57.28175241633045 and totals['candidate']<=313.57343558266685,'reports':reports,'scorer_calls':0,'emulator_calls':0,'campaign_restart_authorized':False,'at':guard.utc()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--run');a=p.parse_args()
    if a.worker:worker(a.worker);return
    if not a.run:p.error('explicit --run required')
    out=Path(a.run);frozen=guard.read(out/'frozen.json')
    assert guard.sha(__file__)==frozen['script_sha256']
    budget=frozen['accounting'];assert budget['prior_charged_and_reserved_seconds']+900<=7200
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'6','MKL_NUM_THREADS':'6','PYTHONDONTWRITEBYTECODE':'1','USE_LLM':'false'}
    report=guard.supervise([sys.executable,str(Path(__file__).resolve()),'--worker',str(out)],time.monotonic()+900,8589934592,out/'worker.log',sampler=lambda _:guard.process_rss(os.getpid()),interval=.5,env=env)
    guard.write(out/'execution.json',report)
    print(report['status'])

if __name__=='__main__':main()
