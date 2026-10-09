#!/usr/bin/env python3
"""Descriptive post-hoc display, all original worlds; never a new stopping trial."""
import csv,hashlib,json,pathlib,statistics
ROOT=pathlib.Path(__file__).resolve().parents[2]
root=ROOT/'results/research_pev_shift30_mean_confirmation/shift30'
out=ROOT/'presentations/ACE_theory_and_evidence/recorded_curves_2026-10-09.json'
pins={}; curves={}; final={}
for policy in ('pev','nonleaf_random_ens'):
 runs=[]
 for seed in range(5000,5020):
  path=root/policy/f'seed_{seed}'/'trajectory.csv';pins[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
  rows=list(csv.DictReader(path.open()));assert len(rows)==32
  budgets=[50*(i+1)+40*(i//3) for i in range(32)]
  assert [int(r['step']) for r in rows]==list(range(32))
  assert [int(r['query_samples']) for r in rows]==budgets
  runs.append([float(r['feasible_mean_nonroot_loss']) for r in rows])
 final[policy]=[r[-1] for r in runs]
 curves[policy]=[{'batches':i+1,'intervention_responses':50*(i+1),'observational_responses':40*(i//3),'total_responses':budgets[i],'mean_mse':statistics.mean(r[i] for r in runs),'world_mse':[r[i] for r in runs]} for i in range(32)]
threshold=curves['nonleaf_random_ens'][-1]['mean_mse']
hits={p:next(row for row in rows if row['mean_mse']<=threshold) for p,rows in curves.items()}
system=root/'pev/seed_5000/system.json';s=json.loads(system.read_text());pins[str(system.relative_to(ROOT))]=hashlib.sha256(system.read_bytes()).hexdigest()
value={'source_scope':'20 fixed 30-node DAGs within one five-layer generator, seeds5000–5019; observed-parent noise-free feasible nonroot mechanism MSE','curves':curves,'target':threshold,'target_rule':'post-hoc random final arithmetic mean; first recorded crossing, no interpolation, no monotonic smoothing','first_crossings':hits,'batch_reduction_fraction':1-hits['pev']['batches']/hits['nonleaf_random_ens']['batches'],'total_response_reduction_fraction':1-hits['pev']['total_responses']/hits['nonleaf_random_ens']['total_responses'],'final_mean_error_reduction_fraction':1-curves['pev'][-1]['mean_mse']/threshold,'final_world_wins':sum(a<b for a,b in zip(final['pev'],final['nonleaf_random_ens'])),'graph':s['graph'],'nodes':s['nodes'],'pins':pins,'limitations':['descriptive post-hoc group-mean crossing, not a prospective stopping policy or per-system savings','random crosses back above target later','secondary random contrast; primary coverage gate failed; variance-score superiority unresolved','earliest checkpoints favor random; report full curves','action calls each buy50responses; observational refresh cost included separately']}
out.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({k:value[k] for k in ['target','batch_reduction_fraction','total_response_reduction_fraction','final_mean_error_reduction_fraction','final_world_wins','graph']}))
