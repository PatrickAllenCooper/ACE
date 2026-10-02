"""One frozen archived-fit abstention rule; development only."""
import csv,hashlib,json,pathlib,statistics
root=pathlib.Path('results/local_prior_gate_dev_20260926'); cells={}; hashes={}
for seed in range(100,112):
 p=root/f'seed_{seed}';b=(p/'metrics.csv').read_bytes();h=hashlib.sha256(b).hexdigest();assert h==json.loads((p/'complete.json').read_text())['metrics_sha256'];hashes[str(seed)]=h
 for row in csv.DictReader(b.decode().splitlines()):cells[seed,row['condition'],int(row['budget']),row['method']]=row
out=[]
for budget in [16,32,64]:
 for condition in ['correct','wrong']:
  base=[];chosen=[];accepted=0
  for seed in range(100,112):
   broad=cells[seed,condition,budget,'broad'];proposal=cells[seed,condition,budget,'proposal'];gate=cells[seed,condition,budget,'validation_gate']
   accept=float(gate['validation_proposal_sse'])<=.8*float(gate['validation_base_sse']);accepted+=accept
   base.append(float(broad['mse']));chosen.append(float((proposal if accept else broad)['mse']))
  ratio=statistics.mean(chosen)/statistics.mean(base)
  out.append({'budget':budget,'condition':condition,'accepted':accepted,'n':12,'broad_mse':statistics.mean(base),'abstaining_mse':statistics.mean(chosen),'ratio':ratio,'gate_pass':ratio<=1.05 if condition=='wrong' else ratio<1})
p=pathlib.Path('results/prior_abstention_diagnostic_20261002');p.mkdir(exist_ok=False)
r={'protocol_commit':'1a3b01d5','source_hashes':hashes,'rows':out,'all_gates_pass':all(x['gate_pass'] for x in out),'new_queries':0,'model_calls':0,'scope':'post hoc development; no statistical safety guarantee'};b=(json.dumps(r,indent=2)+'\n').encode();(p/'result.json').write_bytes(b);(p/'complete.json').write_text(json.dumps({'result_sha256':hashlib.sha256(b).hexdigest(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n');print(json.dumps(out,indent=2))
