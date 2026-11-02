from pathlib import Path
import hashlib,json,sys,importlib.metadata
from transformers import AutoTokenizer
from llm_intervention_contract import Action,Node,ChargedExecutor,canonical
from llm_intervention_scm import LinearGaussianSCM
from llm_intervention_task import PredictionTask,make_envelope
from llm_intervention_language_policy import prepare_task_tokens
root=Path(sys.argv[1]); inputs=json.loads((root/'input_pins.json').read_bytes())
for section,folder in [('tokenizer_files','tokenizer'),('sources','source')]:
    for name,pin in inputs[section].items():
        assert hashlib.sha256((root/folder/name).read_bytes()).hexdigest()==pin
tokenizer=AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True,trust_remote_code=False)
results=[]
for i,(n,m,h,q) in enumerate([(2,2,1,2),(2,2,8,2),(5,8,8,8),(5,8,32,8),(8,32,80,64)]):
    case=root/f'case{i}';case.mkdir()
    graph=tuple(Node(f'X{j}',tuple(f'X{k}' for k in range(j)),'unit',(-1.,1.) if j==0 else None) for j in range(n))
    measured=tuple(f'X{j}' for j in range(1,n))
    menu=tuple(Action(f'a{k:02}', 'intervene',(('X0',-1.+2.*k/(m-1)),),measured,1) for k in range(m))
    queries=tuple(Action(f'q{k:02}','intervene',(('X0',-1.+2.*k/(q-1)),),(f'X{n-1}',),1) for k in range(q))
    task=PredictionTask(f'X{n-1}',queries,tuple(1./q for _ in range(q)),1.,.01)
    learner=LinearGaussianSCM(graph,{node.name:1. for node in graph})
    # Deliberately artificial zero measurements, not a scientific world.
    with ChargedExecutor(graph,menu,100,case/'artificial_journal.jsonl',lambda a,r:{name:0. for name in a.measure}) as ex:
        for k in range(h):
            learner.update(ex.execute_completion(canonical({'action_id':menu[k % m].action_id})))
        envelope=make_envelope(ex.state('Predict the structural response under the listed interventions',learner.summaries()),task)
    (case/'envelope.json').write_text(canonical(envelope)+'\n')
    grammar,meta=prepare_task_tokens(envelope,tokenizer,max_tokens=32768,max_new_tokens=32)
    (case/'tokens.json').write_text(json.dumps({'prompt_ids':grammar.prompt_ids,'completion_ids':grammar.completion_ids})+'\n')
    results.append({'nodes':n,'menu':m,'history':h,'queries':q,**meta})
report={'scope':'local real-tokenizer capacity profile on five artificial states, no weights or GPU','runtime':{name:importlib.metadata.version(name) for name in ['transformers','tokenizers','torch']},'cases':results,'artificial_responses':sum(x['history'] for x in results),'scientific_responses':0,'pretrained_weight_loads':0,'target_runtime_qualified':False}
(root/'profile.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
