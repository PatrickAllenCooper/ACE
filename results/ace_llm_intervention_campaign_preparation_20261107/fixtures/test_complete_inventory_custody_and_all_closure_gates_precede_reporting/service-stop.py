import sys
from pathlib import Path
sys.path.insert(0, '/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-campaign-preparation-20261107-01/source')
from llm_intervention_persistent_policy import serve_policy
count=0
def initialize():
    with Path('initializations.txt').open('x') as f: f.write('once')
    return decide
def decide(envelope):
    global count
    assert set(envelope)=={'version','state','task'}
    assert 'mechanisms' not in envelope['state'] and 'private_evaluation' not in envelope
    count+=1
    if 'stop'=='fail' and count==2: raise RuntimeError('deliberate second-call transport failure')
    return '{"stop":true}' if 'stop'=='stop' else '{"action_id":"observe"}'
serve_policy(initialize,identity={'model_sha256': 'ca05842e6610ea2f309f2e14f7bc1b567d9b6b686b29cb5478d21bcfb138fe49', 'source_sha256': 'ffa8d1137f78a08e06df40961cbaa684c757a7e294acab58999617cfa90e9f70', 'adapter_sha256': 'cad352f8082631587d811734dd669aab97fd2a8909dccb40ecbfa71c8e47c459', 'numeric_recipe': {'artificial': True}, 'numeric_recipe_sha256': 'cfa2a690605280381dd26a9d199aba338f7d539cfe0ef42dd4662a68161adf92'},max_requests=64)
