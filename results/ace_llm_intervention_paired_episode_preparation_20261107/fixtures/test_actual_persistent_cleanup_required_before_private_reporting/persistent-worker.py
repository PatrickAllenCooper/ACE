import sys
sys.path.insert(0,'/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-paired-episode-preparation-20261107-01/source')
from llm_intervention_persistent_policy import serve_policy
serve_policy(lambda: lambda envelope: '{"action_id":"observe"}', identity={'model_sha256': '1111111111111111111111111111111111111111111111111111111111111111', 'source_sha256': '2222222222222222222222222222222222222222222222222222222222222222', 'adapter_sha256': '3333333333333333333333333333333333333333333333333333333333333333', 'numeric_recipe': {'artificial': True}, 'numeric_recipe_sha256': 'cfa2a690605280381dd26a9d199aba338f7d539cfe0ef42dd4662a68161adf92'})
