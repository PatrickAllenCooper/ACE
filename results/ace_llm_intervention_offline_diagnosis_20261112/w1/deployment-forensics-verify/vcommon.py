"""Common paths and helpers for the independent verifier (deployment-forensics-verify)."""
import sys, site
site.addsitedir('/Users/pat/Library/Python/3.9/lib/python/site-packages')
import json, hashlib, os
GPU = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch/gpu'
CPU = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108/terminal02/remote/launch/cpu'
CAMP = CPU + '/worker/artifacts/campaign'
SCR = '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify'
ACTIONS = ['observe_all', 'set_x_neg', 'set_x_pos', 'set_x_zero', 'set_xy', 'set_y_neg', 'set_y_pos', 'set_y_zero']
SHORT = {'observe_all': 'obs', 'set_x_neg': 'x-', 'set_x_pos': 'x+', 'set_x_zero': 'x0', 'set_xy': 'xy',
         'set_y_neg': 'y-', 'set_y_pos': 'y+', 'set_y_zero': 'y0', 'STOP': 'stop'}
ARMS = ['uniform_random', 'menu_coverage', 'local_information', 'target_risk', 'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm']

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()
