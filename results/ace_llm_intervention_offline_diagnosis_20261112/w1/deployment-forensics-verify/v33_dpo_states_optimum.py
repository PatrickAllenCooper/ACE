import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import pickle, collections
import numpy as np
ev = pickle.load(open(SCR + '/ev_ref.pkl', 'rb'))
for arm in ['dpo_llm', 'sft_llm', 'frozen_llm', 'learned_ranker']:
    print(arm)
    for lo, hi in [(0, 3), (4, 7), (8, 15), (16, 23)]:
        E = [e for e in ev if e['arm'] == arm and lo <= e['d'] <= hi]
        c = collections.Counter(SHORT[e['best']] for e in E)
        print('   d %2d-%2d n=%3d reference optimum on own states: %s' % (lo, hi, len(E), dict(c.most_common())))
# share of decisions where the chosen action is in the bottom two of the affordable menu
for arm in ARMS:
    E = [e for e in ev if e['arm'] == arm and e['action'] != 'STOP']
    print('%-18s chosen action ranked last or second-last among affordable: %.1f%% ; ranked in top two: %.1f%%' % (arm, 100 * np.mean([e['rank'] >= e['n'] - 1 for e in E]), 100 * np.mean([e['rank'] <= 2 for e in E])))
