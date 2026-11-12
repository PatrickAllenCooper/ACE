"""Check that re-running the planner with the SAVED seed and 32 draws reproduces
the SAVED per-action scores and decision on a sample of target_risk decisions."""
import sys, os, pickle, warnings, random
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + '/lib')
from llm_intervention_target_risk import choose_target_risk
from llm_intervention_task import unpack_envelope
D = pickle.load(open(HERE + '/out/decisions.pkl', 'rb'))
T = [r for r in D['rows'] if r['arm'] == 'target_risk']
random.Random(0).shuffle(T)
mx = 0.0; same = 0; n = 0
for r in T[:80]:
    state, task, *_ = unpack_envelope(r['envelope'])
    dec, plan = choose_target_risk(state, seed=int(r['meta']['seed_decimal']), draws=32, **task.risk_arguments())
    saved = r['meta']['planning_metadata']['risk_reduction_per_cost']
    mx = max(mx, max(abs(plan['risk_reduction_per_cost'][a] - saved[a]) for a in saved))
    same += dec.get('action_id') == r['action']; n += 1
print('sampled target_risk decisions replayed with saved seed, 32 draws: %d ; decision identical in %d ; max abs score difference %.3e' % (n, same, mx))
