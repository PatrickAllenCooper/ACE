"""Recompute the deployed target-risk planner's per-action scores on EVERY saved
public state visited by EVERY arm (4,371 decisions), with 256 fantasy draws
(the function's maximum; the deployed planner used 32). Pure deterministic
numpy computation on saved states: no environment response, no model weights,
no new episode. Seeds derive from the envelope SHA256 so results are reproducible
and identical states get identical scores (common random numbers)."""
import sys, os, pickle, time, json, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + '/lib')
import numpy as np

def work(item):
    sha, env = item
    from llm_intervention_target_risk import choose_target_risk
    from llm_intervention_task import unpack_envelope
    state, task, graph, menu, learner = unpack_envelope(env)
    seed = int(sha[:16], 16)
    dec, plan = choose_target_risk(state, seed=seed, draws=256, **task.risk_arguments())
    # posterior-uncertainty features
    feats = {'risk_before': plan['risk_before']}
    for s in state['scm']:
        cov = np.array(s['parameter_covariance'])
        feats['tr_' + s['node']] = float(np.trace(cov))
        feats['rows_' + s['node']] = s['eligible_rows']
    return sha, {'scores': plan['risk_reduction_per_cost'], 'se': plan['monte_carlo_standard_errors'],
                 'risk_after': plan['risk_after'], 'risk_before': plan['risk_before'], 'feats': feats}

if __name__ == '__main__':
    import multiprocessing as mp
    D = pickle.load(open(HERE + '/out/decisions.pkl', 'rb'))
    uniq = {}
    for r in D['rows']:
        uniq.setdefault(r['envelope_sha256'], r['envelope'])
    items = sorted(uniq.items())
    print('unique states', len(items), 'of', len(D['rows']), flush=True)
    t = time.time()
    out = {}
    with mp.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        for i, (sha, res) in enumerate(pool.imap_unordered(work, items, chunksize=8)):
            out[sha] = res
            if i % 250 == 0:
                print(i, round(time.time() - t, 1), flush=True)
    pickle.dump(out, open(HERE + '/out/tr_scores_256.pkl', 'wb'))
    print('done', len(out), round(time.time() - t, 1), flush=True)
