"""SUPPLEMENT: target-risk scores (256 draws) at the 128 non-terminal TRAINING
states embedded in cpu-phase/numeric-training-plan.json. How do the supervised
labels and the measured preference pairs relate to the planner's score?"""
import sys, os, pickle, json, collections, math, warnings, hashlib
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + '/lib')
import numpy as np

def work(item):
    rid, env = item
    from llm_intervention_target_risk import choose_target_risk
    from llm_intervention_task import unpack_envelope
    state, task, graph, menu, learner = unpack_envelope(env)
    seed = int(hashlib.sha256(rid.encode()).hexdigest()[:16], 16)
    dec, plan = choose_target_risk(state, seed=seed, draws=256, **task.risk_arguments())
    return rid, {'scores': plan['risk_reduction_per_cost'], 'se': plan['monte_carlo_standard_errors'], 'risk_before': plan['risk_before']}

if __name__ == '__main__':
    import multiprocessing as mp
    from scipy import stats
    T = pickle.load(open(HERE + '/out/train_states.pkl', 'rb'))
    live = [r for r in T if not r['env']['state']['terminal']]
    cache = HERE + '/out/train_tr_256.pkl'
    if os.path.exists(cache):
        TR = pickle.load(open(cache, 'rb'))
    else:
        with mp.Pool(9) as pool:
            TR = dict(pool.imap_unordered(work, [(r['rid'], r['env']) for r in live], chunksize=2))
        pickle.dump(TR, open(cache, 'wb'))
    print('training states scored:', len(TR))
    A = ['observe_all', 'set_x_neg', 'set_x_zero', 'set_x_pos', 'set_y_neg', 'set_y_zero', 'set_y_pos', 'set_xy']
    # (a) supervised label vs TR
    agree = 0; near = 0; rels = []; ranks = []; bysnap = collections.defaultdict(list); best = collections.Counter()
    for r in live:
        t = TR[r['rid']]; s = t['scores']; o = sorted(s, key=lambda a: (-s[a], a)); lab = r['label']
        best[o[0]] += 1
        agree += lab == o[0]
        nr = (s[o[0]] - s[lab]) <= 2 * math.hypot(t['se'][o[0]], t['se'][lab]); near += nr
        rel = (s[o[0]] - s[lab]) / s[o[0]]; rels.append(rel); ranks.append(o.index(lab) + 1)
        bysnap[r['snap']].append((lab == o[0], nr, rel, o[0]))
    print('\n(a) supervised (SFT) label vs 256-draw TR optimum at the 128 non-terminal training states:')
    print('   label == TR optimum: %d/128 (%.1f%%) ; within 2 MC s.e. of optimum: %d/128 ; mean relative regret of the label %.3f ; TR rank of label: %s' % (agree, 100 * agree / 128, near, np.mean(rels), sorted(collections.Counter(ranks).items())))
    print('   TR optimum distribution at training states:', dict(best.most_common()))
    for k in sorted(bysnap):
        v = bysnap[k]
        print('   snapshot %d: label==TR-opt %d/%d ; near-opt %d ; mean rel regret %.3f ; TR-opt dist %s' % (k, sum(x[0] for x in v), len(v), sum(x[1] for x in v), np.mean([x[2] for x in v]), dict(collections.Counter(x[3] for x in v).most_common(4))))
    # uniform-random reference rel regret at training states
    print('   reference: uniform-random expected relative regret at these states: %.3f ; always-set_x_neg: %.3f' % (np.mean([np.mean([(max(TR[r['rid']]['scores'].values()) - v) / max(TR[r['rid']]['scores'].values()) for v in TR[r['rid']]['scores'].values()]) for r in live]),
          np.mean([(max(TR[r['rid']]['scores'].values()) - TR[r['rid']]['scores']['set_x_neg']) / max(TR[r['rid']]['scores'].values()) for r in live])))
    # (b) preference pairs vs TR
    NP = json.load(open('/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108/terminal02/remote/launch/cpu/worker/artifacts/cpu-phase/numeric-training-plan.json'))
    exposure = {r['record_id']: r for r in NP['exposure']['records']}
    def cmd(c): return json.loads(c).get('action_id', 'STOP')
    tot = 0; ok = 0; tot_e = 0; ok_e = 0; gaps_strict = []; gaps_tie = []; zs = []
    for r in live:
        t = TR[r['rid']]; s = t['scores']; se = t['se']
        for q in r['prefs']:
            d = abs(s[q['left']] - s[q['right']])
            if q['tie']:
                gaps_tie.append(d)
            else:
                gaps_strict.append(d); tot += 1; good = s[q['chosen']] > s[q['rejected']]; ok += good
                zs.append((s[q['chosen']] - s[q['rejected']]) / math.hypot(se[q['chosen']], se[q['rejected']]))
        for q in exposure[r['rid']]['pairs']:
            if q['exposure_count'] > 0:
                c, rj = cmd(q['chosen_command']), cmd(q['rejected_command'])
                tot_e += 1; ok_e += s[c] > s[rj]
    zs = np.array(zs)
    print('\n(b) measured preference pairs vs TR score at the same training state:')
    print('   strict pairs: %d ; preference-chosen action has the higher TR score in %d (%.1f%%) ; binomial p vs 0.5 = %.2g' % (tot, ok, 100 * ok / tot, stats.binomtest(int(ok), tot, 0.5).pvalue))
    print('   among strict pairs whose TR difference exceeds 2 MC s.e. (n=%d): agreement %.1f%%' % (int((np.abs(zs) > 2).sum()), 100 * (zs[np.abs(zs) > 2] > 0).mean()))
    print('   the 119 pairs actually exposed in DPO: preference-chosen has higher TR score in %d/%d (%.1f%%)' % (ok_e, tot_e, 100 * ok_e / tot_e))
    print('   |TR score difference| : strict pairs median %.5f mean %.5f ; tied pairs (n=%d) median %.5f mean %.5f' % (np.median(gaps_strict), np.mean(gaps_strict), len(gaps_tie), np.median(gaps_tie), np.mean(gaps_tie)))
    # by snapshot
    for k in range(4):
        tt = 0; oo = 0
        for r in live:
            if r['snap'] != k: continue
            s = TR[r['rid']]['scores']
            for q in r['prefs']:
                if not q['tie']:
                    tt += 1; oo += s[q['chosen']] > s[q['rejected']]
        print('   snapshot %d: strict pairs %d, TR agreement %.1f%%' % (k, tt, 100 * oo / tt if tt else float('nan')))
    # (c) per-state: does the preference win-count ranking correlate with the TR ranking?
    rh = []
    for r in live:
        s = TR[r['rid']]['scores']; wins = collections.Counter()
        strict = [q for q in r['prefs'] if not q['tie']]
        if len(strict) < 3: continue
        for q in strict: wins[q['chosen']] += 1; wins[q['rejected']] -= 1
        rh.append(stats.spearmanr([wins[a] for a in A], [s[a] for a in A])[0])
    rh = np.array([x for x in rh if not np.isnan(x)])
    print('\n(c) per training state with >=3 strict pairs (n=%d): Spearman rho between net preference wins and TR score across the 8 actions: mean %+.3f median %+.3f ; >0 in %.1f%%' % (len(rh), rh.mean(), np.median(rh), 100 * (rh > 0).mean()))
    # history composition: fraction of training snapshots (snap>=1) whose post-init history is a single repeated action
    mono = sum(1 for r in live if r['snap'] >= 1 and len(set(r['hist'][8:])) == 1)
    print('\n(d) training snapshots (index>=1, n=%d) whose post-initialization history is one repeated action: %d ; with >=4 distinct post-init actions: %d' % (sum(1 for r in live if r['snap'] >= 1), mono, sum(1 for r in live if r['snap'] >= 1 and len(set(r['hist'][8:])) >= 4)))
    frac_xneg = [np.mean([h == 'set_x_neg' for h in r['hist'][8:]]) for r in live if r['snap'] >= 1]
    print('    share of set_x_neg in post-init training histories: mean %.3f max %.3f  (deployment DPO: 1.000 at every decision after the first)' % (np.mean(frac_xneg), np.max(frac_xneg)))
