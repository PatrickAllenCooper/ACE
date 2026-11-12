"""Matched training exposure saved inside the CPU deployment custody
(cpu-phase/numeric-training-plan.json): SFT label and DPO pair distributions.
This is the exposure shared by the numerical ranker and (by design) the LLM."""
import json, collections, os
P = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108/terminal02/remote/launch/cpu/worker/artifacts/cpu-phase/numeric-training-plan.json'
p = json.load(open(P))
print('top keys:', list(p.keys()))
ex = p['exposure']
print('counts:', json.dumps(ex['counts']))
recs = ex['records']
print('records', len(recs), 'worlds', len({r['world_id'] for r in recs}))
print('record keys:', sorted(recs[0].keys()))
print('pair keys:', sorted(recs[0]['pairs'][0].keys()))
def cmd(c):
    if c is None: return None
    d = json.loads(c)
    return d.get('action_id', 'STOP')
sft = collections.Counter(cmd(r['supervised_command']) for r in recs)
print('\nSFT supervised_command distribution over 160 records:', dict(sft.most_common()))
print('SFT exposure counts:', collections.Counter(r['sft_exposure_count'] for r in recs))
# by snapshot index
bysnap = collections.defaultdict(collections.Counter)
ncand = collections.defaultdict(collections.Counter)
for r in recs:
    snap = int(r['record_id'].rsplit('-', 1)[1])
    bysnap[snap][cmd(r['supervised_command'])] += 1
    ncand[snap][len(r['action_ids'])] += 1
for s in sorted(bysnap):
    print('  snapshot', s, 'n_candidates', dict(ncand[s]), 'labels', dict(bysnap[s].most_common()))
print('action_ids example:', recs[0]['action_ids'], '| snapshot-004 example:', recs[4]['action_ids'], recs[4]['supervised_command'], 'pairs', len(recs[4]['pairs']))
# pairs
allp = [q for r in recs for q in r['pairs']]
print('\npairs total', len(allp))
print('pair example strict:', json.dumps(next(q for q in allp if q['chosen_command'] is not None))[:600])
strict = [q for q in allp if q['chosen_command'] is not None]
print('strict pairs', len(strict), 'exposed (exposure_count>0):', sum(q['exposure_count'] > 0 for q in allp), 'total exposures', sum(q['exposure_count'] for q in allp))
ch_all = collections.Counter(cmd(q['chosen_command']) for q in strict)
rj_all = collections.Counter(cmd(q['rejected_command']) for q in strict) if 'rejected_command' in strict[0] else None
print('ALL strict pairs: chosen', dict(ch_all.most_common()))
print('ALL strict pairs: rejected', dict(rj_all.most_common()) if rj_all else 'n/a')
exp = [q for q in allp if q['exposure_count'] > 0]
ch = collections.Counter(); rj = collections.Counter()
for q in exp:
    ch[cmd(q['chosen_command'])] += q['exposure_count']
    rj[cmd(q['rejected_command'])] += q['exposure_count']
print('EXPOSED DPO pairs (119): chosen', dict(ch.most_common()))
print('EXPOSED DPO pairs (119): rejected', dict(rj.most_common()))
A = ['observe_all', 'set_x_neg', 'set_x_zero', 'set_x_pos', 'set_y_neg', 'set_y_zero', 'set_y_pos', 'set_xy']
print('\nnet (chosen - rejected) among exposed DPO pairs, and among all strict pairs:')
for a in A:
    print('  %-11s exposed: +%d / -%d = %+d    all strict: +%d / -%d = %+d' % (a, ch[a], rj[a], ch[a] - rj[a], ch_all[a], rj_all[a], ch_all[a] - rj_all[a]))
# win rate of each action in strict pairs
print('\nper-action strict-pair win rate (wins / strict pairs involving the action), all 687 strict pairs:')
for a in A:
    inv = ch_all[a] + rj_all[a]
    print('  %-11s %d/%d = %.3f' % (a, ch_all[a], inv, ch_all[a] / inv if inv else float('nan')))
# records by world cost menu? (not in plan) -> world ids
print('\nworld ids:', sorted({r['world_id'] for r in recs})[:3], '...', sorted({r['world_id'] for r in recs})[-1])
# schedule
sch = ex['numeric_schedule']
print('schedule stages:', collections.Counter(s['stage'] for s in sch))
rid2 = {r['record_id']: r for r in recs}
sft_idx = collections.Counter(rid2[s['record_id']]['action_ids'][s['chosen_index']] if rid2[s['record_id']]['action_ids'][s['chosen_index']] else 'STOP' for s in sch if s['stage'] == 'sft')
print('SFT schedule label distribution (160 updates):', dict(sft_idx.most_common()))
dpo_c = collections.Counter(); dpo_r = collections.Counter()
for s in sch:
    if s['stage'] != 'sft':
        ids = rid2[s['record_id']]['action_ids']
        dpo_c[ids[s['chosen_index']] or 'STOP'] += 1; dpo_r[ids[s['rejected_index']] or 'STOP'] += 1
print('DPO schedule chosen (119 updates):', dict(dpo_c.most_common()))
print('DPO schedule rejected (119 updates):', dict(dpo_r.most_common()))
pairsdir = collections.Counter()
for s in sch:
    if s['stage'] != 'sft':
        ids = rid2[s['record_id']]['action_ids']
        pairsdir[(ids[s['chosen_index']], ids[s['rejected_index']])] += 1
print('DPO (chosen > rejected) pair types:', sorted(pairsdir.items(), key=lambda kv: -kv[1])[:40])
