"""Training exposure stored beside the deployment data (numeric-training-plan.json). Verifier's own code."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
from vtr import *
import collections, pickle
P = CPU + '/worker/artifacts/cpu-phase/numeric-training-plan.json'
d = json.load(open(P))
ex = d['exposure']
recs = {r['record_id']: r for r in ex['records']}
raws = {k: json.loads(v) for k, v in d['raw_records_utf8'].items()}
print('records', len(recs), 'raw', len(raws), 'worlds', len({r['world_id'] for r in recs.values()}))
# supervised labels
lab = collections.Counter()
term_stop = 0; only_stop = 0
levels = collections.Counter(); levels_by_snap = collections.defaultdict(collections.Counter)
lab_by_snap = collections.defaultdict(collections.Counter)
for rid, r in recs.items():
    raw = raws[rid]; st = raw['envelope']['state']
    sup = raw.get('supervised_action_id')
    cmd = json.loads(r['supervised_command'])
    a = 'STOP' if cmd.get('stop') is True else cmd['action_id']
    lab[a] += 1
    snap = rid.split('/')[-1]
    levels[st['remaining_units']] += 1; levels_by_snap[snap][st['remaining_units']] += 1
    lab_by_snap[snap][SHORT[a]] += 1
    if a == 'STOP':
        term_stop += st['terminal'] or st['remaining_units'] == 0
        only_stop += r['action_ids'] == [None]
print('SFT labels', {SHORT[k]: v for k, v in lab.most_common()})
print('stop labels on terminal/zero-budget states', term_stop, '; with stop the only legal command', only_stop)
print('remaining_units levels across 160 snapshots', sorted(levels.items()))
for s in sorted(levels_by_snap): print('  ', s, 'remaining', dict(levels_by_snap[s]), 'labels', dict(lab_by_snap[s]))
# sft schedule
sched = ex['schedule']
print('schedule stages', collections.Counter(s['stage'] for s in sched))
# pairs
st_counter = collections.Counter(); exposed = []
for rid, r in recs.items():
    for p in r['pairs']:
        st_counter[p['status']] += 1
        if p['exposure_count'] > 0: exposed.append((rid, p))
print('pair statuses', dict(st_counter), 'total', sum(st_counter.values()), 'exposed', len(exposed))
ch = collections.Counter(); rj = collections.Counter()
for rid, p in exposed:
    c = json.loads(p['chosen_command']); rr = json.loads(p['rejected_command'])
    ch['STOP' if c.get('stop') else c['action_id']] += 1; rj['STOP' if rr.get('stop') else rr['action_id']] += 1
print('exposed chosen', {SHORT[k]: v for k, v in ch.most_common()})
print('exposed rejected', {SHORT[k]: v for k, v in rj.most_common()})
net = {a: ch[a] - rj[a] for a in set(ch) | set(rj)}
print('exposed net (chosen - rejected)', {SHORT[k]: v for k, v in sorted(net.items(), key=lambda kv: -kv[1])})
# all strict pairs: chosen/rejected counts
ch2 = collections.Counter(); rj2 = collections.Counter()
for rid, r in recs.items():
    for p in raws[rid]['preferences']:
        if not p['tie']:
            ch2[p['chosen']] += 1; rj2[p['rejected']] += 1
print('strict chosen', {SHORT.get(k, k): v for k, v in ch2.most_common()}); print('strict rejected', {SHORT.get(k, k): v for k, v in rj2.most_common()})
# TR reference on non-terminal training states
rng = np.random.default_rng(424242)
agree_lab = 0; n_lab = 0; strict_agree = 0; strict_n = 0; exp_agree = 0; exp_n = 0
ref = {}
with np.errstate(all='ignore'):
    for rid, r in recs.items():
        env = raws[rid]['envelope']
        if env['state']['terminal'] or not any(a['cost_units'] <= env['state']['remaining_units'] for a in env['state']['menu']):
            continue
        before, out = vec_scores(env, 16384, rng)
        ref[rid] = out
        best = min(out, key=lambda a: (-out[a][0], a))
        cmd = json.loads(r['supervised_command']); a = 'STOP' if cmd.get('stop') is True else cmd['action_id']
        n_lab += 1; agree_lab += a == best
        for p in raws[rid]['preferences']:
            if not p['tie'] and p['chosen'] in out and p['rejected'] in out:
                strict_n += 1; strict_agree += out[p['chosen']][0] > out[p['rejected']][0]
        for p in r['pairs']:
            if p['exposure_count'] > 0:
                c = json.loads(p['chosen_command'])['action_id']; rr = json.loads(p['rejected_command'])['action_id']
                exp_n += 1; exp_agree += out[c][0] > out[rr][0]
print('non-terminal training states', n_lab, '| SFT label == TR(16384-draw) optimum:', agree_lab)
print('strict pairs with both actions scored', strict_n, 'agree with TR ordering', strict_agree, '%.3f' % (strict_agree / strict_n))
print('exposed pairs', exp_n, 'agree with TR ordering', exp_agree, '%.3f' % (exp_agree / exp_n))
# TR optimum distribution on training states and by snapshot
opt = collections.Counter(); opt_by_snap = collections.defaultdict(collections.Counter)
for rid, out in ref.items():
    b = min(out, key=lambda a: (-out[a][0], a)); opt[SHORT[b]] += 1; opt_by_snap[rid.split('/')[-1]][SHORT[b]] += 1
print('TR optimum on training states', dict(opt))
for s in sorted(opt_by_snap): print('  ', s, dict(opt_by_snap[s]))
pickle.dump(ref, open(SCR + '/train_ref.pkl', 'wb'))
