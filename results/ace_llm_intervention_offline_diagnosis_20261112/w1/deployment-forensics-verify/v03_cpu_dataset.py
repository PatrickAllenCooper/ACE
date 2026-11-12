"""Build decision table for the five CPU arms from decision-NNNN.{input,process}.json. Verifier's own code."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import pickle, collections, re
camp = json.load(open(CAMP + '/campaign.json'))
pids = camp['policy_ids']
rows = []
problems = []
for p in range(5):
    arm = pids[p]
    for w in range(24):
        ep = f'{CAMP}/policy-{p:02d}/world-{w:02d}/episode'
        files = sorted(os.listdir(ep))
        dec_inputs = sorted(f for f in files if re.fullmatch(r'decision-\d{4}\.input\.json', f))
        dec_procs = sorted(f for f in files if re.fullmatch(r'decision-\d{4}\.process\.json', f))
        if len(dec_inputs) != len(dec_procs):
            problems.append(('count mismatch', arm, w, len(dec_inputs), len(dec_procs)))
        term = json.load(open(ep + '/terminal.json'))
        for k, f in enumerate(dec_inputs):
            idx = int(f[9:13])
            if idx != k: problems.append(('index gap', arm, w, f))
            env = json.load(open(f'{ep}/{f}'))
            proc = json.load(open(f'{ep}/decision-{idx:04d}.process.json'))
            out = json.loads(proc['stdout'])
            try:
                meta = json.loads(proc['stderr'])
            except Exception as e:
                meta = None; problems.append(('stderr', arm, w, idx, proc['stderr'][:200]))
            if proc['returncode'] != 0: problems.append(('rc', arm, w, idx, proc['returncode']))
            esha = digest(env)
            if esha != proc['input_sha256']: problems.append(('input sha', arm, w, idx))
            action = 'STOP' if out.get('stop') is True else out['action_id']
            rows.append(dict(arm=arm, world=w, world_id=camp['sources'][w]['plan']['world']['world_id'],
                             d=idx, envelope=env, envelope_sha256=esha, action=action, meta=meta,
                             elapsed=proc['elapsed_seconds'],
                             terminal_status=term['status'], policy_decisions=term['policy_decisions'],
                             final_remaining=term['final_state']['remaining_units']))
print('rows', len(rows), 'problems', len(problems))
for pr in problems[:20]: print(pr)
by = collections.Counter(r['arm'] for r in rows)
print('decisions per arm (files):', dict(by))
for arm in pids[:5]:
    c = collections.Counter(r['action'] for r in rows if r['arm'] == arm)
    print(arm, {SHORT[k]: v for k, v in sorted(c.items(), key=lambda kv: -kv[1])}, 'acquired', sum(v for k, v in c.items() if k != 'STOP'))
# per (arm, world) decision counts and terminal info
for arm in pids[:5]:
    per = collections.Counter((r['world']) for r in rows if r['arm'] == arm)
    pd = {r['world']: (r['policy_decisions'], r['final_remaining'], r['terminal_status']) for r in rows if r['arm'] == arm}
    print(arm, 'files per world', [per[w] for w in range(24)])
    print('   terminal policy_decisions', [pd[w][0] for w in range(24)], 'final_remaining', [pd[w][1] for w in range(24)], set(pd[w][2] for w in range(24)))
pickle.dump(rows, open(SCR + '/cpu_rows.pkl', 'wb'))
