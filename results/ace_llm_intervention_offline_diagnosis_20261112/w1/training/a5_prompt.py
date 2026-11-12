import json, sys, statistics, collections, re
plan = json.load(open(sys.argv[1]))
def canonical(v): return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)
rows = []
for rec in plan['records']:
    raw = json.loads(rec['raw_record_utf8']); env = raw['envelope']; st = env['state']
    total = len(canonical(env))
    parts = {k: len(canonical(st[k])) for k in st}
    parts['task'] = len(canonical(env['task']))
    nums = re.findall(r'-?\d+\.\d+(?:[eE][-+]?\d+)?', canonical(env))
    rows.append(dict(total=total, parts=parts, tokens=len(rec['prompt_ids']), terminal=st['terminal'], nfloat=len(nums),
                     floatchars=sum(map(len, nums)), hist=len(st['history'])))
nt = [r for r in rows if not r['terminal']]
print('envelope chars (nonterminal): min %d median %d max %d ; prompt tokens min %d median %d max %d' % (
    min(r['total'] for r in nt), statistics.median(r['total'] for r in nt), max(r['total'] for r in nt),
    min(r['tokens'] for r in nt), statistics.median(r['tokens'] for r in nt), max(r['tokens'] for r in nt)))
keys = sorted(nt[0]['parts'])
for k in keys:
    print('  %-20s mean chars %8.1f  mean share %.3f' % (k, statistics.mean(r['parts'][k] for r in nt), statistics.mean(r['parts'][k]/r['total'] for r in nt)))
print('decimal literals per prompt: mean %.1f ; share of envelope chars that are decimal literals: %.3f' % (
    statistics.mean(r['nfloat'] for r in nt), statistics.mean(r['floatchars']/r['total'] for r in nt)))
print('history length (responses) dist at nonterminal:', sorted(collections.Counter(r['hist'] for r in nt).items()))
# common prefix of prompt token ids across nonterminal records
ps = [rec['prompt_ids'] for rec in plan['records'] if len(rec['commands']) == 9]
cp = 0
while all(len(p) > cp and p[cp] == ps[0][cp] for p in ps): cp += 1
print('common leading prompt tokens across all 128 nonterminal prompts:', cp)
cs = 0
while all(len(p) > cs and p[-1-cs] == ps[0][-1-cs] for p in ps): cs += 1
print('common trailing prompt tokens:', cs)
print('distinct prompts among 160:', len({tuple(rec['prompt_ids']) for rec in plan['records']}))
ex = json.loads(plan['records'][1]['raw_record_utf8'])['envelope']['state']
print('example scm block:', canonical(ex['scm'])[:900])
print('example history entry:', canonical(ex['history'][-1]))
print('example menu entry:', canonical(ex['menu'][1]))
print('objective:', ex['objective']); print('domain_notes:', ex['domain_notes'])
print('task:', canonical(json.loads(plan['records'][1]['raw_record_utf8'])['envelope']['task'])[:700])
