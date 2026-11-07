import json,sys
v=json.load(sys.stdin)
assert set(v)=={'version','task','state'}
assert 'mechanisms' not in v['state'] and 'private_evaluation' not in v
mode=sys.argv[1]
assert int(sys.argv[2]) in (51,52)
print('broken' if mode=='broken' else '{"stop":true}' if mode=='stop' else '{"action_id":"observe"}')
