import json,sys
data=json.load(sys.stdin)
assert set(data)=={'version','state','task'}
assert 'mechanisms' not in data['state'] and 'private_evaluation' not in data
state=data['state']
kind=sys.argv[1]
if kind=='broken': print('not JSON')
elif kind=='stop': print('{"stop":true}')
elif kind=='costly': print(json.dumps({'action_id':'costly' if state['remaining_units']>=2 else 'cheap'}))
else: print('{"action_id":"observe"}')
