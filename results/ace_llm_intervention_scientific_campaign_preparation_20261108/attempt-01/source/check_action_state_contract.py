"""Engineering fixtures only: no independently adjudicated language gold."""
import hashlib,itertools,json,pathlib
from action_state_contract import transition,legal_commands,sequence
contract={'domains':{'latch':['open','closed'],'shaking':[False,True]},'commands':{
 'close_latch':{'requires':{'shaking':False},'effects':{'latch':'closed'}},
 'open_latch':{'requires':{'shaking':False},'effects':{'latch':'open'}},
 'start':{'requires':{'latch':'closed','shaking':False},'effects':{'shaking':True}},
 'stop':{'requires':{},'effects':{'shaking':False}}}}
state={'latch':'open','shaking':False}
assert sequence(contract,state,['close_latch','start','stop','open_latch'])['accepted']
assert sequence(contract,state,['start'])['rejected_index']==0
assert sequence(contract,state,['close_latch','start','open_latch'])['rejected_index']==2
assert state=={'latch':'open','shaking':False}
try:transition(contract,{'latch':'closed','shaking':0},'start')
except ValueError:pass
else:raise AssertionError('boolean/integer confusion')
# Exhaustively enumerate sequences through depth four and check no accepted
# transition opens a latch while shaking. Initial state is fixed and reachable.
count=0
for depth in range(5):
 for commands in itertools.product(contract['commands'],repeat=depth):
  result=sequence(contract,state,commands);count+=1
  if result['accepted']:assert not(result['state']['latch']=='open' and result['state']['shaking'])
p=pathlib.Path('results/action_state_contract_check_20261002');p.mkdir(exist_ok=False)
r={'scope':'internally authored software fixtures; zero independently adjudicated tasks','sequences_checked':count,'legal_initial_commands':legal_commands(contract,state),'checks':'passed','hardware_calls':0,'model_calls':0}
b=(json.dumps(r,indent=2)+'\n').encode();(p/'result.json').write_bytes(b);(p/'contract.json').write_text(json.dumps(contract,indent=2)+'\n');(p/'complete.json').write_text(json.dumps({'files':{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['result.json','contract.json']},'sources':{n:hashlib.sha256(pathlib.Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['action_state_contract.py','check_action_state_contract.py']}},indent=2)+'\n');print(b.decode())
