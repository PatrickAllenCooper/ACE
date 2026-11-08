"""Finite typed state transitions for offline action-contract evaluation.

Formal rules are evaluator input. Supplying them to a runtime policy requires
an exact-enumeration baseline. This module executes no hardware commands.
"""
from copy import deepcopy

def equal(a,b):
    return type(a) is type(b) and a == b

def audit(contract):
    if set(contract) != {'domains','commands'}:
        raise ValueError('contract requires domains and commands')
    domains,commands=contract['domains'],contract['commands']
    if not isinstance(domains,dict) or not domains or not isinstance(commands,dict):
        raise ValueError('invalid domains/commands')
    for name,values in domains.items():
        if not isinstance(name,str) or not isinstance(values,list) or not values:
            raise ValueError('finite state domains required')
        if any(type(v) not in (str,bool,int) for v in values):
            raise ValueError('only string, boolean, integer states supported')
        if any(equal(v,w) for i,v in enumerate(values) for w in values[:i]):
            raise ValueError('duplicate domain value')
    for name,cmd in commands.items():
        if not isinstance(name,str) or set(cmd)!= {'requires','effects'}:
            raise ValueError('malformed command')
        for part in ['requires','effects']:
            if not isinstance(cmd[part],dict):raise ValueError('malformed predicate')
            for key,val in cmd[part].items():
                if key not in domains or not any(equal(val,v) for v in domains[key]):
                    raise ValueError('undeclared state/value')

def transition(contract,state,command):
    audit(contract)
    if not isinstance(state,dict) or set(state)!=set(contract['domains']):
        raise ValueError('complete declared state required')
    for key,val in state.items():
        if not any(equal(val,v) for v in contract['domains'][key]):
            raise ValueError('invalid typed state')
    if not isinstance(command,str) or command not in contract['commands']:
        raise ValueError('unknown command')
    rule=contract['commands'][command]
    if any(not equal(state[k],v) for k,v in rule['requires'].items()):
        return False,deepcopy(state)
    result=deepcopy(state);result.update(deepcopy(rule['effects']))
    return True,result

def legal_commands(contract,state):
    audit(contract)
    return [c for c in sorted(contract['commands']) if transition(contract,state,c)[0]]

def sequence(contract,state,commands):
    current=deepcopy(state)
    for i,command in enumerate(commands):
        legal,next_state=transition(contract,current,command)
        if not legal:return {'accepted':False,'rejected_index':i,'state':current}
        current=next_state
    return {'accepted':True,'rejected_index':None,'state':current}
