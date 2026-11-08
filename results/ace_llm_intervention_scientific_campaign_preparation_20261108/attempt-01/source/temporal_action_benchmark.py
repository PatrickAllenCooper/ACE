"""Strict offline sequence scorer; gold contract stays evaluator-private."""
import hashlib,itertools,json
from action_state_contract import audit,sequence,transition

PROVENANCE={'publisher','source_url','source_sha256','source_group','context_author','gold_adjudicator','adjudication_status'}
def key(value):return json.dumps(value,separators=(',',':'),sort_keys=True)

def audit_task(task):
    if set(task)!={'id','description','provenance','contract','initial_state','candidate_sequences','gold_valid_sequences','ambiguous'}:
        raise ValueError('invalid task fields')
    if not isinstance(task['id'],str) or not task['id'] or not isinstance(task['description'],str) or not task['description'].strip():
        raise ValueError('id/description required')
    p=task['provenance']
    if not isinstance(p,dict) or set(p)!=PROVENANCE or any(not isinstance(v,str) or not v.strip() for v in p.values()):
        raise ValueError('complete source and adjudication provenance required')
    if len(p['source_sha256'])!=64 or any(c not in '0123456789abcdef' for c in p['source_sha256']):
        raise ValueError('source digest required')
    if p['context_author']==p['gold_adjudicator']:
        raise ValueError('separate adjudicator required')
    if p['adjudication_status']!='verified':raise ValueError('unverified task cannot enter benchmark')
    if type(task['ambiguous']) is not bool:raise ValueError('ambiguous must be boolean')
    audit(task['contract'])
    candidates=task['candidate_sequences']
    if not isinstance(candidates,list) or not candidates or len(candidates)>1000:raise ValueError('bounded candidates required')
    for cmds in candidates:
        if not isinstance(cmds,list) or not 1<=len(cmds)<=8 or any(not isinstance(c,str) for c in cmds):raise ValueError('invalid sequence')
        for c in cmds:
            if c not in task['contract']['commands']:raise ValueError('undeclared candidate command')
    candidate_keys={key(c) for c in candidates}
    if len(candidate_keys)!=len(candidates):raise ValueError('duplicate candidate')
    if task['ambiguous']:
        if task['gold_valid_sequences'] is not None:raise ValueError('ambiguous task must have no definitive gold')
        # Validate initial state even when ambiguity forbids definitive scoring.
        transition(task['contract'],task['initial_state'],next(iter(task['contract']['commands'])))
        return None,candidate_keys
    gold=task['gold_valid_sequences']
    if not isinstance(gold,list):raise ValueError('gold list required')
    gold_keys={key(c) for c in gold}
    exact={key(c) for c in candidates if sequence(task['contract'],task['initial_state'],c)['accepted']}
    if len(gold_keys)!=len(gold) or gold_keys!=exact:raise ValueError('gold does not match complete candidate enumeration')
    return exact,candidate_keys

def score(task,proposal):
    gold,candidates=audit_task(task)
    if not isinstance(proposal,dict) or set(proposal)!={'id','abstain','sequences'} or proposal['id']!=task['id']:
        raise ValueError('strict proposal fields/id required')
    if type(proposal['abstain']) is not bool or not isinstance(proposal['sequences'],list):raise ValueError('invalid proposal types')
    if proposal['abstain'] and proposal['sequences']:raise ValueError('abstention must have empty sequences')
    for cmds in proposal['sequences']:
        if not isinstance(cmds,list) or any(not isinstance(c,str) for c in cmds):raise ValueError('invalid proposed sequence')
    keys=[key(c) for c in proposal['sequences']];selected=set(keys)
    if gold is None:return {'ambiguity_abstention_correct':proposal['abstain'],'exact_menu':None,'false_legal':None}
    false=len(selected-gold)
    return {'ambiguity_abstention_correct':None,'exact_menu':not proposal['abstain'] and selected==gold and len(keys)==len(selected),
            'false_legal':false,'out_of_candidates':len(selected-candidates),'duplicates':len(keys)-len(selected),
            'recall':len(selected&gold)/len(gold) if gold else 1.,'abstained':proposal['abstain']}

def public_task(task):
    """Only independently authored description, context and command candidates.

    initial_state is declared public task context. Formal preconditions/effects
    and complete legal-menu gold are never included in this projection.
    """
    audit_task(task)
    return {k:task[k] for k in ['id','description','initial_state','candidate_sequences']}
