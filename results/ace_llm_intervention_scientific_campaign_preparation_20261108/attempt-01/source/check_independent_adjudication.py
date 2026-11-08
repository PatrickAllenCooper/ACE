"""Internal review-pipeline tests; mock reviews never become gold."""
import copy,json,pathlib
from independent_adjudication import validate,reconcile
p=pathlib.Path('results/temporal_independent_review_20261003');blind=json.loads((p/'blind_review.json').read_text());template=json.loads((p/'review_template.json').read_text())
def mock(identity,kind):
 r=copy.deepcopy(template);r.update(reviewer_id=identity,review_kind=kind,independence_attestation=True,robot_api_scope='mock scope',submitted_at='mock timestamp')
 for l in r['labels']:l.update(decision='unresolved',source_span_sha256=[blind['review_input']['spans'][0]['sha256']],rationale='Mock unresolved test')
 d={'designation_status':'assigned','designated_by':'test harness','reviewer_id':identity,'review_kind':kind,'author_ids':['author']}
 return r,d
r,d=mock('mock1','automated');s,e=mock('mock2','human');out=reconcile([r,s],blind,[d,e]);assert not out['gold_released'] and len(out['unresolved_sequences'])==78
for mutate in ['author','hash','coverage']:
 rr,dd=copy.deepcopy(r),copy.deepcopy(d)
 if mutate=='author':dd['author_ids']=['mock1']
 elif mutate=='hash':rr['packet_sha256']='wrong'
 else:rr['labels'].pop()
 try:validate(rr,blind,dd)
 except ValueError:pass
 else:raise AssertionError(mutate)
assert 'context_author' not in blind['review_input'] and all('private_gold' not in t for t in blind['review_input']['tasks'])
(p/'checks.json').write_text(json.dumps({'checks':'passed','scope':'mock pipeline checks only','actual_reviews':0,'gold_released':False,'candidate_sequences':78},indent=2)+'\n')
print('Author conflict, packet binding, missing coverage, blinding and unresolved-release checks passed')
