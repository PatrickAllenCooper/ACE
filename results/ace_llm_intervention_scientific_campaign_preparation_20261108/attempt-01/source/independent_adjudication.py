"""Blinded review custody and agreement checks; no automatic gold promotion.

Reviewer identity/designation must be established outside this local file API.
Hashes bind content; they do not authenticate a human or prove independence.
"""
import argparse,hashlib,json,pathlib

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def prepare(packet):
    for span in packet['spans']:
        if hashlib.sha256(span['text'].encode()).hexdigest()!=span['sha256']:raise ValueError('source span hash mismatch')
    public={k:packet[k] for k in ['publisher','source_group','source_url','source_sha256','spans','robot_context','command_bindings']}
    public['tasks']=[{k:t[k] for k in ['id','initial_state','candidate_sequences']} for t in packet['tasks']]
    return {'packet_sha256':digest(public),'review_input':public,'instructions':'Label every candidate valid, invalid, or unresolved. Cite retained source spans. Do not infer missing semantics. Record robot/API scope. No author draft or model answer is supplied.'}

def validate(review,blinded,designation):
    if designation.get('designation_status')!='assigned' or not designation.get('designated_by'):
        raise ValueError('explicit reviewer designation required')
    if review.get('reviewer_id')!=designation.get('reviewer_id') or review.get('packet_sha256')!=blinded['packet_sha256']:
        raise ValueError('identity/packet mismatch')
    if designation['reviewer_id'] in designation.get('author_ids',[]):raise ValueError('author cannot adjudicate own contexts')
    if review.get('review_kind') not in ['human','automated']:raise ValueError('review kind required')
    if review['review_kind']!=designation.get('review_kind'):raise ValueError('review kind mismatch')
    if review.get('independence_attestation') is not True:raise ValueError('independence attestation required')
    if not review.get('robot_api_scope') or not review.get('submitted_at'):raise ValueError('scope/timestamp required')
    tasks=blinded['review_input']['tasks'];expected={(t['id'],digest(c)) for t in tasks for c in t['candidate_sequences']}
    spans={s['sha256'] for s in blinded['review_input']['spans']};seen=set()
    for label in review.get('labels',[]):
        k=(label.get('task_id'),label.get('sequence_sha256'))
        if k not in expected or k in seen:raise ValueError('unknown/duplicate sequence')
        seen.add(k)
        if label.get('decision') not in ['valid','invalid','unresolved']:raise ValueError('invalid decision')
        refs=label.get('source_span_sha256',[])
        if not isinstance(refs,list) or not refs or any(r not in spans for r in refs):raise ValueError('retained source evidence required')
        if not isinstance(label.get('rationale'),str) or not label['rationale'].strip():raise ValueError('rationale required')
    if seen!=expected:raise ValueError('complete candidate coverage required')
    return {k:v for k,v in review.items()}

def reconcile(reviews,blinded,designations):
    checked=[validate(r,blinded,d) for r,d in zip(reviews,designations)]
    if len(reviews)!=len(designations) or len(checked)<2:raise ValueError('two designated reviews required')
    if len({r['reviewer_id'] for r in checked})!=len(checked):raise ValueError('reviewers must differ')
    maps=[{(l['task_id'],l['sequence_sha256']):l['decision'] for l in r['labels']} for r in checked]
    unresolved=[];agreement=[]
    for k in maps[0]:
        values=[m[k] for m in maps]
        if len(set(values))>1 or 'unresolved' in values:unresolved.append(k)
        else:agreement.append({'task_id':k[0],'sequence_sha256':k[1],'decision':values[0]})
    return {'status':'pending_human_gold_release','packet_sha256':blinded['packet_sha256'],'review_hashes':[digest(r) for r in checked],'agreed_labels':agreement,'unresolved_sequences':unresolved,'human_reviews':sum(r['review_kind']=='human' for r in checked),'automated_reviews':sum(r['review_kind']=='automated' for r in checked),'gold_released':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--packet',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    result=prepare(json.loads(a.packet.read_text()));a.output.mkdir(exist_ok=False,parents=True)
    (a.output/'blind_review.json').write_text(json.dumps(result,indent=2)+'\n')
    template={'reviewer_id':None,'packet_sha256':result['packet_sha256'],'review_kind':None,'independence_attestation':False,'robot_api_scope':None,'submitted_at':None,'labels':[{'task_id':t['id'],'sequence_sha256':digest(c),'sequence':c,'decision':None,'source_span_sha256':[],'rationale':None} for t in result['review_input']['tasks'] for c in t['candidate_sequences']]}
    (a.output/'review_template.json').write_text(json.dumps(template,indent=2)+'\n')
    (a.output/'designation_template.json').write_text(json.dumps({'designation_status':'unassigned','designated_by':None,'reviewer_id':None,'review_kind':None,'author_ids':['ACE preparation agent'],'identity_verification':'requires actual reviewer designation; file fields alone are not identity proof'},indent=2)+'\n')
if __name__=='__main__':main()
