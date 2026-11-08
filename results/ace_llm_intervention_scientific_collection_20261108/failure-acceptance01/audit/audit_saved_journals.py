"""Future local saved-journal census only. Never emits acceptance or executes project code.
Run only after complete custody is independently authorized and byte-verified.
Every invocation requires a NEW attempt directory; failures retain partial findings.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import stat
import time

HERE = Path(__file__).resolve().parent

def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(b):
    return hashlib.sha256(b).hexdigest()

def decode(b):
    def pairs(rows):
        v = {}
        for k, x in rows:
            if k in v:
                raise ValueError('duplicate JSON key')
            v[k] = x
        return v
    def reject(_):
        raise ValueError('nonfinite JSON')
    v = json.loads(b, object_pairs_hook=pairs, parse_constant=reject)
    canonical(v)
    return v

def read(root, rel, maximum=64*1024**2):
    q = Path(rel)
    if q.is_absolute() or '..' in q.parts:
        raise ValueError('unsafe relative path')
    p = root
    for part in q.parts:
        p = p / part
        if p.is_symlink():
            raise ValueError('symlink in evidence path')
    with os.fdopen(os.open(p, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK), 'rb') as f:
        before = os.fstat(f.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError('bounded regular evidence required')
        b = f.read(maximum+1)
        after = os.fstat(f.fileno())
    identity = lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if len(b)>maximum or identity(before)!=identity(after) or identity(after)!=identity(p.lstat()):
        raise ValueError('evidence changed while read')
    return b

def save(path,v):
    with path.open('x') as f:
        json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
        f.flush();os.fsync(f.fileno())

def census(root, world):
    """Independent serialization/accounting checks, no learner or label recomputation."""
    rows=[];errors=[];files={};events=collections.Counter();raw_names=set()
    for entry in world['journal_plan']:
        rel=world['path']+'/'+entry['path']; reservations={};known=set();accepted=set()
        previous=None;last_time=None;terminated=False
        try:
            b=read(root,rel);files[rel]={'bytes':len(b),'sha256':digest(b)}
            if not b.endswith(b'\n'): errors.append({'path':rel,'error':'truncated journal; all readable reservations still charged'})
            lines=b.splitlines()
            for i,line in enumerate(lines):
                if not b.endswith(b'\n') and i==len(lines)-1:
                    break
                e=decode(line);pin=e.pop('sha256')
                if e['previous_sha256']!=previous or digest(canonical(e))!=pin:
                    raise ValueError('journal chain mismatch at line '+str(i+1))
                t=e['monotonic_ns']
                if type(t) is not int or (last_time is not None and t<last_time):
                    raise ValueError('invalid journal clock')
                previous=pin;last_time=t;kind=e['event'];events[kind]+=1
                if i==0 and kind!='begin': raise ValueError('missing initial begin')
                if terminated: raise ValueError('event after closed')
                if kind=='closed': terminated=True
                request=e.get('request_id')
                if kind=='reserve':
                    if type(request) is not int or request<0 or request in reservations:
                        raise ValueError('duplicate/invalid reservation')
                    cost=e['cost_units'];action=e['action']['action_id']
                    if type(cost) is not int or cost<=0: raise ValueError('invalid reservation charge')
                    reservations[request]={'journal':entry['path'],'kind':entry['kind'],
                        'request_id':request,'action_id':action,'cost_units':cost}
                    menu=world['queries'] if entry['kind']=='scoring' else world['menu']
                    costs={a['action_id']:a['cost_units'] for a in menu}
                    if costs.get(action)!=cost: errors.append({'path':rel,'error':'reservation action/cost differs from frozen public menu','request_id':request})
                    rp=str(Path(rel).with_name(Path(rel).stem+f'.return-{request:03d}.json'))
                    raw_names.add(rp)
                    try:
                        rb=read(root,rp);rv=decode(rb);files[rp]={'bytes':len(rb),'sha256':digest(rb)}
                        if rv['request_id']!=request or rv['journal']!=Path(rel).name or rv['returned_to_wrapper'] is not True:
                            raise ValueError('return binding differs')
                        known.add(request)
                    except FileNotFoundError:
                        errors.append({'path':rp,'error':'missing original raw return'})
                    except (ValueError,KeyError,TypeError,OSError) as exc:
                        errors.append({'path':rp,'error':type(exc).__name__})
                elif kind=='response':
                    if request not in reservations or request in accepted:
                        raise ValueError('response without unique reservation')
                    accepted.add(request);known.add(request)
                elif kind=='failure':
                    if request not in reservations: raise ValueError('failure without reservation')
                    if e.get('response_returned_to_driver') is True: known.add(request)
                    errors.append({'path':rel,'error':'saved executor failure','request_id':request})
                elif kind in ('driver_failure','rejected'):
                    errors.append({'path':rel,'error':'saved '+kind})
            if not terminated: errors.append({'path':rel,'error':'missing closed event'})
        except (ValueError,KeyError,TypeError,OSError) as exc:
            errors.append({'path':rel,'error':type(exc).__name__+': '+str(exc)})
        # Keep every readable charge even on malformed/failed/unfinished journals.
        for request,v in reservations.items():
            rows.append({**v,'known_return':request in known,'executor_accepted':request in accepted})
    totals={}
    for kind in ('trajectory','branch','scoring'):
        subset=[r for r in rows if r['kind']==kind]
        totals[kind]={'reserved_cost_units':sum(r['cost_units'] for r in subset),
            'reserved_response_attempts':len(subset),'known_returns':sum(r['known_return'] for r in subset),
            'executor_accepted_responses':sum(r['executor_accepted'] for r in subset),
            'unknown_return_attempts':sum(not r['known_return'] for r in subset),
            'returned_without_acceptance':sum(r['known_return'] and not r['executor_accepted'] for r in subset),
            'unknown_return_reserved_cost':sum(r['cost_units'] for r in subset if not r['known_return'])}
    total={k:sum(v[k] for v in totals.values()) for k in ('reserved_response_attempts','known_returns','executor_accepted_responses','unknown_return_attempts')}
    total['total_reserved_cost_units']=sum(v['reserved_cost_units'] for v in totals.values())
    return dict(world_id=world['world_id'],path=world['path'],by_kind=totals,accounting=total,
        reservations=rows,event_counts=dict(events),errors=errors,counts_are_lower_bounds=bool(errors),
        file_pins=files,expected_return_paths=sorted(raw_names))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collection-root',required=True)
    ap.add_argument('--expected-schedule-sha256',required=True)
    ap.add_argument('--attempt-directory',required=True)
    args=ap.parse_args();out=Path(args.attempt_directory)
    # NEW audit output only; never create or mutate source custody directories.
    root=Path(args.collection_root).resolve(strict=True)
    if not root.is_dir() or not out.is_absolute() or root==out.resolve() or root in out.resolve().parents:
        raise ValueError('independent absolute attempt directory required')
    out.mkdir(mode=0o700,exist_ok=False)
    started=time.monotonic();cpu=time.process_time();report=dict(acceptance=False,status='audit_incomplete',worlds=[])
    try:
        raw=read(HERE,'expected-public-schedule.json')
        if digest(raw)!=args.expected_schedule_sha256: raise ValueError('independent schedule pin differs')
        schedule=decode(raw)
        save(out/'invocation.json',dict(collection_root=str(root),schedule_sha256=digest(raw),
            audit_source_sha256=digest(read(HERE,Path(__file__).name)),acceptance=False))
        # Opaque inventory names only; no private descriptor or feature decoding.
        observed_journals=[];observed_returns=[];failure_files=[]
        for p in sorted(root.rglob('*')):
            if p.is_symlink(): raise ValueError('symlink in original collection')
            if not p.is_file(): continue
            rel=p.relative_to(root).as_posix()
            if p.suffix=='.jsonl': observed_journals.append(rel)
            if '.return-' in p.name and p.suffix=='.json': observed_returns.append(rel)
            if 'failure' in p.name or 'interruption' in p.name: failure_files.append(rel)
        for world in schedule['worlds']:
            row=census(root,world);save(out/(world['path']+'.json'),row);report['worlds'].append(row)
        totals={k:sum(w['accounting'][k] for w in report['worlds']) for k in report['worlds'][0]['accounting']}
        expected_journals={w['path']+'/'+e['path'] for w in schedule['worlds'] for e in w['journal_plan']}
        expected_returns={r for w in report['worlds'] for r in w['expected_return_paths']}
        report.update(status='journal_census_only_review_pending',accounting=totals,
            saved_failure_files=failure_files,other_jsonl_for_adapter_audit=sorted(set(observed_journals)-expected_journals),
            missing_executor_journals=sorted(expected_journals-set(observed_journals)),
            unplanned_raw_returns=sorted(set(observed_returns)-expected_returns),
            errors=sum((w['errors'] for w in report['worlds']),[]),
            counts_are_lower_bounds=any(w['counts_are_lower_bounds'] for w in report['worlds']),
            remaining_gates=['custody','records_and_pairs','adapter_lineage','runtime','scheduler','cleanup','review'])
    except BaseException as exc:
        report.update(status='audit_incomplete',exception_type=type(exc).__name__,message=str(exc))
        raise
    finally:
        report.update(process_cpu_seconds=time.process_time()-cpu,elapsed_seconds=time.monotonic()-started)
        # Detailed world evidence is already separate; avoid duplicating every journal pin.
        report['worlds']=[{'world_id':w['world_id'],'path':w['path'],'accounting':w['accounting'],'errors':len(w['errors'])} for w in report['worlds']]
        save(out/'journal-census.json',report)

if __name__=='__main__':main()
