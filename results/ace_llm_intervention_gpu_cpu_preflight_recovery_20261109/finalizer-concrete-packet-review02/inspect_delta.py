"""Independent read-only byte/metadata delta; no producer imports or execution."""
from pathlib import Path
import datetime, hashlib, json, os, resource, stat, time, traceback

REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
PREP=REPO/'results/ace_llm_intervention_gpu_cpu_preflight_recovery_20261109'
HERE=Path(__file__).resolve().parent
ORIGINAL=REPO/'results/ace_llm_intervention_scientific_finalization_20261108/launch01'
ORIGINAL_EXT=BASE/'ACE-LLM-intervention-scientific-finalization-20261108-01'
NEW=ORIGINAL.with_name('launch02')
NEW_EXT=BASE/'ACE-LLM-intervention-scientific-finalization-20261108-02'
PUB=PREP/'finalizer-operational-publication01'
OLD_SHA='1a30c9bdfef0b81f98089c774efa5ff781ddc2b5bc8411c9d08e3b198adb87d3'
NEW_SHA='dcd34d5c040b7a9bf2769736a8ee5fa103090c4ef1af3f7c663cd961eabbc433'
INNER='c71bafdf9f5a791c540c6a5a1a12ba0f2b84b9b221c2b018cbdebf9684728571'
SOURCES={'build_packet.py':'e46cd29633c9dcd83db174fb77ff5c7acf751553bfb46963129cdb1129a255cb',
 'launch_once.py':'4508ff44bd2a7c0d8de678bfc18d54b73cf6c2829dfe62835285e1a344e2991c',
 'monitor_once.py':'4d22fb95146f6e308fab021f35023a491909d03af49b9b5ea4d4bc5c618cf373'}
COLLECTOR='b1994191042e910923ffe23e504f2ecd9af6d5242ad6743a5537a2aa4e73e5c3'
report={'checks':[],'facts':{},'errors':[],'source_execution':False,'tests_rerun':0,'remote_actions':0,
        'execution_qualified':False,'scientific_qualified':False}
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(o):return (json.dumps(o,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def check(ok,label):
    if not ok:raise ValueError(label)
    report['checks'].append(label)
def info(p):
    before=p.lstat();check(stat.S_ISREG(before.st_mode) and before.st_nlink==1,'ordinary sole-link '+str(p))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    after=p.lstat();identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
    check(identity(before)==identity(after),'stable read '+str(p))
    return {'bytes':after.st_size,'sha256':h.hexdigest()}
def j(p):return json.loads(p.read_bytes())
def main():
    check(info(ORIGINAL/'packet.json')['sha256']==info(ORIGINAL_EXT/'packet.json')['sha256']==OLD_SHA,'original two outer pins retained')
    check(info(NEW/'packet.json')['sha256']==info(NEW_EXT/'packet.json')['sha256']==NEW_SHA,'new two outer pins')
    old=j(ORIGINAL/'packet.json');new=j(NEW/'packet.json')
    check(old.keys()==new.keys(),'identical outer fields')
    changed=[n for n in old if old[n]!=new[n]];check(changed==['new_input_files'],'only outer member hash map changed')
    a=old['new_input_files'];b=new['new_input_files'];check(a.keys()==b.keys(),'unchanged outer membership')
    delta={n:b[n] for n in b if a[n]!=b[n]};check(delta==SOURCES,'exact three source hash replacements')
    check(new['finalizer_packet_sha256']==b['inputs/packet.json']==INNER,'same independent inner pin')
    check((NEW/'packet.json').read_bytes()==canonical(new),'canonical new outer bytes')
    expected={**b,'packet.json':NEW_SHA};maps=[]
    for root in (NEW,NEW_EXT):
        check(root==root.resolve() and root.is_dir(),'canonical new root '+str(root))
        mapping={}
        for p in sorted(root.rglob('*')):
            check(not p.is_symlink(),'no link '+str(p))
            if p.is_file():mapping[p.relative_to(root).as_posix()]=info(p)
        check(set(mapping)==set(expected),'exact new membership '+str(root))
        check(all(mapping[n]['sha256']==h for n,h in expected.items()),'every new member pin '+str(root))
        maps.append(mapping)
    check(maps[0]==maps[1],'two complete byte-identical copies')
    check(all(not os.path.samestat((NEW/n).stat(),(NEW_EXT/n).stat()) for n in expected),'independent copy identities')
    check(len(expected)==223 and sum(v['bytes'] for v in maps[0].values())==782762459,'exact223 files782762459bytes')
    check(sum(n not in SOURCES and n!='packet.json' and expected[n]==a[n] for n in expected)==219,'all219 original member pins unchanged')
    # Compare to prior authenticated pins without repeating old full-copy inspection.
    prior=PREP/'finalizer-concrete-packet-review01/review.md'
    check(sha(prior.read_bytes())=='4698740072ae466d813cbee8ae8454b1724db832a487cbb35c15b4b2d8bf3213','historical review unchanged')
    delta_review=PREP/'finalizer-outer-reader128-review01/review.md'
    check(sha(delta_review.read_bytes())=='df74fcfaf09a0d23b47d956f162b0938d1156efb9bf4a795c32fd1f7560b5fd8','independent successor source review')
    for n,h in SOURCES.items():
        check(info(PREP/'finalizer-outer-reader128-review01/final-source'/n)['sha256']==h,'exact independently captured successor '+n)
    collector=PREP/'finalizer-outer-reader128-preparation01/implementation/pull_terminal.py'
    check(info(collector)['sha256']==COLLECTOR,'standalone collector successor')
    pub=j(PUB/'publication.json');run=j(PUB/'run.json');parent_run=j(PUB/'parent-run.json')
    check(pub['outer_sha256']==NEW_SHA and pub['inner_sha256']==INNER and pub['original_outer_sha256']==OLD_SHA,'publication original/new/inner joins')
    check(pub['source_pins']==SOURCES and pub['standalone_collector_sha256']==COLLECTOR,'publication source pins')
    check(pub['changed_members']==sorted([*SOURCES,'packet.json']) and pub['unchanged_members']==219,'publication exact delta')
    check(pub['new_roots']==[str(NEW),str(NEW_EXT)] and pub['old_roots']==[str(ORIGINAL),str(ORIGINAL_EXT)],'publication path lineage')
    check(run['returncode']==0 and run['builder_invoked'] is run['submission_invoked'] is False and run['remote_actions']==0,'saved one metadata-only publication')
    evidence={}
    for n,v in pub['anchors'].items():
        check(info(PUB/n)==dict(bytes=v['bytes'],sha256=v['sha256']),'captured publication anchor '+n)
        check(info(Path(v['path']))==dict(bytes=v['bytes'],sha256=v['sha256']),'independent anchor original '+n)
    ext=Path((PUB/'external-path.txt').read_text().strip());inv=j(PUB/'inventory.json')
    for n,v in inv['files'].items():
        check(info(PUB/n)==info(ext/n)==v,'publication evidence mirror '+n);evidence[n]=v
    check((PUB/'inventory.json').read_bytes()==(ext/'inventory.json').read_bytes(),'publication inventory mirror')
    groups={}
    for label in ('root','inputs'):
        selected={n:v for n,v in maps[0].items() if n.startswith('inputs/')==(label=='inputs')}
        size=sum(v['bytes'] for v in selected.values());timeout=120+(size+1024**2-1)//1024**2
        check(timeout<=900,'transfer remains within named900 cap '+label)
        groups[label]={'files':len(selected),'bytes':size,'timeout_seconds':timeout}
    check(groups=={'root':{'files':105,'bytes':738753046,'timeout_seconds':825},'inputs':{'files':118,'bytes':44009413,'timeout_seconds':162}},'exact successor group sizing')
    report['facts']={'outer_sha256':NEW_SHA,'inner_sha256':INNER,'original_outer_sha256':OLD_SHA,'changed_source_pins':SOURCES,
       'collector_sha256':COLLECTOR,'packet_files':223,'packet_bytes':782762459,'unchanged_members':219,'groups':groups,
       'publication_pin':info(PUB/'publication.json'),'publication_run':run,'publication_parent_run':parent_run,
       'publication_evidence':evidence,'new_packet_inventory':maps[0],'old_complete_review_reused':True,
       'original_reader_size_miss_corrected_by_distinct_source_review':True}

start=time.monotonic();cpu=time.process_time()
try:main()
except BaseException as exc:
    report['errors'].append({'type':type(exc).__name__,'message':str(exc)})
    with (HERE/'failure.traceback.txt').open('x') as f:f.write(traceback.format_exc())
finally:
    report.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,
       peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=sha(Path(__file__).read_bytes()),
       utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (HERE/'checks.json').open('x') as f:json.dump(report,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({'checks':len(report['checks']),'errors':report['errors'],'wall_seconds':report['wall_seconds']}))
raise SystemExit(bool(report['errors']))
