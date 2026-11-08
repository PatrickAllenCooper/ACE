from pathlib import Path
import datetime,hashlib,io,json,os,shutil,tarfile
repo=Path('/Users/pat/code/ACE')
roots={'initial':Path('/tmp/ace-scientific-campaign-outer-y__40vxa'),
       'correction':Path('/tmp/ace-scientific-campaign-gpu-corrections-td2w121t'),
       'review':Path('/Users/pat/.codex/tmp/ace-scientific-campaign-gpu-outer-review-7uexqwrp')}
external=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-campaign-gpu-preparation-20261108-01')
local=repo/'results/ace_llm_intervention_scientific_campaign_gpu_preparation_20261108/preparation01'
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=lambda v:(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
members=[];links={};runs={}
for label,root in roots.items():
    for path in sorted(root.rglob('*')):
        name=label+'/'+path.relative_to(root).as_posix()
        if path.is_symlink():
            links[name]={'target':os.readlink(path),'scope':'negative fixture metadata only, target not read'};continue
        if path.is_dir():continue
        assert path.is_file(),path
        value=path.read_bytes();members.append((name,value))
        if path.name=='run.json':runs.setdefault(sha(value),{'paths':[],'record':json.loads(value)})['paths'].append(str(path))
external.mkdir(mode=0o700,exist_ok=False)
def save(name,value):
    p=external/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(value)
with tarfile.open(external/'attempts.tar.gz','w:gz',dereference=True) as archive:
    for name,value in members:
        info=tarfile.TarInfo(name);info.size=len(value);info.mode=0o600;archive.addfile(info,io.BytesIO(value))
with tarfile.open(external/'attempts.tar.gz','r:gz') as archive:
    assert len(archive.getmembers())==len(members)
    for info,(name,value) in zip(archive.getmembers(),members):
        assert info.isfile() and info.name==name and archive.extractfile(info).read()==value
save('archive-members.json',raw({n:dict(sha256=sha(v),bytes=len(v)) for n,v in members}))
save('symlink-metadata.json',raw(links))
for n,pin in {'qualify_llm_intervention_scientific_campaign_gpu.py':'1a4ce8c4a6e91b819f3477ef857da37c2e4590cc0d9f6ef621cdd951038c0556',
    'test_qualify_llm_intervention_scientific_campaign_gpu.py':'1b70742769218d7e1d84acd04d83c77aacd2a1ba8e8177a9d20ac8bd9ca4de6c'}.items():
    value=(repo/'scripts/research'/n).read_bytes();assert sha(value)==pin;save('source/'+n,value)
save('review.md',(repo/'docs/development/guidance/reviews/ace_llm_intervention_scientific_campaign_gpu_outer_review_2026-11-08.md').read_bytes())
save('packaging.py',Path(__file__).read_bytes())
files={p.relative_to(external).as_posix():sha(p.read_bytes()) for p in sorted(external.rglob('*')) if p.is_file()}
receipt=dict(schema='ace-scientific-campaign-gpu-preparation-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    files=files,original_roots={k:str(v) for k,v in roots.items()},archive_regular_files=len(members),
    archive_regular_bytes=sum(len(v) for _,v in members),negative_symlink_metadata=links,
    archive_aliases=0,complete_retained_regular_bytes_verified=True,original_runs_by_receipt_sha256=runs,
    new_fixture_methods=9,required_findings_initial=2,required_findings_remaining=0,
    scientific_execution=False,scientific_worlds_generated=0,new_environment_responses=0,
    model_runtime_execution=False,new_allocations=0,tests_repeated_during_packaging=0,
    closed_allocated_cpu_seconds=11040,closed_gpu_slice_seconds=1601,cumulative_cpu_preparation_allowance=3600,
    scope='Source, fabricated boundaries and independent review only. Actual full freeze, predecessor acceptances, scheduler/runtime execution and custody remain required.',
    cost_scope='Original command records count each distinct receipt once. Copying or reviewing them is not another test run. Packaging/development/review overhead unmetered.')
save('preparation.json',raw(receipt));local.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(external,local)
for path in external.rglob('*'):
    if path.is_file():assert path.read_bytes()==(local/path.relative_to(external)).read_bytes()
print(json.dumps(dict(receipt_sha256=sha(raw(receipt)),files=len(files)+1,regular=len(members),negative_links=len(links))))
