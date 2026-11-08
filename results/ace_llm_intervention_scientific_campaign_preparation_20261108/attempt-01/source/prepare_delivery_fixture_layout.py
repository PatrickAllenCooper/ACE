"""Stage and preflight a source-only B fixture layout; never execute fixtures.

Original project/learner files and the prior location derivation are preserved.
Explicit import-resolution and cached-module checks describe this instant only.
No sys.path/meta_path mutation, model import, runtime or training qualification.
"""
import argparse
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import re
import stat
import sys

from prepare_delivery_prospective_fixture_locations import (
    IO, MODEL, ORIGINAL, PINS, REGISTRATION, REPLACEMENTS, REVISION, derive_model,
    unique)
from prepare_delivery_publication_snapshot import (
    exclusive_directory, plain_path, seal_directory, store)

RUNTIME = {'torch':'2.9.1','numpy':'2.2.6','scipy':'1.15.3',
           'pandas':'2.3.3','sympy':'1.14.0','PyYAML':'6.0.3'}
SEARCH = ['fixtures','project/scripts/research','project','source']
FALSE_FLAGS = ['fixtures_executed','runtime_qualified',
               'original_training_reproduction_qualified',
               'anonymous_execution_qualified','public_release_approved']


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:raise ValueError(message)


def decode(raw):
    return json.loads(raw,object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite metadata')))


def relative(value):
    require(type(value) is str and bool(value),'relative source path required')
    p=Path(value)
    require(not p.is_absolute() and '..' not in p.parts and p.as_posix()==value,
            'canonical relative source path required')
    return p


def snapshot(path,pin):
    require(type(pin) is str and re.fullmatch('[0-9a-f]{64}',pin),
            'independent SHA256 pin required')
    fd=os.open(plain_path(path),os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode),'regular source required')
        raw=stream.read(32*2**20+1)
    require(len(raw)<=32*2**20 and digest(raw)==pin,'source size or digest differs')
    return raw


def module_plan(paths):
    """The two derived fixtures intentionally precede their private originals."""
    modules={}
    for name in sorted(paths):
        p=relative(name)
        if p.suffix!='.py':continue
        if name.startswith('originals/') and len(p.parts)==2:continue
        if name.startswith('project/scripts/research/') and len(p.parts)==4:
            module=p.stem
        elif name=='project/baselines.py':module='baselines'
        elif name=='project/experiments/large_scale_scm.py':module='experiments.large_scale_scm'
        elif name.startswith('source/ace/') and len(p.parts)==3:
            module='ace' if p.stem=='__init__' else 'ace.'+p.stem
        elif name.startswith('fixtures/') and len(p.parts)==2:
            # Applied after original project modules below.
            continue
        else:raise ValueError('unsupported original Python layout: '+name)
        require(module not in modules,'ambiguous original import name')
        modules[module]=name
    for basename in (Path(MODEL).name,Path(IO).name):
        require('fixtures/'+basename in paths,'both derived fixtures required')
        modules[Path(basename).stem]='fixtures/'+basename
    require('ace' in modules and 'experiments.large_scale_scm' in modules,
            'learner package and generator required')
    return modules


def capture(closure,closure_pin,locations,location_pin):
    closure,locations=plain_path(closure),plain_path(locations)
    closure_raw=snapshot(closure,closure_pin);record=decode(closure_raw)
    location_raw=snapshot(locations,location_pin);contract=decode(location_raw)
    require(record['source_revision']==REVISION and record['registration_sha256']==REGISTRATION
            and record['outcomes_accessed'] is False,'original closure identity differs')
    require(contract['schema']=='delivery-B-fixture-locations-v1'
            and contract['original_revision']==REVISION
            and contract['registration_sha256']==REGISTRATION
            and contract['original_closure_sha256']==closure_pin,
            'fixture predecessor lineage differs')
    for key in ('fixtures_executed','original_training_reproduction_qualified',
                'anonymous_execution_qualified','public_release_approved'):
        require(contract[key] is False,'unexpected predecessor qualification')
    require(contract['location_edits']=={n:{'original_expression':ORIGINAL[n],
            'derived_expression':REPLACEMENTS[n]} for n in ORIGINAL},'location contract differs')
    project,learner=record['project_files'],record['learner_files']
    require(len(project)==24 and len(learner)==19
            and all(project.get(n)==pin for n,pin in PINS.items()),'original43 closure differs')
    captured={}
    for folder,mapping in (('project',project),('source',learner)):
        for name,pin in mapping.items():
            captured[folder+'/'+name]=snapshot(closure.parent/folder/relative(name),pin)
    expected={'originals/'+Path(MODEL).name:captured['project/'+MODEL],
              'originals/'+Path(IO).name:captured['project/'+IO],
              'fixtures/'+Path(MODEL).name:derive_model(captured['project/'+MODEL]),
              'fixtures/'+Path(IO).name:captured['project/'+IO]}
    entries={e['path']:e for e in contract['files']}
    require(len(entries)==len(contract['files'])==4 and set(entries)==set(expected),
            'exact fixture predecessor membership required')
    for name,want in expected.items():
        e=entries[name];raw=snapshot(locations.parent/relative(name),e['sha256'])
        require(raw==want and e['bytes']==len(raw),'fixture derivation bytes differ')
        captured[name]=raw
    captured['source-closure.json']=closure_raw
    captured['fixture-location-contract.json']=location_raw
    return captured


def prepare(closure,closure_pin,locations,location_pin,destination):
    destination=plain_path(destination)
    require(not destination.exists(),'new exclusive layout required')
    for root in (plain_path(closure).parent,plain_path(locations).parent,
                 Path(__file__).resolve().parents[2]):
        require(not destination.is_relative_to(root),'output overlaps source/repository')
    captured=capture(closure,closure_pin,locations,location_pin)
    modules=module_plan(captured)
    contract={'schema':'delivery-B-fixture-layout-v1','scope':'private source/import preflight only',
              'original_revision':REVISION,'registration_sha256':REGISTRATION,
              'original_closure_sha256':closure_pin,'location_contract_sha256':location_pin,
              'files':{n:{'sha256':digest(raw),'bytes':len(raw)} for n,raw in sorted(captured.items())},
              'search_order':SEARCH,'modules':modules,
              'namespace_packages':{'experiments':'project/experiments'},
              'dynamic_file_imports':{'delivery_audited_large_scm':'project/experiments/large_scale_scm.py'},
              'required_environment':{'ACE_DELIVERY_FIXTURE_ROOT':'explicit absolute layout root'},
              'required_python':'3.11','required_dependencies':RUNTIME,
              'source_changes':'none beyond the predecessor two location expressions',
              **{key:False for key in FALSE_FLAGS},'new_fits':0,'new_responses':0,
              'remaining_gates':['enforced authenticated import execution and exact runtime',
                  'explicit fixture execution/resource authorization and supervised receipts',
                  'batch/collection/training location and guard contracts',
                  'new source/notice/manifest disposition and deliberate anonymity review']}
    raw=(json.dumps(contract,sort_keys=True,indent=2)+'\n').encode()
    fd=exclusive_directory(destination)
    try:
        for name,data in captured.items():store(fd,name,data)
        store(fd,'layout-contract.json',raw);seal_directory(fd)
        a,b=os.stat(plain_path(destination),follow_symlinks=False),os.fstat(fd)
        require((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino),'output custody moved')
    finally:os.close(fd)
    return {'layout_contract_sha256':digest(raw),'files':len(captured)+1,
            'original_source_objects':43,'derived_fixtures':2,'fixtures_executed':False}


def preflight(root,expected_contract,modules=None):
    """Inspect an explicit layout, with no imports through the inspected paths.

    Explicit FileFinders avoid sys.path hooks and importer caches. No loader's
    exec_module is called. This does not install a guard for later execution.
    """
    require(type(root) in (str,Path) or isinstance(root,Path),'explicit root required')
    require(Path(root).is_absolute(),'absolute fixture root required')
    root=plain_path(root)
    contract=decode(snapshot(root/'layout-contract.json',expected_contract))
    require(contract['schema']=='delivery-B-fixture-layout-v1'
            and contract['original_revision']==REVISION and contract['registration_sha256']==REGISTRATION,
            'layout identity differs')
    require(contract['search_order']==SEARCH and contract['required_dependencies']==RUNTIME
            and contract['required_python']=='3.11'
            and contract['required_environment']=={'ACE_DELIVERY_FIXTURE_ROOT':'explicit absolute layout root'}
            and type(contract['new_fits']) is int and contract['new_fits']==0
            and type(contract['new_responses']) is int and contract['new_responses']==0,
            'layout search/runtime contract differs')
    require(all(contract[k] is False for k in FALSE_FLAGS),'unexpected execution qualification')
    files=contract['files'];observed=set()
    for path in root.rglob('*'):
        require(not path.is_symlink(),'layout symlink forbidden')
        if path.is_file():observed.add(path.relative_to(root).as_posix())
        else:require(path.is_dir(),'nonregular layout object')
    require(observed==set(files)|{'layout-contract.json'},'exact source-only layout membership differs')
    captured=capture(root/'source-closure.json',contract['original_closure_sha256'],
                     root/'fixture-location-contract.json',contract['location_contract_sha256'])
    require(set(captured)==set(files),'layout and predecessor membership differ')
    require(all(len(raw)==files[n]['bytes'] and digest(raw)==files[n]['sha256']
                for n,raw in captured.items()),'layout bytes or predecessor binding differs')
    plan=module_plan(files)
    require(plan==contract['modules'],'layout module map differs')
    require(contract['namespace_packages']=={'experiments':'project/experiments'}
            and contract['dynamic_file_imports']=={'delivery_audited_large_scm':'project/experiments/large_scale_scm.py'},
            'namespace/dynamic import disposition differs')
    cached=sys.modules if modules is None else modules
    roots={n.split('.')[0] for n in plan}|set(contract['dynamic_file_imports'])
    conflicts=sorted(n for n in cached if any(n==r or n.startswith(r+'.') for r in roots))
    require(not conflicts,'cached source module requires a fresh process: '+', '.join(conflicts))
    resolved={}
    def lookup(name,paths):
        for path in paths:
            finder=importlib.machinery.FileFinder(str(path),
                (importlib.machinery.SourceFileLoader,importlib.machinery.SOURCE_SUFFIXES))
            spec=finder.find_spec(name)
            if spec is not None:return spec
        raise ValueError('source import not found: '+name)
    ns=lookup('experiments',[root/n for n in SEARCH])
    require(ns.origin is None and list(ns.submodule_search_locations)==[str(root/'project/experiments')],
            'original namespace package differs')
    resolved['experiments']={'origin':None,'locations':list(ns.submodule_search_locations)}
    for name in sorted(plan,key=lambda n:(n.count('.'),n)):
        if '.' in name:
            parent=name.rpartition('.')[0]
            require(parent in resolved,'parent package not resolved')
            spec=lookup(name,resolved[parent]['locations'])
        else:spec=lookup(name,[root/n for n in SEARCH])
        require(spec.origin is not None and plain_path(spec.origin)==root/plan[name],
                'import resolves to unexpected source: '+name)
        locations=list(spec.submodule_search_locations or [])
        resolved[name]={'origin':spec.origin,'locations':locations,'source_sha256':files[plan[name]]['sha256']}
        if name=='ace':require(locations==[str(root/'source/ace')],'learner package path differs')
    return {'layout_contract_sha256':expected_contract,'files_verified':len(files),
            'source_modules_resolved':len(plan),'namespace_packages':1,
            'cached_source_modules':0,'resolved':resolved,
            'import_path_mutated':False,'loaders_executed':False,
            **{key:False for key in FALSE_FLAGS},'new_fits':0,'new_responses':0,
            'scope':'current source bytes and explicit source-only lookup; no later execution guarantee'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare')
    for n in ('closure','locations','destination'):a.add_argument('--'+n,type=Path,required=True)
    for n in ('closure-pin','location-pin'):a.add_argument('--'+n,required=True)
    a=sub.add_parser('preflight');a.add_argument('--root',type=Path,required=True);a.add_argument('--contract-pin',required=True)
    a=p.parse_args()
    result=prepare(a.closure,a.closure_pin,a.locations,a.location_pin,a.destination) if a.command=='prepare' else preflight(a.root,a.contract_pin)
    print(json.dumps(result,indent=2))
