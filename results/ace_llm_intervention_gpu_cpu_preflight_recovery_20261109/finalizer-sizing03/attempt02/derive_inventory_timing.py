"""Derive counts/timing brackets from two completed saved transports, never origins.

No producer imports, payload decode, remote calls or live-tree reads. An output is
new sizing evidence only, not custody/GPU acceptance or operational adoption.
"""
from pathlib import Path, PurePosixPath
import argparse
import ast
from bisect import bisect_right
from datetime import datetime
import hashlib
import json
import os
import re
import stat


def need(ok, message):
    if not ok:
        raise ValueError(message)


def capture(path, expected):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as f:
        before = os.fstat(f.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_size <= 128*1024**2, 'bounded regular metadata required')
        value = f.read(128*1024**2+1)
        after = os.fstat(f.fileno())
    identity = lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(identity(before) == identity(after) and len(value) == before.st_size, 'metadata changed during capture')
    need(hashlib.sha256(value).hexdigest() == expected, 'supplied metadata pin differs')
    return value


def decode(value):
    def unique(items):
        result = {}
        for key, item in items:
            need(key not in result, 'duplicate metadata key')
            result[key] = item
        return result
    return json.loads(value, object_pairs_hook=unique)


def observation(value):
    transport = decode(value)
    need(transport['returncode'] == 0 and not transport.get('timeout'), 'completed successful transport required')
    observed = decode(transport['stdout'])
    need(observed.get('terminal_gate') is True and not observed.get('observation_error')
         and observed.get('upstream_errors') == {}, 'complete recorded observation required')
    files = observed['files']
    need(type(files) is dict and files, 'complete full file map required')
    for name, item in files.items():
        need(type(name) is str and not name.startswith('/') and '..' not in PurePosixPath(name).parts
             and type(item) is dict and set(item) == {'bytes','sha256'}
             and type(item['bytes']) is int and item['bytes'] >= 0
             and type(item['sha256']) is str and re.fullmatch('[0-9a-f]{64}',item['sha256']),
             'ordinary inventory record required')
    source = ast.parse(transport['python_source'])
    loops = [n for n in ast.walk(source) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)
             and [getattr(v,'id',None) for v in n.target.elts] == ['label','tree','present']
             and isinstance(n.iter,ast.Tuple)]
    need(len(loops) == 1, 'unique captured original traversal required')
    labels = [ast.literal_eval(n.elts[0]) for n in loops[0].iter.elts]
    need(labels == ['control','launch','cpu','training','token','failed-token01','common','failed01'],
         'GPU02 root order differs; do not infer new traversal semantics')
    # sorted(Path.rglob()) orders path COMPONENTS, not slash-joined strings.
    ordered = [n for label in labels for n in sorted((n for n in files if n.startswith(label+'/')),key=PurePosixPath)]
    indexes = {n:i+1 for i,n in enumerate(ordered)}
    samples = [(0,0.0)]
    for line in transport['stderr'].splitlines():
        if not line.startswith('{'):
            continue
        item = decode(line)
        if item.get('phase') != 'terminal_inventory_verification':
            continue
        count, elapsed = item['files_read'],item['elapsed_seconds']
        need(type(count) is int and count == samples[-1][0]+1000
             and type(elapsed) in (int,float) and samples[-1][1] <= elapsed < 1230,
             'ordered 1000-read saved progress required')
        samples.append((count,elapsed))
    need(len(samples)>1, 'saved progress unavailable')
    cpu = 'cpu/cpu/'; campaign = cpu+'worker/artifacts/campaign/'
    selectors = {
        'cpu_output_excluding_campaign':lambda n:n.startswith(cpu) and not n.startswith(campaign),
        'gpu_output':lambda n:n.startswith('launch/gpu/'),
        'campaign':lambda n:n.startswith(campaign),
        'cpu_inputs':lambda n:n.startswith('cpu/inputs/'),
        'gpu_inputs':lambda n:n.startswith('control/inputs/'),
        'attempt_hashes':lambda n:n.startswith(campaign) and re.match(r'policy-0[0-7]/',n[len(campaign):])
            and not re.fullmatch(r'policy-0[0-7]/attempt(?:\.candidate)?\.json',n[len(campaign):]),
        'arm_files':lambda n:n.startswith(campaign) and re.match(r'policy-0[0-7]/world-[0-9]{2}/',n[len(campaign):]),
        'sealed_input_rereads':lambda n:n.startswith(campaign) and '/sealed-inputs/' in n,
        'arm_receipts':lambda n:n.startswith(campaign) and re.fullmatch(r'policy-0[0-7]/world-[0-9]{2}/arm-receipt\.json',n[len(campaign):]),
    }
    groups = {}
    for group, select in selectors.items():
        members = [n for n in ordered if select(n)]
        selected = sorted(indexes[n] for n in members)
        need(selected and max(selected) <= samples[-1][0], 'group not fully bracketed by saved samples: '+group)
        inside, intersect = [], []
        for (left,t0),(right,t1) in zip(samples,samples[1:]):
            count = bisect_right(selected,right)-bisect_right(selected,left)
            if count:
                intersect.append((left,right,t1-t0))
            if count == right-left:
                inside.append((left,right,t1-t0))
        groups[group] = dict(files=len(members),bytes=sum(files[n]['bytes'] for n in members),
            first_counter=min(selected),last_counter=max(selected),
            wholly_contained_sample_seconds=sum(x[2] for x in inside),
            intersecting_sample_seconds=sum(x[2] for x in intersect),
            intersecting_counter_intervals=[[x[0],x[1]] for x in intersect],
            warning='Upper bracket includes adjacent files/directories; sums across groups can overlap. Not a future latency bound.')
    root_counts = {label:dict(files=sum(n.startswith(label+'/') for n in files),
        bytes=sum(v['bytes'] for n,v in files.items() if n.startswith(label+'/'))) for label in labels}
    result = dict(observation_utc=observed['utc'],transport_utc=transport['utc'],
        elapsed_utc_bracket_seconds=(datetime.fromisoformat(transport['utc'])-datetime.fromisoformat(observed['utc'])).total_seconds(),
        utc_bracket_caveat='Includes scheduling queries, serialization, shell/network/transport; UTC clock agreement is not established here.',
        process_cpu_seconds=None,process_cpu_caveat='Original inventory observer did not record process CPU.',
        full_files=len(files),full_bytes=sum(v['bytes'] for v in files.values()),
        roots=root_counts,groups=groups,progress_samples=samples[1:],
        captured_source_sha256=hashlib.sha256(transport['python_source'].encode()).hexdigest(),
        last_counter_is_not_unique_files=True,source_reads_after_roots='journal and independently pinned upstream rereads',
        accepted_gpu=False,custody_qualified=False)
    return result,files,transport['stderr']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ('before','after'):
        parser.add_argument('--'+role,required=True)
        parser.add_argument('--'+role+'-sha256',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    captured = {role:capture(getattr(args,role),getattr(args,role+'_sha256')) for role in ('before','after')}
    outputs = {role:observation(value) for role,value in captured.items()}
    need(outputs['before'][1] == outputs['after'][1], 'completed before/after full maps differ')
    out = Path(args.output);out.mkdir(mode=0o700,exist_ok=False)
    result = dict(schema='ace-finalizer-inventory-sizing-derivation-v1',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_pins={role:dict(path=str(Path(getattr(args,role)).absolute()),sha256=getattr(args,role+'_sha256'),bytes=len(value)) for role,value in captured.items()},
        observations={role:value[0] for role,value in outputs.items()},full_path_timing_supported=False,
        resource_admission_accepted=False,scientific_qualified=False,remote_actions=0,allocations=0)
    for role, (_,_,stderr) in outputs.items():
        (out/(role+'.stderr')).write_text(stderr)
    (out/'derived.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
