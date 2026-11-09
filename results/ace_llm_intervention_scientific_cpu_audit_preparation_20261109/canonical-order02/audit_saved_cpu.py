"""Prospective saved-evidence aid. Never runs a producer or issues acceptance.

All input pins are caller supplied. Only this aid's exclusive output is written.
Remote archive verification is a separately supplied, pinned observation; this
program neither contacts CURC nor claims that omitted bytes exist locally.
See contract.md for mandatory independent review gates outside this aid.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import resource
import stat
import sys
import time
import traceback

ENTRY = 'prepare_llm_intervention_scientific_campaign_cpu.py'
ENTRY_SHA = 'ba439d819f49f6832c97181e4ff727c4813f29031e7b67f8085670f845b0f893'
FREEZE = '15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d'
SOURCE = 'a9346022f145cc29439325b6f291597b2e6ec037'
POLICIES = ('uniform_random', 'menu_coverage', 'local_information', 'target_risk',
            'learned_ranker', 'frozen_llm', 'sft_llm', 'dpo_llm')
CPU = 'launch/cpu/'
ART = CPU + 'worker/artifacts/'
PHASE = ART + 'cpu-phase/'
CAMPAIGN = ART + 'campaign/'
RESOURCES = dict(account='ucb736_asc2', partition='acpu', qos='cpu-normal',
                 cpus=2, memory_mib=7680, gpus=0, requeue=False, wall_seconds=6000)
DEADLINES = dict(startup=180, work=5940, hard=5980)
NUMERIC_SOURCES = {
    'screen': {'llm_intervention_'+n+'.py' for n in
               'contract episode numerical_policy replicated_branches replicated_feedback scm screen_policy screen_protocol supervision target_risk task training_dataset world'.split()},
    'ranker': {'llm_intervention_'+n+'.py' for n in
               'contract episode learning numerical_policy policy_cli ranker scm target_risk task'.split()},
}
ACCEPTANCE_KEYS = sorted(('schema', 'status', 'cpu_packet_sha256', 'job_id',
    'allocation', 'cpu_output', 'campaign_root', 'cpu_output_files',
    'cpu_output_excluded_directory', 'campaign_files', 'bundle_path',
    'bundle_sha256', 'phase_terminal_path', 'phase_terminal_sha256',
    'attempt_pins', 'scientific_qualified'))
MANUAL_GATES = (
    'Independent source/runtime review: complete check_prefix, check_stage, '
    'metadata/cache/provenance/module/native-map gates from frozen validators; '
    'this aid checks bindings/versions/closure but does not reimplement every gate.',
    'Opaque campaign bundle/source-plan semantics: exact 24 supplied identities, '
    'proposal designs/priors/namespaces, policy argv and five CPU/eight total '
    'bindings; compare frozen source and saved producer validation, without '
    'generating worlds, loading tensors or evaluating private outcomes.',
    'Original training ancestor: reuse independently accepted full custody, '
    'all failed attempts and checkpoint roles; resolve the pinned existing '
    'ancestor reference, not a fictional recursive compact-capsule copy.',
    'Independent scheduler/source/collector review including exact commit, '
    'both namespace registries and all original input/source pins. Verify '
    'the pinned collector/helper archive readback receipt is authentic for this '
    'sealed snapshot. A local program cannot independently attest SSH.',
    'Review complete machine report and every retained checker failure; only '
    'then author the existing exact 16-key CPU acceptance and separate sidecars. '
    'This aid intentionally has no acceptance-writing option.',
)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pin(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def raw(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def decode(data):
    def pairs(items):
        result = {}
        for k, v in items:
            need(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    def bad(value):
        raise ValueError('nonfinite JSON: '+value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad)


def name(value):
    p = PurePosixPath(value)
    need(type(value) is str and value and not p.is_absolute()
         and '..' not in p.parts and str(p) == value, 'unsafe inventory name')
    return value


def identity(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)


def file_bytes(path, expected=None, maximum=32*1024**2, materialize=True):
    path = Path(path)
    need(path.is_absolute(), 'absolute input path required')
    for parent in (path, *path.parents):
        need(not parent.is_symlink(), 'symlink evidence path forbidden: '+str(path))
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as f:
        before = os.fstat(f.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
             'independent regular evidence file required')
        need(before.st_size <= maximum, 'metadata bound exceeded: '+str(path))
        h = hashlib.sha256(); chunks = []; size = 0
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b); size += len(b)
            need(size <= maximum, 'input grew beyond bound')
            if materialize:
                chunks.append(b)
        need(identity(before) == identity(os.fstat(f.fileno())) == identity(path.lstat()),
             'evidence changed while read')
    info = dict(bytes=size, sha256=h.hexdigest())
    if expected is not None:
        need(pin(expected) and info['sha256'] == expected, 'independent byte pin differs: '+str(path))
    return b''.join(chunks) if materialize else info


def inventory(root):
    root = Path(root)
    need(root.is_dir() and not root.is_symlink(), 'ordinary evidence directory required')
    files = {}; directories = []
    for base, ds, fs in os.walk(root, followlinks=False):
        for n in sorted(ds):
            p = Path(base)/n
            need(not p.is_symlink(), 'directory alias forbidden')
            directories.append(p.relative_to(root).as_posix())
        for n in sorted(fs):
            p = Path(base)/n
            files[p.relative_to(root).as_posix()] = file_bytes(
                p, maximum=128*1024**2, materialize=False)
    return dict(files=files, directories=sorted(directories))


def validate_inventory(v):
    need(set(v) == {'files', 'directories'} and type(v['files']) is dict,
         'exact inventory schema required')
    need(v['directories'] == sorted(set(v['directories'])), 'unique sorted directories required')
    for n in v['directories']:
        name(n)
    for n, row in v['files'].items():
        name(n)
        need(set(row) == {'bytes', 'sha256'} and type(row['bytes']) is int
             and row['bytes'] >= 0 and pin(row['sha256']), 'invalid file inventory')
    need(not (set(v['directories']) & set(v['files'])), 'file/directory collision')


def normalized_observation_inventory(observation):
    # Preserve original observation bytes and reject duplicate directory names.
    need(type(observation['directories']) is list, 'original directory list required')
    value = dict(files=observation['files'], directories=sorted(observation['directories']))
    validate_inventory(value)
    return value


def object_parts(data):
    """Top-level lexical spans; unselected response/descriptor values stay opaque."""
    text = data.decode('utf-8'); decoder = json.JSONDecoder(); i = 0; result = {}
    def ws(p):
        while p < len(text) and text[p].isspace():
            p += 1
        return p
    i = ws(i); need(i < len(text) and text[i] == '{', 'object required'); i += 1
    while True:
        i = ws(i)
        if i < len(text) and text[i] == '}':
            i += 1; break
        k, end = decoder.raw_decode(text, i)
        need(type(k) is str and k not in result, 'duplicate/nonstring public key')
        i = ws(end); need(text[i:i+1] == ':', 'object colon required'); start = ws(i+1)
        j = start; stack = []; quoted = escaped = False
        while j < len(text):
            c = text[j]
            if quoted:
                if escaped: escaped = False
                elif c == '\\': escaped = True
                elif c == '"': quoted = False
            elif c == '"': quoted = True
            elif c in '[{': stack.append(c)
            elif c in ']}':
                if not stack: break
                need((stack.pop(), c) in (('[', ']'), ('{', '}')), 'unbalanced JSON')
            elif c == ',' and not stack: break
            j += 1
        need(j > start and not quoted and not stack, 'incomplete JSON value')
        result[k] = text[start:j].strip().encode(); i = ws(j)
        need(i < len(text) and text[i] in ',}', 'object delimiter required')
        if text[i] == '}': i += 1; break
        i = ws(i+1); need(i < len(text) and text[i] != '}', 'trailing comma')
    need(ws(i) == len(text), 'trailing JSON')
    return result


def public(data, keys):
    return {k: decode(v) for k, v in object_parts(data).items() if k in keys}


def normalize(rows):
    result = []
    for r in rows:
        need(len(r) >= 8 and not any(r[8:]) and r[0].isdigit()
             and r[4].isdigit() and r[5].isdigit(), 'malformed accounting row')
        ts = [x.split('=', 1) for x in r[7].split(',') if x]
        need(all(len(x) == 2 for x in ts) and len(dict(ts)) == len(ts), 'ambiguous TRES')
        result.append([r[0], r[1].strip(), r[2].strip(), r[3].strip(), int(r[4]),
                       int(r[5]), r[6].strip(), dict(ts)])
    need(len({r[0] for r in result}) == len(result), 'duplicate job')
    return sorted(result, key=lambda r: int(r[0]))


def costs(rows):
    cpu = gpu = 0
    for r in normalize(rows):
        g = {k: v for k, v in r[7].items() if k.startswith('gres/')}
        need(r[7] or r[4] == r[5] == 0, 'unknown allocated resource count')
        need(all(v.isdigit() and (k == 'gres/gpu' or k.startswith('gres/gpu:'))
                 for k, v in g.items()), 'unknown accelerator charge')
        typed = [int(v) for k, v in g.items() if k.startswith('gres/gpu:')]
        count = int(g['gres/gpu']) if 'gres/gpu' in g else sum(typed)
        need(not typed or 'gres/gpu' not in g or count == sum(typed), 'GPU count disagrees')
        cpu += r[4]*r[5]; gpu += count*r[5]
    return cpu, gpu


class Audit:
    def __init__(self, cfg):
        self.c = cfg; self.findings = []; self.missing = {}; self.opaque = {}; self.facts = {}
        self.full = {}; self.local = {}; self.root = None

    def artifact(self, key, maximum=128*1024**2):
        row = self.c['artifacts'][key]
        need(set(row) == {'path', 'sha256'}, 'independent artifact path/pin pair required')
        return file_bytes(Path(row['path']).resolve(), row['sha256'], maximum)

    def j(self, n):
        return decode(self.b(n))

    def b(self, n):
        name(n); need(n in self.full, 'required original member missing: '+n)
        expected = self.full[n]
        if n in self.local:
            p = self.root/n
        elif n in self.c.get('fetched', {}):
            row = self.c['fetched'][n]
            need(row['sha256'] == expected['sha256'], 'fetched independent pin mismatch')
            p = Path(row['path']).resolve()
        else:
            self.missing[n] = expected
            raise ValueError('exact saved text fetch needed: '+n)
        need(not n.endswith(('.pt', '.bin', '.npy', '.npz', '.safetensors', '.tar', '.gz')),
             'opaque payload may not be decoded')
        data = file_bytes(p, expected['sha256'], maximum=128*1024**2)
        need(len(data) == expected['bytes'], 'local evidence length differs')
        return data

    def h(self, n, expected=None):
        need(n in self.full, 'full archived member missing: '+n)
        h = self.full[n]['sha256']
        if expected is not None:
            need(h == expected, 'archived member pin differs: '+n)
        if n not in self.local:
            self.opaque[n] = dict(**self.full[n], location='authenticated remote archive',
                                  locally_hashed=False, tensor_deserialized=False)
        return h

    def subtree(self, prefix, excluded=()):
        return {n[len(prefix):]: v['sha256'] for n, v in self.full.items()
                if n.startswith(prefix) and n[len(prefix):] not in excluded}

    def custody(self):
        need(self.c['schema'] == 'ace-prospective-cpu-saved-audit-input-v1', 'input contract differs')
        need(type(self.c['job_id']) is str and self.c['job_id'].isdigit(), 'actual terminal job required')
        self.d = decode(self.artifact('disposition'))
        self.seal = decode(self.artifact('custody'))
        transport = decode(self.artifact('before'))
        later = decode(self.artifact('after'))
        need(transport['returncode'] == later['returncode'] == 0, 'original inventory transport failed')
        self.before = decode(transport['stdout']); after = decode(later['stdout'])
        for v in (self.before, after):
            need(v['terminal_gate'] is True and v.get('observation_error') is None,
                 'conclusive complete terminal observation required')
        self.receipt = decode(self.artifact('remote_custody'))
        verify = (decode(self.artifact('archive_verification'))
                  if 'archive_verification' in self.c['artifacts'] else None)
        full = self.receipt['full_inventory']; validate_inventory(full)
        local = self.receipt['local_inventory']; validate_inventory(local)
        need(full == normalized_observation_inventory(self.before)
             == normalized_observation_inventory(after), 'full original before/after archive inventory differs')
        need(self.receipt['full_backup_verified'] is True and self.receipt['full_local_copy'] is False
             and self.receipt['backup_kind'] == 'ustar', 'complete remote backup required; not full local custody')
        need(self.receipt['helper_sha256'] == self.c['helper_sha256'], 'independent helper pin differs')
        need(self.receipt['selection_policy'] == 'current-launch-text-and-namespaces-v1', 'selection policy differs')
        need(self.receipt['source_observation_sha256'] == sha(raw(self.before))
             and all(self.receipt[k] == sha(raw(full)) for k in
                     ('original_before_inventory_sha256', 'original_after_inventory_sha256', 'project_readback_inventory_sha256')),
             'authenticated original/project readback inventory binding differs')
        if verify is not None:
            need(verify['full_backup_verified'] is True and verify['full_inventory'] == full
                 and verify['local_inventory'] == local
                 and verify['receipt_sha256'] == sha(raw(self.receipt))
                 and verify['backup_archive_sha256'] == self.receipt['backup_archive_sha256']
                 and verify['helper_sha256'] == self.c['helper_sha256'], 'optional independent saved archive verification differs')
        need(all(full['files'].get(n) == v for n, v in local['files'].items()), 'selected byte outside archive')
        omitted = {n: v for n, v in full['files'].items() if n not in local['files']}
        need(self.receipt['omitted_files'] == omitted and self.receipt['omitted_bytes'] == sum(v['bytes'] for v in omitted.values()), 'omitted inventory differs')
        copies = [Path(p).resolve() for p in self.c['capsules']]
        need(len(copies) == 2 and copies[0] != copies[1], 'two distinct compact copies required')
        need(self.seal['identical'] is True and self.seal['complete_local_raw_copy'] is False, 'compact two-copy seal required')
        views = []
        for p in copies:
            view = inventory(p)
            need(view['files'].pop('custody-inventory.json')['sha256'] == self.c['artifacts']['custody']['sha256'], 'capsule seal differs')
            recorded = [v for k, v in self.seal['copies'].items() if Path(k).resolve() == p]
            need(len(recorded) == 1 and recorded[0] == view, 'complete compact capsule seal differs')
            need(inventory(p/'remote') == local, 'selected local subtree differs')
            views.append(view)
        need(views[0] == views[1], 'compact copies disagree')
        need(self.d['errors'] == self.d['failure_members'] == self.d['input_pin_mismatches'] == []
             and self.d['custody_bytes_verified'] is self.d['current_lineage_verified'] is True
             and self.d['complete_local_raw_copy'] is False, 'custody/disposition disagreement or failure retained')
        need(self.d['remote_files'] == full['files'] and self.d['remote_directories'] == full['directories']
             and self.d['remote_custody_receipt_sha256'] == sha(raw(self.receipt)), 'disposition archive binding differs')
        self.root = copies[0]/'remote'; self.full = full['files']; self.local = local['files']
        self.facts['custody'] = dict(full_raw_files=len(self.full), full_raw_bytes=sum(v['bytes'] for v in self.full.values()),
            local_selected_files=len(self.local), full_local_copy=False, archive_path=self.receipt['backup_archive_path'],
            archive_sha256=self.receipt['backup_archive_sha256'], collector_project_readback_bound=True,
            separate_saved_verify_backup_result_present=verify is not None)

    def packet(self):
        self.outer = decode(self.artifact('outer')); self.p = decode(self.artifact('inner'))
        self.packet_pin = self.c['artifacts']['inner']['sha256']
        self.admission_pin = self.c['artifacts']['admission']['sha256']
        self.admission = decode(self.artifact('admission'))
        for key, member in [('outer', 'launch/packet.json'), ('inner', 'launch/inputs/packet.json'),
                            ('admission', 'launch/inputs/admission.json'), ('result', 'launch/submission.result.json'),
                            ('remote_intent', 'launch/submission.intent.json')]:
            self.h(member, self.c['artifacts'][key]['sha256'])
        need(self.p['schema'] == 'ace-scientific-campaign-cpu-v1'
             and self.p['source_commit'] == SOURCE and self.p['freeze_sha256'] == FREEZE
             and self.p['resources'] == RESOURCES and self.p['deadlines'] == DEADLINES
             and (self.p['numeric_work_seconds'], self.p['numeric_timeout_seconds']) == (50, 55), 'frozen CPU contract differs')
        need(self.admission['packet_sha256'] == self.packet_pin and self.admission['resources'] == RESOURCES
             and self.admission['deadlines'] == DEADLINES and self.admission['source_commit'] == SOURCE
             and self.admission['freeze_sha256'] == FREEZE, 'original CPU admission differs')
        need(self.p['files'][ENTRY] == ENTRY_SHA, 'frozen producer source differs')
        for n, h in self.p['files'].items():
            self.h('launch/inputs/'+n, h)
        for n, h in self.outer['new_input_files'].items():
            self.h('launch/'+n, h)
        failures = [n for n in self.full if n.startswith(CPU) and
                    (PurePosixPath(n).name.endswith('failure.json') or PurePosixPath(n).name in
                     ('publication-interruption.json', 'worker.lifecycle-errors.json'))]
        need(not failures, 'whole original failed attempts retained; success blocked: '+repr(failures))
        self.facts['packet'] = dict(inner_sha256=self.packet_pin, entry_sha256=ENTRY_SHA,
                                   captured_inputs=len(self.p['files']), source_commit=SOURCE)

    def ancestor(self):
        accepted = decode(self.artifact('training_acceptance'))
        seal_raw = self.artifact('training_seal'); seal = decode(seal_raw)
        original = decode(self.artifact('training_inventory'))
        need(original['returncode'] == 0, 'accepted original training inventory unavailable')
        original = decode(original['stdout'])
        need(accepted['schema'] == 'ace-scientific-training-gpu-acceptance-v1'
             and accepted['status'] == 'scientific_training_gpu_accepted' and accepted['required_findings'] == 0
             and accepted['complete_custody_verified'] is True and accepted['training_gpu_qualified'] is True
             and accepted['scientific_utility_qualified'] is False and accepted['scientific_freeze_sha256'] == FREEZE
             and accepted['source_commit'] == SOURCE and accepted['custody_inventory_sha256'] == sha(seal_raw)
             and accepted['inventory_transport_sha256'] == self.c['artifacts']['training_inventory']['sha256'],
             'independently accepted ancestor pins differ')
        self.h('launch/inputs/gpu-training-acceptance.json', self.c['artifacts']['training_acceptance']['sha256'])
        need(seal['identical'] is True and len(seal['copies']) == 2, 'original full ancestor seal differs')
        ref_path = self.root.parent/'predecessor-reference.json'
        ref = decode(file_bytes(ref_path))
        need(ref['custody_inventory_sha256'] == sha(seal_raw)
             and ref['complete_existing_capsule_authenticated'] is True
             and ref['recursive_copy_created'] is False and ref['original_preserved'] is True
             and Path(ref['path']).resolve() == Path(self.c['artifacts']['training_seal']['path']).resolve().parent,
             'compact capsule existing ancestor reference differs')
        expected = self.p['files']
        for n, field in [('sft.adapter.pt', 'sft_adapter_sha256'), ('final.adapter.pt', 'final_adapter_sha256'),
                         ('sft.checkpoint.json', 'sft_checkpoint_sha256'), ('final.checkpoint.json', 'final_checkpoint_sha256'),
                         ('engine-terminal.json', 'engine_terminal_sha256')]:
            need(expected[n] == accepted[field], 'accepted original checkpoint role pin differs')
            self.h('launch/inputs/'+n, accepted[field])
        self.facts['ancestor'] = dict(acceptance_sha256=self.c['artifacts']['training_acceptance']['sha256'],
            full_custody_inventory_sha256=sha(seal_raw), original_raw_files=len(original['files']),
            reused_prior_acceptance=True, ancestor_raw_reaudit_performed=False, recursive_copy_claimed=False)

    def accounting(self):
        baseline = decode(self.artifact('baseline')); job = self.c['job_id']; obs = self.before
        rows = obs['related_accounting']; old = [r for r in rows if r[0] != job]
        current = [r for r in rows if r[0] == job]
        need(baseline['terminal_jobs'] == len(old) == 45 and len(rows) == 46 and len(current) == 1
             and baseline['submission_journals'] == len(baseline['journal_pins']) == 92
             and normalize(old) == normalize(baseline['related_accounting']), 'exact complete 45+1 history differs')
        need(all(r[2].split()[0] in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
                                   'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED') for r in normalize(rows)), 'nonterminal historical job')
        r = normalize(current)[0]; self.elapsed = r[5]
        need(r == normalize([obs['parent_accounting']])[0] and r[2:4] == ['COMPLETED', '0:0']
             and r[4] == 2 and 0 < r[5] <= 6000 and r[6] == 'ucb736_asc2'
             and r[7].get('cpu') == '2' and r[7].get('node') == '1'
             and r[7].get('mem') in ('7680M', '7.5G', '7.50G')
             and not any(k.startswith('gres/') for k in r[7]), 'actual new CPU allocation differs')
        allocation = self.j(CPU+'allocation.json')
        need(allocation['returncode'] == 0, 'original running allocation command failed')
        fields = dict(re.findall(r'([^\s=]+)=(\S+)', allocation['stdout']))
        need(all(fields.get(k) == v for k, v in dict(JobId=job, Account='ucb736_asc2',
             Partition='acpu', QOS='cpu-normal', NumCPUs='2', NumNodes='1', NumTasks='1',
             Requeue='0', Restarts='0', TimeLimit='01:40:00').items()), 'original requested resources/deadline/requeue differs')
        details = obs['results']['details']; need(details['returncode'] == 0, 'saved resource details missing')
        detail_rows = [s.split('|') for s in details['stdout'].splitlines() if s.strip()]
        need(len(detail_rows) == 1 and detail_rows[0][:4] == [job, 'acpu', 'cpu-normal', '100']
             and detail_rows[0][5] == 'paco0228', 'scheduler partition/QoS/wall/owner differs')
        req = dict(part.split('=', 1) for part in detail_rows[0][4].split(',') if part)
        need(req.get('cpu') == '2' and req.get('node') == '1' and req.get('mem') in ('7680M', '7.5G', '7.50G')
             and not any(k.startswith('gres/') for k in req), 'requested CPU/memory/no-GPU differs')
        result = decode(self.artifact('result')); local = decode(self.artifact('local_intent'))
        remote = decode(self.artifact('remote_intent'))
        need(result['job_id'] == job and result['intent_sha256'] == self.c['artifacts']['remote_intent']['sha256'], 'submission result differs')
        need(set(remote) == set(local) | {'remote_utc', 'fresh_readiness'}
             and {k: remote[k] for k in local} == local, 'original local/remote intent semantic join differs')
        expected = dict(baseline['journal_pins'])
        expected[self.d['remote_root']+'/submission.intent.json'] = result['intent_sha256']
        expected[self.d['remote_root']+'/submission.result.json'] = self.c['artifacts']['result']['sha256']
        need(len(expected) == 94 and obs['journal_pins'] == expected, 'exact92+2 journals differ')
        oldcost = costs(old); newcost = costs(rows)
        need(oldcost == (baseline['closed_cpu_seconds'], baseline['closed_gpu_slice_seconds'])
             and newcost == (oldcost[0]+2*self.elapsed, oldcost[1]), 'whole history charges differ')
        need(baseline['outstanding_reservation'] == 0 and baseline['preparation_allowance_cpu_seconds'] == 3600
             and newcost[0]+3600 <= 57600 and newcost[1] <= 7200, 'remaining resource ceiling differs')
        need(self.d['closed_cpu_seconds'] == newcost[0] and self.d['closed_gpu_slice_seconds'] == newcost[1]
             and self.d['cpu_admission_after_release'] == newcost[0]+3600, 'disposition charge mismatch')
        self.facts['accounting'] = dict(jobs=46, journals=94, closed_cpu_seconds=newcost[0],
            closed_gpu_slice_seconds=newcost[1], current_allocated_cpu_seconds=2*self.elapsed,
            current_gpu_seconds=0, preparation_allowance_cpu_seconds=3600,
            cpu_admission_seconds=newcost[0]+3600, utilization_inferred=False)

    def runtime(self):
        started = self.j(CPU+'worker.started.json'); ready = self.j(CPU+'worker/work-start.json')
        resources = self.j(CPU+'worker.resources.json'); clean = self.j(CPU+'worker.cleanup.json')
        supervisor = self.j(CPU+'supervisor.json'); stage = self.j(CPU+'stage.json')
        origin = started['started_monotonic']; self.started = started; self.stage = stage
        need(type(origin) in (float, int) and math.isfinite(origin), 'original finite clock required')
        need(started['startup_deadline'] == origin+180 and started['deadline_monotonic'] == origin+5940
             and started['supervisor_deadline_monotonic'] == origin+5980
             and started['pid'] == started['pgid'] == ready['pid']
             and ready['packet_sha256'] == self.packet_pin
             and origin <= ready['monotonic'] < origin+180, 'original absolute startup/work/hard clock differs')
        need(resources['returncode'] == 0 and resources['startup_observed'] is True and resources['errors'] == []
             and resources['cleanup'] == clean and clean['group_still_exists'] is False
             and clean['errors'] == [] and clean['returncode'] == 0 and clean['kill_sent'] is False,
             'natural leader and owned group cleanup not confirmed')
        need(0 <= resources['elapsed_including_cleanup_seconds'] < 5980
             and 0 <= supervisor['elapsed_seconds'] < 5980
             and supervisor['status'] == 'cpu_campaign_complete' and supervisor['job_id'] == self.c['job_id']
             and supervisor['packet_sha256'] == self.packet_pin and supervisor['admission_sha256'] == self.admission_pin,
             'parent original clock/identity differs')
        stage_pin = self.h(CPU+'stage.json'); self.stage_pin = stage_pin
        need(supervisor['stage_sha256'] == stage_pin and stage['archive_sha256'] == self.p['archive']['sha256']
             and stage['manifest_sha256'] == self.p['manifest']['sha256'], 'staged runtime pins differ')
        need(stage['version'] == 'ace-runtime-archive-v1' and stage['objects'] == 49966
             and stage['file_bytes'] == 8397575073 and stage['model_loaded'] is stage['runtime_qualified'] is False,
             'original archive staging completion differs')
        custody = self.j(CPU+'custody.json')
        need(custody == self.subtree(CPU, ('custody.json', 'supervisor.json'))
             and supervisor['custody_sha256'] == self.h(CPU+'custody.json')
             and supervisor['session'] == resources, 'original parent output custody/session differs')
        bound = self.j(ART+'runtime-stage-binding.json')
        need(bound['stage'] == stage and bound['stage_sha256'] == stage_pin and bound['contract'] == self.p['runtime']
             and bound['gpu_mapping_enabled'] is False, 'CPU runtime binding/device policy differs')
        auth = self.j(ART+'authenticated-before.json')
        need(auth == self.j(ART+'authenticated-after.json') and auth['files'] == self.p['files']
             and auth['packet_sha256'] == self.packet_pin and auth['admission_sha256'] == self.admission_pin,
             'complete before/after input authentication differs')
        need(self.j(PHASE+'training-lineage.json') == auth['roles'], 'original checkpoint role lineage differs')
        for phase in ('after-service-imports', 'after-cpu-phase'):
            snap = self.j(ART+'runtime-'+phase+'.json')
            versions = {k: v['version'] for k, v in snap['packages'].items()}
            versions.update(snap['additional_package_versions'])
            need(versions == self.p['runtime']['imported_versions'] and snap['python_version'] == '3.10.19', 'actual imported runtime versions differ')
            expected = {Path(n).stem: self.p['files'][n] for n in self.p['source_names'] if n != ENTRY}
            actual = snap['captured_project_modules']
            need(set(expected) <= set(actual) <= set(expected) | {'__main__', '__mp_main__'}
                 and all(actual[n]['sha256'] == h for n, h in expected.items()), 'actual captured module closure differs')
        self.facts['runtime'] = dict(original_origin=origin, stage_sha256=stage_pin,
            startup_elapsed=ready['monotonic']-origin, elapsed_seconds=supervisor['elapsed_seconds'],
            owned_group_closed=True, complete_runtime_qualification_by_aid=False)

    def ranker(self):
        plan = self.j('launch/inputs/numerical-plan.json'); root = PHASE+'ranker/'
        need(sha(canonical(plan)) == self.p['numerical_plan_sha256'], 'original numerical semantic plan differs')
        need(plan == self.j(PHASE+'numeric-training-plan.json') == self.j(root+'plan.json'), 'matched numerical plan differs')
        need(plan['feature_width'] == 8891 and plan['parameter_count'] == 1154817
             and plan['checkpoint_steps'] == [159, 278], 'source-derived ranker width/size/checkpoint schedule differs')
        t = self.j(root+'terminal.json'); train = self.j(root+'training_complete.json')
        need(self.h(root+'terminal.json') == self.h(root+'terminal-candidate.json')
             and t['status'] == 'complete' and t['scope'] == 'policy_train'
             and (t['completed_steps'], t['sft_steps'], t['dpo_steps']) == (279, 160, 119)
             and t['plan_sha256'] == self.p['numerical_plan_sha256']
             and t['exposure_sha256'] == plan['exposure_sha256'] and t['environment_responses'] == 0
             and t['retry_allowed'] is t['scientific_provenance_qualified'] is False, 'ranker terminal differs')
        self.h(root+'training_complete.json', t['training_receipt_sha256'])
        need(train['plan_sha256'] == t['plan_sha256'] and train['exposure_sha256'] == t['exposure_sha256']
             and train['completed_steps'] == len(train['steps']) == 279, 'training receipt differs')
        expected_steps = {root+f'step-{i:04d}.complete.json' for i in range(279)}
        need({n for n in self.full if n.startswith(root+'step-') and n.endswith('.complete.json')} == expected_steps,
             'exact279 completed step roster differs')
        need(len(plan['exposure']['numeric_schedule']) == 279, 'exposure schedule length differs')
        for i, row in enumerate(train['steps']):
            need(self.j(root+f'step-{i:04d}.complete.json') == row and row['index'] == i
                 and all(row[k] == v for k, v in plan['exposure']['numeric_schedule'][i].items())
                 and row['optimizer_started'] is row['optimizer_returned'] is True
                 and math.isfinite(row['loss']) and math.isfinite(row['gradient_norm']), 'ranker scheduled update differs')
            gs = self.j(root+f'step-{i:04d}.gradients.json')['parameters']
            need(gs and all(g['present'] is True and g['finite'] is True for g in gs), 'ranker gradient evidence differs')
        need(set(t['exports']) == {'sft', 'dpo'} and set(train['checkpoints']) == {'initial', 'sft', 'final'}, 'checkpoint roles differ')
        for role, e in t['exports'].items():
            need(e['path'] == role+'.ranker.pt' and e['all_training_score_rows_equal'] is True
                 and e['training_receipt_sha256'] == t['training_receipt_sha256'], 'saved ranker reload equality differs')
            self.h(root+e['path'], e['sha256'])
        for role, h in train['checkpoints'].items():
            self.h(root+role+'.state.pt', h)
        ref = self.j(root+'reference.json')
        need(ref['completed_sft_steps'] == 160 and ref['parameter_signature'] == train['reference_signature']
             and ref['sft_checkpoint_sha256'] == train['checkpoints']['sft'], 'fixed SFT reference differs')
        self.h(root+'reference.pt', ref['state_sha256'])
        self.facts['ranker'] = dict(sft_updates=160, dpo_updates=119, total_updates=279,
            exposure_sha256=t['exposure_sha256'], checkpoints='full authenticated archive hashes; not tensor decoding')

    def journal(self, member):
        data = self.b(member); need(data.endswith(b'\n'), 'partial journal tail')
        previous = None; monotonic = -1; reserves = {}; returned = set(); proposals = {}; spent = 0; closed = False
        for i, line in enumerate(data.splitlines()):
            parts = object_parts(line)
            fields = {k: decode(v) for k, v in parts.items() if k in {
                'event', 'previous_sha256', 'sha256', 'monotonic_ns', 'budget_units', 'request_id',
                'cost_units', 'action_id', 'proposal_id', 'spent_units', 'requests', 'returned_responses', 'provenance'}}
            # Source writes sorted compact canonical JSON. Hash raw retained value spans,
            # including opaque measurement values, without interpreting those values.
            encoded = b'{'+b','.join(canonical(k)+b':'+parts[k] for k in sorted(parts))+b'}'
            unsigned = b'{'+b','.join(canonical(k)+b':'+parts[k] for k in sorted(parts) if k != 'sha256')+b'}'
            need(encoded == line and fields['sha256'] == sha(unsigned)
                 and fields['previous_sha256'] == previous and fields['monotonic_ns'] >= monotonic,
                 'original charged journal hash/clock chain differs')
            previous = fields['sha256']; monotonic = fields['monotonic_ns']; event = fields['event']
            need(not closed, 'event after terminal journal event')
            if i == 0:
                need(event == 'begin' and fields['budget_units'] == 32, '8initial+24acquisition budget differs')
            elif event == 'proposal':
                pid = fields['proposal_id']; need(pid == len(proposals), 'proposal order differs')
                proposals[pid] = fields['provenance']
            elif event == 'reserve':
                rid = fields['request_id']; units = fields['cost_units']
                need(rid == len(reserves) and fields['proposal_id'] in proposals and type(units) is int and units > 0,
                     'reservation order/charge differs')
                if rid < 8:
                    need(proposals[fields['proposal_id']]['kind'] == 'shared_initial_schedule' and units == 1, 'charged shared8 differs')
                reserves[rid] = units; spent += units; need(spent <= 32, 'charged cap exceeded')
            elif event == 'response':
                rid = fields['request_id']
                need(rid in reserves and rid not in returned and fields['cost_units'] == reserves[rid], 'unjoined/duplicate response')
                if rid < 8: need(fields['action_id'] == 'observe_all', 'initial action differs')
                returned.add(rid)
            elif event in ('stop', 'closed'):
                need(fields['spent_units'] == spent, 'terminal spent differs'); closed = True
            else:
                raise ValueError('failed/rejected/unknown charged event retained: '+event)
        need(closed and len(reserves) >= 8 and returned == set(reserves), 'unclosed/unresolved charged attempt')
        return dict(responses=len(returned), cost_units=spent, initial_responses=8, initial_cost_units=8,
                    acquisition_cost_units=spent-8, journal_sha256=self.h(member), values_decoded=False)

    def attempts(self):
        attempts = self.j(PHASE+'attempt-pins.json'); self.attempt_pins = attempts
        need(set(attempts) == set(POLICIES[:5]), 'five original CPU attempts required')
        pins = self.j(CAMPAIGN+'source-plan-pins.json')
        need(len(pins) == 24 and [r['world_id'] for r in pins] == list(range(202611070100, 202611070124)), '24 ordered development IDs differ')
        roots = {n[len(CAMPAIGN):].split('/')[0] for n in self.full if n.startswith(CAMPAIGN)}
        need(roots == {'campaign.json', 'source-plan-pins.json', 'start.json'} | {f'policy-{i:02d}' for i in range(5)}, 'campaign root/attempt membership differs')
        self.cells = []; self.processes = []
        for pi, policy in enumerate(POLICIES[:5]):
            base = CAMPAIGN+f'policy-{pi:02d}/'; a = self.j(base+'attempt.json')
            self.h(base+'attempt.json', attempts[policy]); self.h(base+'attempt.candidate.json', attempts[policy])
            need(a['policy_id'] == policy and a['phase'] == 'cpu' and a['source_plan_pins'] == pins
                 and a['errors'] == [] and a['retry_allowed'] is a['private_scores_computed'] is False
                 and a['service'] == dict(status='not_started', file=None, sha256=None)
                 and len(a['arms']) == 24 and a['evidence_sha256'] == self.subtree(base, ('attempt.json', 'attempt.candidate.json')),
                 'original CPU attempt failed or custody differs')
            for wi, arm in enumerate(a['arms']):
                need(arm['status'] == 'completed' and arm['reason'] is None and arm['path'] == f'world-{wi:02d}'
                     and arm['world_id'] == pins[wi]['world_id'], 'whole24 arm roster differs')
                path = base+arm['path']+'/'; receipt = self.j(path+'arm-receipt.json')
                self.h(path+'arm-receipt.json', arm['receipt_sha256']); self.h(path+'arm-receipt.candidate.json', arm['receipt_sha256'])
                need(receipt['status'] == 'raw_ready' and receipt['plan_sha256'] == pins[wi]['sha256']
                     and receipt['policy_id'] == policy and receipt['new_training_updates'] == 0
                     and receipt['private_scores_computed'] is receipt['retry_allowed'] is receipt['scientific_provenance_qualified'] is False
                     and receipt['source_evidence_sha256'] == self.subtree(path, ('arm-receipt.json', 'arm-receipt.candidate.json')), 'arm source/custody/scope differs')
                for n, h in receipt['input_sha256'].items():
                    self.h(path+'sealed-inputs/'+name(n), h)
                diag = self.j(path+'execution-diagnostics.json')
                need(diag['errors'] == [] and diag['episode_returned_cleanly'] is diag['adapters_closed_cleanly'] is True, 'episode/adapter cleanup differs')
                counts = self.journal(path+'episode/journal.jsonl')
                terminal = object_parts(self.b(path+'episode/terminal.json'))
                top = {k: decode(terminal[k]) for k in ('status', 'initialized_responses', 'policy_decisions', 'scientific_evaluation_performed')}
                final = public(terminal['final_state'], {'spent_units', 'remaining_units', 'requests', 'returned_responses', 'terminal'})
                need(top['status'] == 'complete' and top['initialized_responses'] == 8 and top['scientific_evaluation_performed'] is False
                     and final == dict(spent_units=counts['cost_units'], remaining_units=32-counts['cost_units'],
                                       requests=counts['responses'], returned_responses=counts['responses'], terminal=True), 'episode terminal charge totals differ')
                procs = sorted(n for n in self.full if n.startswith(path) and re.search(r'/decision-\d{4}\.process\.json$', n))
                need(len(procs) == top['policy_decisions'], 'saved process decision count differs')
                for n in procs:
                    process = self.j(n); need(process.get('returncode') == 0, 'failed numerical process retained')
                    # The input is canonical JSON plus newline; hash it without decoding histories.
                    inp = self.b(n.replace('.process.json', '.input.json'))
                    need(inp.endswith(b'\n') and sha(inp[:-1]) == process['input_sha256'], 'original public input byte pin differs')
                    self.processes.append(process)
                self.cells.append(dict(policy_id=policy, world_id=arm['world_id'], policy_decisions=len(procs), **counts))
        need(len(self.cells) == 120 and 120 <= len(self.processes) <= 2880, '120 cells/bounded decisions differ')
        self.facts['campaign'] = dict(completed_cpu_cells=120, declared_cells=192, methods=list(POLICIES),
            executed_methods=list(POLICIES[:5]), responses=sum(r['responses'] for r in self.cells),
            charged_cost_units=sum(r['cost_units'] for r in self.cells), shared_initial_cost_units=960,
            decision_processes=len(self.processes), utility_or_attainment_evaluated=False)

    def numeric(self):
        seen = set(); prefix = ART+'numeric-calls/'
        for process in self.processes:
            command = process['command']; split = command.index('--')
            kind = command[command.index('--numeric-entry')+1]; argv = command[split+1:]
            wid = int(command[command.index('--world-id')+1])
            need(kind in ('screen', 'ranker') and 202611070100 <= wid < 202611070124, 'numeric kind/world differs')
            cid = sha(canonical(dict(kind=kind, world_id=wid, argv=argv, input_sha256=process['input_sha256'])))
            need(cid not in seen, 'duplicate numeric attempt'); seen.add(cid); d = prefix+cid+'/'
            need(set(self.subtree(d)) == {'before.json', 'after.json', 'input.utf8', 'command.json', 'stdout.log', 'stderr.log', 'terminal.json'}, 'numeric evidence roster differs')
            t = self.j(d+'terminal.json'); launch = self.j(d+'command.json')
            expected_sources = {n: self.p['files'][n] for n in NUMERIC_SOURCES[kind] | {ENTRY}}
            need(t['status'] == 'complete' and t['call_id'] == cid and t['packet_sha256'] == self.packet_pin
                 and t['input_sha256'] == process['input_sha256'] == self.h(d+'input.utf8')
                 and t['stdout_sha256'] == sha(process['stdout'].encode()) == self.h(d+'stdout.log')
                 and t['stderr_sha256'] == sha(process['stderr'].encode()) == self.h(d+'stderr.log')
                 and 0 <= t['elapsed_seconds'] < 50 and t['cpu_seconds'] >= 0 and t['retry_allowed'] is False
                 and launch['entry'] == kind and launch['argv'] == argv and launch['world_id'] == wid
                 and launch['packet_sha256'] == self.packet_pin and launch['stage_sha256'] == self.stage_pin
                 and launch['source_sha256'] == t['source_sha256']
                 and t['source_sha256'] == expected_sources, 'numeric source/call/response/time differs')
            before = self.j(d+'before.json'); after = self.j(d+'after.json')
            required = {'numpy'} | ({'torch'} if kind == 'ranker' else set())
            need(before['versions'] == before['used_module_paths'] == before['source_sha256'] == {}
                 and after['versions'] == {n: self.p['runtime']['imported_versions'][n] for n in required}
                 and after['source_sha256'] and set(after['source_sha256']) <= NUMERIC_SOURCES[kind]
                 and all(t['source_sha256'].get(n) == h for n, h in after['source_sha256'].items())
                 and before['pid'] == after['pid'], 'numeric import/source identity differs')
            for snap in (before, after):
                need(snap['prefix'] == self.stage['staged_prefix'] and snap['python_version'] == '3.10.19'
                     and type(snap['pid']) is int and snap['pid'] > 0 and snap['pid'] != self.started['pgid']
                     and snap['pgid'] == self.started['pgid'] and snap['archive_rescanned'] is snap['llm_model_or_adapter_loaded'] is False
                     and all(PurePosixPath(self.stage['staged_prefix']) in PurePosixPath(p).parents for p in
                             [*snap['used_module_paths'].values(), *snap['standard_library_paths'].values()]), 'numeric prefix/owned group differs')
        need({n[len(prefix):].split('/')[0] for n in self.full if n.startswith(prefix)} == seen, 'unmatched/extra numeric attempt')

    def terminal(self):
        phase = self.j(PHASE+'terminal.json'); t = self.j(ART+'terminal.json')
        need(phase['status'] == 'cpu_phase_complete' and phase['failed_policies'] == []
             and phase['attempt_pins'] == self.attempt_pins and phase['numeric_training_updates'] == 279
             and phase['completed_cpu_cells'] == 120 and phase['declared_cells'] == 192
             and phase['llm_training_updates'] == phase['gpu_services_started'] == 0
             and phase['private_scores_computed'] is phase['scientific_qualified'] is phase['retry_allowed'] is False, 'phase terminal differs')
        expected = dict(schema='ace-scientific-campaign-cpu-v1', status='cpu_campaign_complete',
            packet_sha256=self.packet_pin, admission_sha256=self.admission_pin, runtime_stage_sha256=self.stage_pin,
            phase_terminal_sha256=self.h(PHASE+'terminal.json'), attempt_pins=self.attempt_pins,
            numeric_training_updates=279, completed_cpu_cells=120, declared_cells=192,
            llm_training_updates=0, llm_model_or_adapter_loaded=False, tokenizer_calls=0,
            worlds_generated=0, gpu_services_started=0, private_scores_computed=False,
            scientific_qualified=False, retry_allowed=False, external_acceptance_required=True)
        need(t == expected, 'exact worker terminal differs')
        events = [self.j(n) for n in sorted(self.full) if n.startswith(ART+'phase-') and '/' not in n[len(ART):]]
        names = [e['phase'] for e in events]
        order = ('service_cpu_import_start', 'service_cpu_import_complete', 'cpu_ranker_training_start',
                 'cpu_ranker_training_complete', 'campaign_frozen_before_responses')
        need(all(names.count(n) == 1 for n in order) and [names.index(n) for n in order] == sorted(names.index(n) for n in order)
             and [e['policy_id'] for e in events if e['phase'] == 'cpu_policy_start'] == list(POLICIES[:5])
             and [e['policy_id'] for e in events if e['phase'] == 'cpu_policy_complete'] == list(POLICIES[:5])
             and names.index('cpu_policy_start') > names.index(order[-1]), 'original phase/order differs')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True); ap.add_argument('--config-sha256', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args(); out = Path(a.output).absolute()
    need(not out.exists(), 'new exclusive audit attempt output required')
    out.mkdir(mode=0o700)
    started = time.monotonic(); cpu = time.process_time(); report = dict(acceptance=False,
        scientific_qualified=False, machine_checks_complete=False, mandatory_independent_gates=list(MANUAL_GATES),
        acceptance_keys=ACCEPTANCE_KEYS, errors=[], completed_checks=[])
    audit = None; code = 1
    try:
        cfg_raw = file_bytes(Path(a.config).resolve(), a.config_sha256)
        (out/'input-contract.json').write_bytes(cfg_raw)
        (out/'audit-source.py').write_bytes(Path(__file__).read_bytes())
        audit = Audit(decode(cfg_raw))
        for method in ('custody', 'packet', 'ancestor', 'accounting', 'runtime', 'ranker', 'attempts', 'numeric', 'terminal'):
            getattr(audit, method)(); report['completed_checks'].append(method)
        report['machine_checks_complete'] = True; code = 0
    except BaseException as exc:
        report['errors'].append(dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
    finally:
        if audit is not None:
            report.update(facts=audit.facts, necessary_exact_fetches=audit.missing, opaque_remote_evidence=audit.opaque)
            if hasattr(audit, 'cells'): report['all_cpu_cells'] = audit.cells
        report.update(elapsed_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu,
                      peak_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      audit_source_sha256=sha(Path(__file__).read_bytes()), config_sha256=a.config_sha256,
                      retries_performed=0, remote_actions=0, scientific_execution=False)
        with (out/'report.json').open('xb') as f: f.write(raw(report))
    print(str(out/'report.json')); return code


if __name__ == '__main__':
    sys.exit(main())
