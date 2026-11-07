"""Pure memory fault probe of exact source functions; no project imports/work."""
import ast
import hashlib
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

REPO = Path('/Users/pat/code/ACE')
source = REPO/'scripts/research/llm_intervention_campaign.py'
paired_source = REPO/'scripts/research/llm_intervention_paired_episode.py'
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
assert source_hash == '87c12c245d38c9a63ac888bf90651e46682542bfd78580cec0a05186721f0c6d'

def extract(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]

def case(mode):
    files, dirs = {}, {'root'}
    class MemoryPath:
        def __init__(self, value): self.value = value
        def __truediv__(self, name): return MemoryPath(self.value+'/'+name)
        def with_suffix(self, suffix): return MemoryPath(self.value.rsplit('.',1)[0]+suffix)
        def mkdir(self, exist_ok=False):
            if self.value in dirs and not exist_ok: raise FileExistsError(self.value)
            dirs.add(self.value)
        def read_bytes(self): return files[self.value]
    def save(path, value):
        if path.value in files: raise FileExistsError(path.value)
        files[path.value] = (json.dumps(value, sort_keys=True)+'\n').encode()
    def link(candidate, final):
        if mode == 'before_link': raise OSError('injected before publication')
        if final.value in files: raise FileExistsError(final.value)
        files[final.value] = files[candidate.value]
        if mode == 'after_link': raise KeyboardInterrupt('injected after link became visible')
    publish = extract(paired_source, '_publish', {'_save':save, 'os':SimpleNamespace(link=link)})
    # Empty mocked inputs isolate only terminal publication/exception handling.
    # They are NOT an admitted campaign; no arm, service or private scorer runs.
    campaign = {'policy_ids':[], 'services':{}, 'sources':[], 'strata_fields':[], 'scope':'artificial'}
    namespace = {
        '_root':lambda *args:(campaign,MemoryPath('root')),
        '_keys':lambda *args:None,
        '_utc':lambda:'fixed probe timestamp',
        '_error':lambda exc:type(exc).__name__+': '+str(exc),
        'paired':SimpleNamespace(_save=save, _publish=publish, _sha=lambda raw:hashlib.sha256(raw).hexdigest()),
        'summarize_screen':lambda **kwargs:{'cell_count':0},
        'VERSION':'probe', 'traceback':traceback,
    }
    finalize = extract(source, 'finalize_campaign', namespace)
    raised = None
    try:
        finalize(campaign=campaign, campaign_sha256='probe', source_plan_pins=[], output='root', attempt_pins={})
    except BaseException as exc:
        raised = type(exc).__name__
    return {'fault':mode, 'raised':raised, 'files':sorted(files),
            'published_success': 'root/finalization.json' in files,
            'published_failure': 'root/private/failure.json' in files}

results = [case(mode) for mode in ('none','before_link','after_link')]
assert results[0]['published_success'] and not results[0]['published_failure']
assert not results[1]['published_success'] and results[1]['published_failure']
assert results[2]['published_success'] and results[2]['published_failure']
print(json.dumps({'campaign_sha256':source_hash,
                  'paired_sha256':hashlib.sha256(paired_source.read_bytes()).hexdigest(),
                  'scope':'Pure memory fault injection; exact extracted finalize_campaign/_publish; no project imports, worlds, policies, subprocesses or remote work.',
                  'results':results}, indent=2))
