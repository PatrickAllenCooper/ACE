from pathlib import Path
import tempfile, shutil, hashlib, json, difflib
old=Path('/tmp/ace-scientific-training-gpu-collector-hxb4yf1k')
token=Path('/tmp/ace-token-stream-transport-authoring-kpeci6s1')
root=Path(tempfile.mkdtemp(prefix='ace-scientific-training-gpu-stream-',dir='/tmp')).resolve()
(root/'predecessors').mkdir()
shutil.copytree(old,root/'predecessors/gpu-collector')
for n in ('pull_terminal.py','test_transport_integration.py','handoff.md'):
    shutil.copyfile(token/n,root/'predecessors'/('token-'+n))
for n in ('stream_custody.py','stream_custody_review.md'):shutil.copyfile(token/n,root/n)
reviews=Path('/Users/pat/code/ACE/docs/development/guidance/reviews')
for n in ('ace_llm_intervention_scientific_training_gpu_collector_review_2026-11-08.md','ace_llm_intervention_scientific_token_stream_integration_review_2026-11-08.md'):
    shutil.copyfile(reviews/n,root/'predecessors'/n)
source=(old/'pull_terminal.py').read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='9346be8d3bc3f9060f1bbc24c9dbe6285c4bddfdd11e5a08951024ed84e5b55a'
source=source.replace('import time\n','import time\nimport shlex\nimport types\n',1)
transport=(token/'pull_terminal.py').read_text().split("STREAM_HELPER_SHA=",1)[1].split('\n\ndef main():',1)[0]
transport='STREAM_HELPER_SHA='+transport
transport=transport.replace("module=types.ModuleType('pinned_token_stream_transport')","module=types.ModuleType('pinned_gpu_training_stream_transport')")
transport=transport.replace("str(LAUNCH/'stream_custody.py')","str(Path(__file__).resolve().parent/'stream_custody.py')")
transport=transport.replace("('launch',ROOT,'root_present'),('common'","('launch',ROOT,'root_present'),('cpu',CPU_ROOT,'cpu_present'),('common'",1)
transport=transport.replace("and after['common_present']==before['common_present']","and after['cpu_present']==before['cpu_present'] and after['common_present']==before['common_present']",1)
binding='''def authenticate_stream_files(commit,helper_repository_path,review_repository_path):
    """Standalone outer-only binding; no added child/core packet members."""
    need(re.fullmatch('[0-9a-f]{40}',commit) is not None,'full packet commit required for stream binding')
    authenticated=[]
    for name,expected,relative,suffix in (
            ('stream_custody.py',STREAM_HELPER_SHA,helper_repository_path,'.py'),
            ('stream_custody_review.md',STREAM_REVIEW_SHA,review_repository_path,'.md')):
        path=Path(relative)
        need(type(relative) is str and relative and not path.is_absolute() and '..' not in path.parts
             and path.as_posix()==relative and path.suffix==suffix,'safe explicit standalone stream repository path required')
        data=read_metadata(Path(__file__).resolve().parent/name,expected)
        git=subprocess.run(['git','--no-replace-objects','show',commit+':'+relative],
                           cwd=REPO,capture_output=True,timeout=20)
        need(git.returncode==0 and git.stdout==data,'stream helper/review must equal independently pinned full packet commit bytes')
        authenticated.append(data)
    return tuple(authenticated)


'''
source=source.replace('\ndef main():','\n'+transport+'\n\n'+binding+'def main():',1)
source=source.replace("'packet-review-sha256'):ap.add_argument", "'packet-review-sha256','stream-helper-repository-path','stream-review-repository-path'):ap.add_argument",1)
needle="    baseline_pin=pins['baseline.json'];baseline=decode(read_metadata(LAUNCH/'baseline.json',baseline_pin))"
source=source.replace(needle,"    helper_source,helper_review=authenticate_stream_files(a.packet_commit,a.stream_helper_repository_path,a.stream_review_repository_path)\n"+needle,1)
source=source.replace("for sub in ('remote','coordinator','namespaces'):","for sub in ('coordinator','namespaces'):",1)
source=source.replace("            write(dest/'pull_terminal.py',Path(__file__).read_bytes())","            write(dest/'pull_terminal.py',Path(__file__).read_bytes())\n            write(dest/'stream_custody.py',helper_source);write(dest/'stream_custody_review.md',helper_review)",1)
source=source.replace("            collector_repository_path=a.collector_repository_path,collector_review_sha256=a.collector_review_sha256,","            collector_repository_path=a.collector_repository_path,collector_review_sha256=a.collector_review_sha256,\n            stream_helper_sha256=STREAM_HELPER_SHA,stream_review_sha256=STREAM_REVIEW_SHA,\n            stream_helper_repository_path=a.stream_helper_repository_path,stream_review_repository_path=a.stream_review_repository_path,",1)
start=source.index('        for parent in (BASE,LOCAL.parent):')
end=source.index("        (LOCAL/'upstream-cpu-custody').mkdir",start)
source=source[:start]+'''        helper=load_stream_helper(helper_source)  # Full commit/self/review/input authentication completed above.
        actual,after=stream_transfer(before,observe=observe,save=save,helper=helper,helper_source=helper_source,
            external=EXT,local=LOCAL,copy_file=copy_file,
            coordinator_bytes=sum(v['bytes'] for info in coordinator.values() for v in info['files'].values())
                              +sum(v['bytes'] for v in cpu_custody_before['files'].values()))
'''+source[end:]
source=source.replace("        for sub in ('remote','coordinator','upstream-cpu-custody'):","        for sub in ('coordinator','upstream-cpu-custody'):",1)
(root/'pull_terminal.py').write_text(source)
(root/'pull_terminal.py.diff').write_text(''.join(difflib.unified_diff((old/'pull_terminal.py').read_text().splitlines(True),source.splitlines(True),fromfile=str(old/'pull_terminal.py'),tofile=str(root/'pull_terminal.py'))))
shutil.copyfile(__file__,root/'derive.py')
(root/'preservation.json').write_text(json.dumps(dict(predecessor=str(old),predecessor_source_sha256=hashlib.sha256((old/'pull_terminal.py').read_bytes()).hexdigest(),source_unchanged=(old/'pull_terminal.py').read_bytes()==(root/'predecessors/gpu-collector/pull_terminal.py').read_bytes(),note='All predecessor files copied as independent bytes; deliberate prior fixture hardlink aliases are retained as separate byte copies, not valid aliases.'),indent=2)+'\n')
Path('/tmp/ace-scientific-training-gpu-stream-current.txt').write_text(str(root)+'\n')
print(root)
