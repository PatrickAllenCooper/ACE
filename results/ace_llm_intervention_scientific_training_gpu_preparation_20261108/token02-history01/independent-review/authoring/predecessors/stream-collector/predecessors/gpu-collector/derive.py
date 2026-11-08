from pathlib import Path
r=Path(__file__).parent;s=(r/'predecessors/token-collector.py').read_text()
s=s.replace('Complete token01 terminal custody, never independent token/scientific acceptance.', 'Complete scientific GPUtrain terminal custody, never training/scientific acceptance.')
s=s.replace('Only public task terminal counts are decoded after custody; no payload/scoring/model execution.', 'Only allowlisted public telemetry scalars are decoded after custody; score values and tensors stay opaque.')
s=s.replace('import time\n','import time\nimport math\n')
s=s.replace("LAUNCH=REPO/'results/ace_llm_intervention_scientific_tokenization_20261108/launch01'", "LAUNCH=REPO/'results/ace_llm_intervention_scientific_training_gpu_20261108/launch01'")
s=s.replace("LAUNCH_EXT=BASE/'ACE-LLM-intervention-scientific-tokenization-20261108-01'", "LAUNCH_EXT=BASE/'ACE-LLM-intervention-scientific-training-gpu-20261108-01'")
s=s.replace("EXT=BASE/'ACE-LLM-intervention-scientific-tokenization-terminal-20261108-01'", "EXT=BASE/'ACE-LLM-intervention-scientific-training-gpu-terminal-20261108-01'")
s=s.replace("ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01'", "ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01'\nCPU_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01'")
s=s.replace("root_present=root.exists(),common_present=", "root_present=root.exists(),cpu_present=Path(CPU_ROOT).exists(),common_present=")
s=s.replace("(('launch',root,v['root_present']),('common'", "(('launch',root,v['root_present']),('cpu',Path(CPU_ROOT),v['cpu_present']),('common'")
# Replace CPU-only resources/public task reader below with new GPU-only functions.
a=s.index('def resources(');z=s.index('def validate_remote_intent(',a);s=s[:a]+(r/'gpu_metadata.txt').read_text()+'\n\n'+s[z:]
a=s.index('def task_counts(');z=s.index('def main():',a);s=s[:a]+s[z:]
s=s.replace("'collector-sha256','packet-review','packet-review-sha256'", "'collector-sha256','collector-repository-path','collector-review','collector-review-sha256','cpu-terminal-custody','cpu-custody-inventory-sha256','packet-review','packet-review-sha256'")
s=s.replace("packet['schema']=='ace-scientific-tokenization-launch-v1'", "packet['schema']=='ace-scientific-training-gpu-launch-v1'")
s=s.replace("    need(pins.get('pull_terminal.py')==a.collector_sha256,'collector must equal committed concrete packet member')", """    collector_path=Path(a.collector_repository_path)
    need(not collector_path.is_absolute() and '..' not in collector_path.parts and collector_path.suffix=='.py',
         'explicit safe collector repository path required')
    own=read_metadata(Path(__file__),a.collector_sha256)
    git=subprocess.run(['git','--no-replace-objects','show',a.packet_commit+':'+collector_path.as_posix()],cwd=REPO,capture_output=True,timeout=20)
    need(git.returncode==0 and git.stdout==own,'standalone collector must equal independently pinned full packet commit bytes')
    collector_review=Path(a.collector_review).absolute();cr=read_metadata(collector_review,a.collector_review_sha256)
    git=subprocess.run(['git','--no-replace-objects','show',a.packet_commit+':'+str(collector_review.relative_to(REPO))],cwd=REPO,capture_output=True,timeout=20)
    need(cr.strip() and git.returncode==0 and git.stdout==cr,'independent collector review must be in packet commit')""")
s=s.replace("baseline['terminal_jobs']==42", "baseline['terminal_jobs']==43").replace("baseline['submission_journals']==86", "baseline['submission_journals']==88")
s=s.replace("len(baseline['related_accounting'])==42", "len(baseline['related_accounting'])==43").replace("len(baseline['journal_pins'])==86", "len(baseline['journal_pins'])==88")
s=s.replace('actual42-job/86-journal','actual43-job/88-journal')
a=s.index("    accepted=decode(read_metadata(LAUNCH/'realization-acceptance.json'");z=s.index('    locks=[];',a)
s=s[:a]+(r/'upstream_binding.txt').read_text()+s[z:]
s=s.replace('COMMON_ROOT=COMMON_ROOT,FAILED_ROOT=FAILED_ROOT', 'COMMON_ROOT=COMMON_ROOT,FAILED_ROOT=FAILED_ROOT,CPU_ROOT=CPU_ROOT')
s=s.replace("accepted_realization_root=REALIZATION_ROOT,accepted_collection_root=COMMON_ROOT,preserved_failed_root=FAILED_ROOT,", "accepted_cpu_root=CPU_ROOT,accepted_collection_root=COMMON_ROOT,preserved_failed_root=FAILED_ROOT,\n            collector_repository_path=a.collector_repository_path,collector_review_sha256=a.collector_review_sha256,\n            cpu_custody_inventory_sha256=a.cpu_custody_inventory_sha256,")
s=s.replace("        # Keep complete local", """        for dest in claimed:
            write(dest/'collector-review.md',cr)
        shutil.copytree(cpu_custody,EXT/'upstream-cpu-custody',copy_function=copy_file)
        need(inventory(cpu_custody)==inventory(EXT/'upstream-cpu-custody')==cpu_custody_before,'accepted CPU custody changed during copy')
        # Keep complete local""")
s=s.replace("(('launch',ROOT,before['root_present']),\n                    ('common'", "(('launch',ROOT,before['root_present']),('cpu',CPU_ROOT,before['cpu_present']),\n                    ('common'")
s=s.replace("label in ('launch','common','failed01')", "label in ('launch','cpu','common','failed01')")
s=s.replace("and after['common_present']==before['common_present']", "and after['cpu_present']==before['cpu_present'] and after['common_present']==before['common_present']")
s=s.replace("        for sub in ('remote','coordinator'):", "        (LOCAL/'upstream-cpu-custody').mkdir(mode=0o700)\n        for sub in ('remote','coordinator','upstream-cpu-custody'):")
s=s.replace("        errors=[]\n        for label", "        errors=[]\n        if inventory(cpu_custody)!=cpu_custody_before:errors.append('original accepted CPU custody changed')\n        for label")
s=s.replace("if not (claim.get('packet_sha256')==pins['realization-packet.json'] and claim.get('realization_root')==REALIZATION_ROOT", "if not (claim.get('realization_root')==REALIZATION_ROOT")
s=s.replace("'accepted training-only realization bytes changed/missing'", "'accepted original CPU/collection input bytes changed/missing'")
s=s.replace('exact42 prior accounting','exact43 prior accounting').replace('==88 and before', '==90 and before').replace('exact86 prior plus two current','exact88 prior plus two current')
s=s.replace('allocation=resources(before,a.job_id)', "allocation=resources(before,a.job_id,EXT/'remote/launch/training',actual['files'])\n        if after['results']['details']!=before['results']['details']:\n            errors.append('scheduler request evidence changed between reads')\n        try:\n            observed_cost=cost(before['related_accounting']);prior_cost=cost(baseline['related_accounting'])\n            need(prior_cost==(baseline['closed_cpu_seconds'],baseline['closed_gpu_slice_seconds']), 'baseline total/row mismatch')\n            need(observed_cost==(prior_cost[0]+allocation['allocated_cpu_seconds'],prior_cost[1]+allocation['gpu_slice_seconds']), 'actual full-chain cost mismatch')\n            need(observed_cost[0]+3600<=57600 and observed_cost[1]<=7200,'actual cumulative resource ceiling exceeded')\n        except (ValueError,TypeError,KeyError) as exc:\n            errors.append('actual accounting unresolved: '+str(exc));accounting_ok=False")
a=s.index("        markers={n:actual['files'].get('launch/preparation/'");z=s.index("        save('disposition.json',disposition)",a)
s=s[:a]+(r/'disposition.txt').read_text()+s[z:]
# Preserve partial bytes in both mirrors even if first transfer or later verification failed.
s=s.replace("        if claimed and not sealed:\n            try:save('collector_failure.json'", """        if claimed and not sealed:
            partial_errors=[]
            if len(claimed)==2:
                for sub in ('remote','coordinator','upstream-cpu-custody'):
                    src=EXT/sub;dst=LOCAL/sub
                    if not src.exists():continue
                    try:
                        for p in sorted(src.rglob('*')):
                            need(not p.is_symlink(),'partial symlink retained only in original copy')
                            q=dst/p.relative_to(src)
                            if p.is_dir():q.mkdir(parents=True,exist_ok=True)
                            elif not q.exists():q.parent.mkdir(parents=True,exist_ok=True);copy_file(p,q)
                            else:need(fingerprint(q)==fingerprint(p),'partial mirror conflict; never overwrite')
                    except BaseException as secondary:partial_errors.append(str(secondary))
            try:save('collector_failure.json'""")
s=s.replace("retry_allowed=False,acceptance=False,partial_custody_preserved=True))", "retry_allowed=False,acceptance=False,partial_custody_preserved=True,partial_mirror_errors=partial_errors))")
(r/'pull_terminal.py').write_text(s)
