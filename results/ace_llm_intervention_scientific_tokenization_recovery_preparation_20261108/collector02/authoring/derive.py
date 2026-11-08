from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'predecessors/pull_terminal.py').read_text()
s=s.replace('Complete token01 terminal custody','Complete token02 terminal custody')
s=s.replace("scientific_tokenization_20261108/launch01'", "scientific_tokenization_20261108/launch02'")
s=s.replace("scientific_tokenization_20261108_01'", "scientific_tokenization_20261108_02'")
s=s.replace("LOCAL=LAUNCH.parent/'terminal01'", "LOCAL=LAUNCH.parent/'terminal02'")
s=s.replace("scientific-tokenization-20261108-01'", "scientific-tokenization-20261108-02'").replace("scientific-tokenization-terminal-20261108-01'", "scientific-tokenization-terminal-20261108-02'")
s=s.replace("COMMON_ROOT=", "FAILED_TOKEN_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01'\nFAILED_TOKEN_JOB='35415334'\nBUILDER_SHA='c1bf0ae82b3fe004d76ffa151bf8181a0921bfc2b8f9120bcb16f44b72cb4fbc'\nCOMMON_ROOT=",1)
s=s.replace("root_present=root.exists(),common_present=", "root_present=root.exists(),failed_token_present=Path(FAILED_TOKEN_ROOT).exists(),common_present=")
s=s.replace("('launch',root,v['root_present']),", "('launch',root,v['root_present']),('token01',Path(FAILED_TOKEN_ROOT),v['failed_token_present']),")
s=s.replace("row[1]=='ace_llm_scientific_token01'", "row[1]=='ace_llm_scientific_token02'")
s=s.replace("('launch',ROOT,'root_present'),", "('launch',ROOT,'root_present'),('token01',FAILED_TOKEN_ROOT,'failed_token_present'),")
s=s.replace("and after['common_present']==before['common_present']", "and after['failed_token_present']==before['failed_token_present']\n         and after['common_present']==before['common_present']")
s=s.replace('def main():',(p/'new_functions.txt').read_text()+'def main():',1)
s=s.replace("'packet-review','packet-review-sha256')", "'packet-review','packet-review-sha256','collector-repository-path','collector-review','collector-review-sha256','failed-token-terminal-custody','failed-token-custody-inventory-sha256')")
s=s.replace("    need(pins.get('pull_terminal.py')==a.collector_sha256,'collector must equal committed concrete packet member')",'''    collector_path=Path(a.collector_repository_path)
    need(not collector_path.is_absolute() and '..' not in collector_path.parts and collector_path.suffix=='.py',
         'safe explicit standalone collector repository path required')
    own=read_metadata(Path(__file__),a.collector_sha256)
    git=subprocess.run(['git','--no-replace-objects','show',a.packet_commit+':'+collector_path.as_posix()],cwd=REPO,capture_output=True,timeout=20)
    need(git.returncode==0 and git.stdout==own,'independent collector/full packet commit differs')
    collector_review=Path(a.collector_review).absolute();cr=read_metadata(collector_review,a.collector_review_sha256)
    git=subprocess.run(['git','--no-replace-objects','show',a.packet_commit+':'+str(collector_review.relative_to(REPO))],cwd=REPO,capture_output=True,timeout=20)
    need(cr.strip() and git.returncode==0 and git.stdout==cr,'independent collector review must match full packet commit')''')
start=s.index("    baseline_pin=pins['baseline.json']");end=s.index('    locks=[];claimed=[];',start)
s=s[:start]+(p/'recovery_admission.txt').read_text()+s[end:]
s=s.replace('COMMON_ROOT=COMMON_ROOT,FAILED_ROOT=FAILED_ROOT)', 'COMMON_ROOT=COMMON_ROOT,FAILED_ROOT=FAILED_ROOT,FAILED_TOKEN_ROOT=FAILED_TOKEN_ROOT)')
s=s.replace("write(dest/'stream_custody.py',helper_source);write(dest/'stream_custody_review.md',helper_review)", "write(dest/'stream_custody.py',helper_source);write(dest/'stream_custody_review.md',helper_review)\n            write(dest/'collector-review.md',cr)")
s=s.replace("accepted_realization_root=REALIZATION_ROOT,accepted_collection_root=COMMON_ROOT,preserved_failed_root=FAILED_ROOT,", "accepted_realization_root=REALIZATION_ROOT,accepted_collection_root=COMMON_ROOT,preserved_failed_root=FAILED_ROOT,\n            preserved_failed_token_root=FAILED_TOKEN_ROOT,token_recovery=ctx['token_recovery'],\n            collector_repository_path=a.collector_repository_path,collector_review_sha256=a.collector_review_sha256,\n            failed_token_custody_inventory_sha256=a.failed_token_custody_inventory_sha256,")
s=s.replace('        # Keep complete local and external coordinator histories,', "        shutil.copytree(prior_custody,EXT/'upstream-token01-custody',copy_function=copy_file)\n        need(inventory(prior_custody)==inventory(EXT/'upstream-token01-custody')==prior_inventory,'original token01 custody changed during copy')\n        # Keep complete local and external coordinator histories,")
s=s.replace("coordinator_bytes=sum(v['bytes'] for info in coordinator.values() for v in info['files'].values()))", "coordinator_bytes=sum(v['bytes'] for info in coordinator.values() for v in info['files'].values())+sum(v['bytes'] for v in prior_inventory['files'].values()))")
s=s.replace("        for sub in ('coordinator',):", "        (LOCAL/'upstream-token01-custody').mkdir(mode=0o700)\n        for sub in ('coordinator','upstream-token01-custody'):")
start=s.index('        for label,expected in preserved.items():');end=s.index("        if before['upstream_pins']",start)
s=s[:start]+"        if inventory(prior_custody)!=prior_inventory:errors.append('original TOKEN01 capsule changed during custody')\n        errors.extend(preserved_roots_equal(actual,preserved,preserved_directories))\n"+s[end:]
start=s.index("        prior=[r for r in before['related_accounting']");end=s.index("        for label,original in (('local-launch'",start)
s=s[:start]+'''        accounting_ok,journal_ok=accounting_extension(before,baseline,a.job_id,result['intent_sha256'],a.submission_result_sha256)
        if not accounting_ok:errors.append('exact43 prior plus current44th accounting row differs')
        if not journal_ok:errors.append('exact88 prior plus current90 journals differ')
        steps=before['results']['steps']
        step_parents=[r.split('|') for r in (steps.get('stdout') or '').splitlines() if r.startswith(a.job_id+'|')]
        if steps['returncode']!=0 or len(step_parents)!=1 or normalize([step_parents[0][:8]])!=normalize([before['parent_accounting']]):
            errors.append('full job/step accounting missing or allocation row disagrees')
'''+s[end:]
s=s.replace("        errors.extend(allocation['errors'])",'''        errors.extend(allocation['errors'])
        allocation['no_requeue']=no_requeue_record(before,a.job_id,EXT/'remote/launch/preparation',actual['files'])
        if not allocation['no_requeue']['verified']:errors.append('actual no-requeue/no-restart evidence missing/different')
        if after['results']['details']!=before['results']['details']:errors.append('scheduler request evidence changed between reads')
        try:
            prior_cost=cost(baseline['related_accounting']);total=cost(before['related_accounting'])
            need(prior_cost==(baseline['closed_cpu_seconds'],baseline['closed_gpu_slice_seconds']),'accepted baseline cost differs from rows')
            need(total==(prior_cost[0]+allocation['allocated_cpu_seconds'],prior_cost[1]+allocation['gpu_slice_seconds']),
                 'complete actual current/history cost differs')
            need(total[0]+3600<=57600 and total[1]<=7200,'actual complete-chain resource ceilings exceeded')
        except (ValueError,TypeError,KeyError) as exc:
            errors.append('actual cost reconciliation unresolved: '+str(exc));accounting_ok=False''')
s=s.replace("no_response_proof_sha256=pins['no-response-proof.json'],preserved_upstream_roots=preserved,", "no_response_proof_sha256=pins['no-response-proof.json'],preserved_upstream_roots=preserved,preserved_upstream_directories=preserved_directories,\n            token_recovery=ctx['token_recovery'],failed_token_root=FAILED_TOKEN_ROOT,\n            failed_token_custody_inventory_sha256=a.failed_token_custody_inventory_sha256,")
# Reuse accepted training collector best-effort partial copying; never overwrite conflicting originals.
old=Path('/tmp/ace-scientific-training-gpu-collector-hxb4yf1k/pull_terminal.py').read_text()
a=old.index('            partial_errors=[]');z=old.index('        raise\n    finally:',a)
partial=old[a:z].replace("('remote','coordinator','upstream-cpu-custody')", "('remote','coordinator','upstream-token01-custody','stream-attempt')")
a=s.index("            try:save('collector_failure.json'");z=s.index('        raise\n    finally:',a)
s=s[:a]+partial+s[z:]
(p/'pull_terminal.py').write_text(s)
