from pathlib import Path
import shutil
r=Path(Path('/tmp/ace-scientific-training-gpu-token02-current.txt').read_text().strip());p=r/'collector/pull_terminal.py'
s=p.read_text()
s=s.replace("CPU_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01'", "CPU_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_02'\nFAILED_CPU_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01'\nFAILED_CPU_JOB='35415334'",1)
s=s.replace("cpu_present=Path(CPU_ROOT).exists(),","cpu_present=Path(CPU_ROOT).exists(),failed_token_present=Path(FAILED_CPU_ROOT).exists(),",1)
s=s.replace("('cpu',Path(CPU_ROOT),v['cpu_present']),","('cpu',Path(CPU_ROOT),v['cpu_present']),('failed-token01',Path(FAILED_CPU_ROOT),v['failed_token_present']),",1)
s=s.replace("('cpu',CPU_ROOT,'cpu_present'),","('cpu',CPU_ROOT,'cpu_present'),('failed-token01',FAILED_CPU_ROOT,'failed_token_present'),",1)
s=s.replace("and after['cpu_present']==before['cpu_present']", "and after['failed_token_present']==before['failed_token_present'] and after['cpu_present']==before['cpu_present']",1)
s=s.replace("CPU_ROOT=CPU_ROOT).items()", "CPU_ROOT=CPU_ROOT,FAILED_CPU_ROOT=FAILED_CPU_ROOT).items()",1)
s=s.replace("len(prior)==43 and len(current)==1 and len(observation['related_accounting'])==44", "len(prior)==44 and len(current)==1 and len(observation['related_accounting'])==45",1)
s=s.replace("len(baseline['journal_pins'])==88 and len(journals)==90", "len(baseline['journal_pins'])==90 and len(journals)==92",1)
s=s.replace("baseline['terminal_jobs']==43", "baseline['terminal_jobs']==44",1).replace("baseline['submission_journals']==88 and len(baseline['related_accounting'])==43 and len(baseline['journal_pins'])==88", "baseline['submission_journals']==90 and len(baseline['related_accounting'])==44 and len(baseline['journal_pins'])==90",1)
s=s.replace('actual43-job/88-journal','actual44-job/90-journal').replace('actual43/88 accepted baseline','actual44/90 accepted baseline').replace('exact43 prior plus current44th','exact44 prior plus current45th').replace('exact88 prior plus current90','exact90 prior plus current92')
s=s.replace("baseline['input_pins']['predecessor']==pins['previous-baseline.json']", "baseline['input_pins']['predecessor']==pins['token-failure-baseline.json']",1)
s=s.replace("a.job_id not in (accepted['job_id'],collection['job_id'],'35407428')", "a.job_id not in (accepted['job_id'],collection['job_id'],'35407428',FAILED_CPU_JOB)",1)
# Add the separately authenticated complete failed root without guessing the producer's top-level label.
start=s.index('    # Independent original CPU capsule pins')
s=s[:start]+'''    token_failure=token_failure_context(pins,packet,intent,baseline,original_files,cpu_result)
    preserved['failed-token01']=token_failure['files']
'''+s[start:]
s=s.replace("p.startswith(COMMON_ROOT+'/'))", "p.startswith(COMMON_ROOT+'/') or p.startswith(FAILED_CPU_ROOT+'/'))",1)
s=s.replace("        label='cpu' if p.startswith(CPU_ROOT+'/') else 'common';root=CPU_ROOT if label=='cpu' else COMMON_ROOT", "        label,root=next((label,root) for label,root in (('cpu',CPU_ROOT),('common',COMMON_ROOT),('failed-token01',FAILED_CPU_ROOT)) if p.startswith(root+'/'))",1)
s=s.replace("            accepted_cpu_root=CPU_ROOT,accepted_collection_root=COMMON_ROOT,preserved_failed_root=FAILED_ROOT,", "            accepted_cpu_root=CPU_ROOT,accepted_collection_root=COMMON_ROOT,preserved_failed_root=FAILED_ROOT,\n            preserved_failed_token_root=FAILED_CPU_ROOT,token_recovery=packet['token_recovery'],",1)
s=s.replace("        if before['upstream_pins']!=upstream_pins", "        if sorted(n[len('failed-token01/'):] for n in actual['directories'] if n.startswith('failed-token01/'))!=token_failure['directories']:\n            errors.append('complete failed TOKEN01 directory custody changed/missing')\n        if before['upstream_pins']!=upstream_pins",1)
s=s.replace("            operational_amendment_sha256=pins['operational-amendment.json'],preserved_upstream_roots=preserved,", "            operational_amendment_sha256=pins['operational-amendment.json'],preserved_upstream_roots=preserved,\n            token_recovery=packet['token_recovery'],preserved_failed_token_root=FAILED_CPU_ROOT,",1)
p.write_text(s)
shutil.copyfile(__file__,r/'derive-collector.py')
print(p)
