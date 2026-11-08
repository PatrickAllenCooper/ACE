from pathlib import Path
p=Path(__file__).parent/'build_packet.py'
s=p.read_text().replace('# Reference bindings only: the collection failure may require a separately reviewed successor.\n# No successor is silently adopted. Rebind these/source closure only after explicit reviewed pins.', '# Original science/source remain immutable; collection02 has separate, exact operational authority.')
s=s.replace("METADATA=tuple(FIXED)+", """FIXED.update({
 'operational-amendment':'fa94858dfc19c8b15c82acde167ad8fe9991d0c863c32444d017743079617855',
 'operational-adoption-review':'db2a672a4f0222a06a46d1e61ebc5bdc22b869640f570e19d36a0f699a1ddbaf',
 'collection-failure-acceptance':'00c4557a5c681904e823441ac8bb99f4dc5e169c0c7e41e764018a1e4e7af5ee',
 'no-response-proof':'59c21d63644cf7ae9765b3ff374538f2c614694dc383583f1e7661c301700998',
 'collection-failure-baseline':'eeec362db1990d5cf1fb7945708293842dc9b9283a2777697e6840ad5a2c147b'})
COLLECTION_ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_02'
TEXT_METADATA={'freeze-review','cpu-report','operational-adoption-review'}
def metadata_name(name):return name+('.md' if name in TEXT_METADATA else '.json')
METADATA=tuple(FIXED)+""")
s=s.replace("'collection-acceptance','baseline','previous-baseline','site-readiness')", "'collection-acceptance','collection-packet','collection-result','collection-disposition','collection-inventory',\n 'baseline','previous-baseline','site-readiness')")
marker='def authenticate(supplied,prepared_raw,prepared_pin,plan_pin,paths):'
helper='''def collection_lineage(m,pins):
    """Join accepted collection02 to the closed failure; never infer future costs or approval."""
    authority=m['operational-amendment'];failure=m['collection-failure-acceptance'];proof=m['no-response-proof']
    failed=m['collection-failure-baseline'];base=m['previous-baseline'];a=m['collection-acceptance']
    packet=m['collection-packet'];result=m['collection-result'];disp=m['collection-disposition'];cpu=m['cpu-launch-packet']
    need(authority['freeze_sha256']==pins['freeze'] and authority['source_commit']==SOURCE_COMMIT
        and authority['recovery_root']==COLLECTION_ROOT and authority['baseline_sha256']==pins['collection-failure-baseline']
        and authority['failure_acceptance_sha256']==pins['collection-failure-acceptance']
        and authority['no_response_proof_sha256']==pins['no-response-proof']
        and authority['namespace_admission_sha256']==pins['namespace-admission']
        and authority['namespace_claim_sha256']==pins['namespace-claim']
        and authority['scientific_inputs_unchanged'] is True and authority['other_stages_unchanged'] is True,
        'exact collection-only operational authority lineage required')
    need(failure['status']=='accepted_failed_before_response' and failure['job_id']=='35407428'
        and proof['status']=='verified_failed_before_collection_worker' and proof['failed_job_id']=='35407428'
        and proof['failure_acceptance_sha256']==pins['collection-failure-acceptance']
        and proof['accounting_sha256']==pins['collection-failure-baseline'], 'accepted failure/proof lineage differs')
    for value in (failure,proof):
        need(all(type(value[k]) is int and value[k]==0 for k in ('environment_response_attempts','environment_responses',
             'unknown_return_attempts','scientific_worlds_generated','records_created','pairs_created',
             'scientific_response_cost_units','training_updates')), 'unknown or nonzero failed science charges')
        need(value['node_local_partial_runtime_cleanup_verified'] is False
            and value['runtime_qualified_for_this_execution'] is False,'failed runtime uncertainty must remain')
    job=result['job_id']
    need(type(job) is str and re.fullmatch('[1-9][0-9]*',job) and job!='35407428'
        and job!=m['cpu-acceptance']['job_id'] and result['returncode']==0 and result['ambiguous'] is False
        and pin(result['intent_sha256']) and a['job_id']==job
        and a['collection_packet_sha256']==pins['collection-packet'] and a['disposition_sha256']==pins['collection-disposition'],
        'independent actual collection02 acceptance/result pins required')
    need(packet['remote_root']==COLLECTION_ROOT and packet['source_commit']==SOURCE_COMMIT
        and packet['resources']==authority['resources'] and packet['deadlines']==authority['deadlines']
        and packet['baseline_sha256']==pins['collection-failure-baseline']
        and packet['operational_amendment_sha256']==pins['operational-amendment']
        and a['collection']['root']==COLLECTION_ROOT+'/collection/session/worker/task/collection',
        'collection02 resource/root/amendment differs; science freeze unchanged')
    for name in ('freeze','namespace-admission','namespace-claim','operational-amendment',
                 'collection-failure-acceptance','no-response-proof'):
        need(packet['new_input_files'][metadata_name(name)]==pins[name], 'collection original authority pin differs: '+name)
    need(packet['new_input_files']['failure-baseline.json']==pins['collection-failure-baseline'],
        'collection original failure baseline differs')
    for key,name in (('operational_amendment_sha256','operational-amendment'),
                     ('operational_adoption_review_sha256','operational-adoption-review'),
                     ('collection_failure_acceptance_sha256','collection-failure-acceptance'),
                     ('no_response_proof_sha256','no-response-proof'),('collection_acceptance_sha256','collection-acceptance'),
                     ('baseline_sha256','previous-baseline')):
        need(cpu[key]==pins[name], 'CPU outer operational lineage differs: '+key)
    need(cpu['collection_job_id']==job,'CPU outer collection job differs')
    for name in ('operational-amendment','operational-adoption-review','collection-failure-acceptance','no-response-proof',
                 'collection-failure-baseline','collection-acceptance','collection-packet','collection-result','collection-disposition'):
        need(cpu['new_input_files'][metadata_name(name)]==pins[name], 'CPU captured collection lineage differs: '+name)
    need(cpu['new_input_files']['baseline.json']==pins['previous-baseline'],'CPU captured actual baseline differs')
    need(disp['job_id']==job and disp['remote_root']==COLLECTION_ROOT and disp['packet_sha256']==pins['collection-packet']
        and disp['submission_result_sha256']==pins['collection-result'] and disp['custody_bytes_verified'] is True
        and disp['current_lineage_verified'] is True and disp['errors']==[] and disp['input_pin_mismatches']==[]
        and disp['exact_prior_accounting_matched'] is True and disp['exact_prior_and_current_journals_matched'] is True
        and disp['observed_related_jobs']==42 and disp['observed_submission_journals']==86,
        'complete independently accepted collection02 custody required')
    transport=m['collection-inventory'];need(transport['returncode']==0,'collection inventory transport failed')
    observed=decode(transport['stdout']);files=observed['files']
    need(observed['terminal_gate'] is True and observed['root_present'] is True and not observed.get('observation_error')
        and len(files)==disp['raw_files'] and sum(v['bytes'] for v in files.values())==disp['raw_bytes'],
        'complete collection terminal inventory required')
    need(literal(transport['python_source'],'ROOT')==COLLECTION_ROOT and literal(transport['python_source'],'JOB')==job,
        'collection inventory captured root/job differs')
    original={'packet.json':pins['collection-packet'],'submission.intent.json':result['intent_sha256'],
              'submission.result.json':pins['collection-result'],**packet['new_input_files']}
    for name,h in original.items():
        relative('launch/'+name)
        need(pin(h) and files['launch/'+name]['sha256']==h,'original collection packet/input/journal inventory differs')
    prefix='launch/collection/session/worker/task/collection/'
    need({n[len(prefix):]:v['sha256'] for n,v in files.items() if n.startswith(prefix)}==a['collection']['files'],
        'complete accepted collection output map differs')
    need(failed['terminal_jobs']==41 and failed['submission_journals']==84
        and base['terminal_jobs']==42 and base['submission_journals']==86
        and failed['outstanding_reservation']==base['outstanding_reservation']==0
        and base['preparation_allowance_cpu_seconds']==3600 and base['cpu_cap_seconds']==57600 and base['gpu_cap_seconds']==7200
        and base['input_pins']['predecessor']==pins['collection-failure-baseline']
        and base['input_pins']['acceptance']==pins['collection-acceptance']
        and base['input_pins']['disposition']==pins['collection-disposition'],'actual41-to42 baseline chain required')
    prior=[r for r in base['related_accounting'] if r[0]!=job];new=[r for r in base['related_accounting'] if r[0]==job]
    need(len(prior)==41 and len(new)==1 and normalized(prior)==normalized(failed['related_accounting'])
        and normalized(new)==normalized([observed['parent_accounting']])
        and normalized(base['related_accounting'])==normalized(observed['related_accounting']),
        'every failed/prior row must survive collection02')
    row=normalized(new)[0];allocation=disp['allocation'];tres=dict(row[7])
    need(row[1]=='ace_llm_scientific_collect02' and row[2:4]==['COMPLETED','0:0'] and row[4]==2
        and 0<=row[5]<=900 and row[6]=='ucb736_asc2'
        and tres.get('cpu')=='2' and tres.get('node')=='1' and tres.get('mem') in ('7680M','7.5G','7.50G')
        and not any(k.startswith('gres/') for k in tres)
        and allocation['resource_contract_matched'] is True and allocation['errors']==[]
        and allocation['allocated_cpus']==2 and allocation['elapsed_seconds']==row[5]
        and allocation['allocated_cpu_seconds']==2*row[5] and allocation['gpu_count']==allocation['gpu_slice_seconds']==0,
        'actual collection02 allocation/cost differs')
    journals={**failed['journal_pins'],COLLECTION_ROOT+'/submission.intent.json':result['intent_sha256'],
              COLLECTION_ROOT+'/submission.result.json':pins['collection-result']}
    need(len(failed['journal_pins'])==84 and len(journals)==86
        and base['journal_pins']==observed['journal_pins']==journals,'all84 failed journals plus collection02 journals required')
    need(base['closed_cpu_seconds']==disp['closed_cpu_seconds']==failed['closed_cpu_seconds']+2*row[5]
        and base['closed_gpu_slice_seconds']==disp['closed_gpu_slice_seconds']==failed['closed_gpu_slice_seconds']
        and base['cpu_admission_seconds']==base['closed_cpu_seconds']+3600,
        'actual collection02 spend required, no defaults')
    return {COLLECTION_ROOT+'/'+n:h for n,h in original.items()}


'''
s=s.replace(marker,helper+marker)
s=s.replace("if n not in ('freeze-review','cpu-report','site-readiness')", "if n not in TEXT_METADATA|{'site-readiness'}")
s=s.replace("    # Metadata only until independent CPU acceptance", "    collection_originals=collection_lineage(m,pins)\n    # Metadata only until independent CPU acceptance")
s=s.replace("    cpu_dir=Path(paths['cpu-packet-relative']).parent", """    for n,h in outer['new_input_files'].items():
        relative('launch/'+n);need(pin(h),'invalid original CPU outer input pin')
        if 'launch/'+n in expected_files:need(expected_files['launch/'+n]==h,'CPU outer input collision')
        expected_files['launch/'+n]=h
    cpu_dir=Path(paths['cpu-packet-relative']).parent""")
s=s.replace("type(old['terminal_jobs']) is int and old['terminal_jobs']>0", "type(old['terminal_jobs']) is int and old['terminal_jobs']==42")
s=s.replace("type(old['submission_journals']) is int and old['submission_journals']>0", "type(old['submission_journals']) is int and old['submission_journals']==86")
s=s.replace("upstream_pins={CPU_ROOT+'/'+n.removeprefix('launch/'):h for n,h in expected_files.items()})", "upstream_pins={**collection_originals,**{CPU_ROOT+'/'+n.removeprefix('launch/'):h for n,h in expected_files.items()}})")
s=s.replace("files[n+('.md' if n in ('freeze-review','cpu-report') else '.json')]=v", "files[metadata_name(n)]=v")
s=s.replace("cpu_job_id=ctx['metadata']['cpu-acceptance']['job_id'],", "operational_amendment_sha256=ctx['pins']['operational-amendment'],\n        collection_job_id=ctx['metadata']['collection-result']['job_id'],\n        cpu_job_id=ctx['metadata']['cpu-acceptance']['job_id'],")
p.write_text(s)
p=p.with_name('launch_once.py');s=p.read_text()
s=s.replace("name+('.md' if name in ('freeze-review','cpu-report') else '.json')", "b.metadata_name(name)")
s=s.replace("and packet['namespace_claim_sha256']==ctx['pins']['namespace-claim']", "and packet['operational_amendment_sha256']==ctx['pins']['operational-amendment']\n               and packet['collection_job_id']==m['collection-result']['job_id']\n               and packet['namespace_claim_sha256']==ctx['pins']['namespace-claim']")
p.write_text(s)
