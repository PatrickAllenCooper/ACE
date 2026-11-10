"""Local saved-metadata joins and explicit sizing arithmetic; no producer imports."""
from pathlib import Path
import hashlib
import json
import sys

HERE=Path(__file__).resolve().parent
CAPSULE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-campaign-gpu-terminal-20261109-02')
PINS={
 'inventory_transport.json':'2ca9bf1fd7f73a6964d9a10e3e4b239329c4bd4b53b2723a0131ded9a0395d66',
 'inventory_after_transport.json':'53b4fc606be2fd900adba72f8729efb3b7f665217ca7450a2275f57cba8bf371',
 'remote-custody.json':'e59bca378f54b0225eb8c74b73bfeebdad830faf5d3fd62fedfc56593de8e27c',
 'disposition.json':'df246a23b4cadd6971e190d16deca72dcddd0a8e0c7e6a43c7ed8608dbaf9a62',
 'custody-inventory.json':'66ddd22d4a1f287df3f2beaaa962eef238e1c94e0690dd81b1db6c27424aebc1'}

helper=(HERE/'derive_inventory_timing.py').read_bytes()
assert hashlib.sha256(helper).hexdigest()=='9ce042cbd024f92266090221e8f6c20486bc0811aaa511e9c7fa56a50e2e7d72'
api={'__name__':'captured_metadata_only','__file__':str(HERE/'derive_inventory_timing.py')}
exec(compile(helper,api['__file__'],'exec'),api)
captured={n:api['capture'](CAPSULE/n,h) for n,h in PINS.items()}
objects={n:api['decode'](b) for n,b in captured.items()}
before=api['decode'](objects['inventory_transport.json']['stdout'])
after=api['decode'](objects['inventory_after_transport.json']['stdout'])
disposition=objects['disposition.json'];receipt=objects['remote-custody.json'];seal=objects['custody-inventory.json']
assert before['files']==after['files']==disposition['remote_files']==receipt['full_inventory']['files']
for v in (before,after,receipt['full_inventory']):
    assert type(v['directories']) is list and all(type(n) is str for n in v['directories'])
    assert len(v['directories'])==len(set(v['directories']))
assert sorted(before['directories'])==sorted(after['directories'])==sorted(receipt['full_inventory']['directories'])
assert seal['identical'] is True and len(seal['copies'])==2
for copy in seal['copies'].values():
    for name,h in PINS.items():
        if name=='custody-inventory.json':continue
        assert copy['files'][name]==dict(bytes=len(captured[name]),sha256=h)
assert disposition['errors']==disposition['input_pin_mismatches']==[]
assert disposition['remote_custody_receipt_sha256']==PINS['remote-custody.json']
assert disposition['job_id']=='35516125' and disposition['observed_related_jobs']==48 and disposition['observed_submission_journals']==98
assert disposition['closed_cpu_seconds']==37073 and disposition['closed_gpu_slice_seconds']==4458
assert disposition['preparation_allowance_cpu_seconds']==3600 and disposition['cpu_admission_after_release']==40673

paired_path=HERE/'attempt01/derived/derived.json'
paired_raw=paired_path.read_bytes();paired=json.loads(paired_raw);observations=paired['observations']
assert paired['input_pins']['before']['sha256']==PINS['inventory_transport.json']
assert paired['input_pins']['after']['sha256']==PINS['inventory_after_transport.json']
groups=observations['before']['groups']
slower={key:max(v['groups'][key]['intersecting_sample_seconds'] for v in observations.values()) for key in groups}
old_raw=(HERE/'closed-timing-evidence.json').read_bytes();old=json.loads(old_raw)
report=old['accepted_artificial_finalization'];point_report=report['report_through_terminal_utc_seconds']*4800/report['curve_points']
query_report=report['report_through_terminal_utc_seconds']*38400/report['point_query_predictions']
scan=sum(slower[n] for n in ('cpu_output_excluding_campaign','gpu_output','campaign'))
input_rereads=2*sum(slower[n] for n in ('cpu_inputs','gpu_inputs'))
prep_payload=2*scan+input_rereads
worker_io=sum(slower[n] for n in ('attempt_hashes','arm_files','sealed_input_rereads','arm_receipts','campaign'))

def scenario(io_multiplier, report_component):
    prep=prep_payload*io_multiplier+150+100+60
    startup=90
    work=report_component+worker_io*io_multiplier+100+50+100
    cleanup=30
    post=scan*io_multiplier+100+50+100
    return dict(io_multiplier=io_multiplier,report_component_seconds=report_component,
        preparation_seconds=prep,startup_seconds=startup,worker_seconds=work,cleanup_seconds=cleanup,parent_post_seconds=post,
        preparation_residual_seconds=1500-prep,startup_relative_residual_seconds=120-startup,
        worker_residual_at_latest_start_seconds=1700-work,parent_post_residual_seconds=840-post,
        joint_schedule=dict(auth_complete=prep,report_start=prep+startup,worker_end=prep+startup+work,
            cleanup_end=prep+startup+work+cleanup,parent_end=prep+startup+work+cleanup+post),
        joint_hard_residual_seconds=4190-(prep+startup+work+cleanup+post),
        latest_start_joint_parent_end=1620+work+cleanup+post,
        latest_start_joint_hard_residual_seconds=4190-(1620+work+cleanup+post))

out=Path(sys.argv[1]);out.mkdir(mode=0o700,exist_ok=False)
result=dict(schema='ace-finalizer-sizing03-author-disposition-v1',
    status='recommended_bounded_attempt_pending_independent_review_and_actual_gpu_acceptance',
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_pins={n:dict(path=str(CAPSULE/n),sha256=h,bytes=len(captured[n])) for n,h in PINS.items()},
    paired_derivation_sha256=hashlib.sha256(paired_raw).hexdigest(),closed_timing_evidence_sha256=hashlib.sha256(old_raw).hexdigest(),
    metadata_joins=dict(before_after_disposition_archive_file_maps_equal=True,directory_sets_equal=True,
        supplied_metadata_pins_in_both_sealed_copy_maps=True,errors_empty=True,scope='sizing metadata association only; no remote archive reread or actual GPU acceptance'),
    profile=dict(cpus=2,memory_mib=7680,gpus=0,wall_seconds=4200,preparation=1500,report_startup=120,report_startup_cap=1620,work=3320,cleanup=3350,hard=4190),
    entry_sha256='8a26ada2fc4778981237ca547fa437c620e6510080e93ab334e9762ebee2cf6f',source_commit='255b75fe37b28c302fb57821c0cc09a60a6cd3ea',
    source_review_sha256='39ef3af657fd64e8174d6c0b462fdb1519c1978ee3ea9d5288f11c998d4a5bc9',
    slower_group_seconds=slower,per_authentication_payload_proxy_seconds=scan,original_input_reread_proxy_seconds=input_rereads,
    preparation_payload_proxy_seconds=prep_payload,worker_existing_io_proxy_seconds=worker_io,
    allowances=dict(preparation_staging_seconds=150,preparation_flat_capture_metadata_seconds=100,preparation_directory_handoff_seconds=60,
        reporting_startup_seconds=90,worker_report_growth_fsync_seconds=100,worker_runtime_envelope_seconds=50,worker_sealing_dimension_variation_seconds=100,
        cleanup_seconds=30,parent_report_growth_seconds=100,parent_report_metadata_seconds=50,parent_directory_publication_variation_seconds=100),
    scenarios=dict(point_count=scenario(1,point_report),selected_slower_group_and_point_query=scenario(1,query_report),extra_25_percent_io_stress=scenario(1.25,query_report)),
    retained_failure=old['gpu01_failure'],
    collector_reported_accounting=dict(closed_cpu_seconds=37073,closed_gpu_seconds=4458,preparation_cpu_seconds=3600,
        finalizer_reservation_cpu_seconds=8400,projected_cpu_seconds=49073,remaining_cpu_seconds=8527,
        predecessor_jobs=48,predecessor_journals=98,final_jobs=49,final_journals=100,independent_actual_baseline_pending=True),
    new_probe_recommended=False,full_path_timing_supported=False,resource_admission_accepted=False,execution_qualified=False,scientific_qualified=False,
    remaining=['independent narrow timing disposition','actual GPU core acceptance/recovery supplement/baseline','explicit operational adoption','concrete finalizer packet review'],
    limits=['Coarse intersecting progress bins can overlap and include unrelated edge reads; they are observed proxies, not isolated finalizer benchmarks.',
        'Per-group maxima deliberately retain slower BEFORE GPU/campaign and slower AFTER CPU/input observations.',
        '150/100/60 and other explicit allowances are sizing judgments, not measured latency maxima.',
        'Existing836-second report sensitivity scales old validation too; adding current full I/O proxies is conservative and can double count.',
        '3-node/8-query arithmetic, future report bytes and filesystem latency remain uncertain; no universal upper bound or completion guarantee.',
        'Known240-second incomplete CPU scan and the stress scenario demonstrate possible failure under fixed limits; retain original failure/custody, no automatic retry.',
        'Actual GPU acceptance is pending; supplied seals establish the saved sizing evidence association only.'])
(out/'disposition.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ('slower_group_seconds','per_authentication_payload_proxy_seconds','original_input_reread_proxy_seconds','worker_existing_io_proxy_seconds','allowances','scenarios')},indent=2))
