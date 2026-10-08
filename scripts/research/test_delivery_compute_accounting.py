"""Scoped cost/identity failures on fabricated metadata, no fits or outcomes."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import audit_delivery_compute_accounting as auditor


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,sort_keys=True,allow_nan=False)+'\n')
    return sha(p)


def fixture(root):
    files=[]
    def meta(name,value):
        h=put(root/name,value);files.append(dict(path=name,sha256=h));return h
    meta('F/protocol.json',{'seeds':list(range(12)),'delivery':{'inits':[0,1,2]}})
    worker=Path(auditor.__file__).parent/'runner_delivery_confirmation.py'
    assert sha(worker)==auditor.CONFIRMATION_WORKER_SHA
    meta('F/runtime.json',{'adapter_sha256':sha(worker)})
    for seed in range(12):
        meta(f'F/cases/{seed}/complete.json',dict(seed=seed,inits=[0,1,2],
            refit_timings=[dict(init=i,seconds=2) for i in range(3)]))
    meta('F/original_terminal.json',{'elapsed_seconds':4})
    meta('F/prior_execution.json',dict(status='time_limit',exit_code=-9,
        elapsed_seconds=10,prior_charged_reserved_seconds=7,aggregate_charged_seconds=17))
    eh=meta('F/execution.json',dict(status='complete',exit_code=0,elapsed_seconds=8,
        prior_charged_reserved_seconds=17,aggregate_charged_seconds=25))
    meta('F/terminal.json',dict(status='complete',execution_sha256=eh,aggregate_charged_reserved_seconds=25))
    manifest=put(root/'manifest.json',{'files':files})
    costs={k:{'scope':'fabricated prior measured subset'} for k in
        ('A_full','A_development_pilot','C_full','C_development_pilot')}
    costs.update(no_sum_across_incompatible_measurement_scopes=True,total_sprint_cpu_core_hours=None,
        B_partial={'actual_pilot_allocated_seconds':{'failed':126,'successful':119}},unknowns=['prior overhead'])
    cp=root/'prior-costs.json';ch=put(cp,costs)
    source='45ebeb89d2c76daa97a55f07728239245e0c4f60';registration='e9fb12aa807010388f2cb4701304f61fc7cd2092a459e9aab393b3aff49a65f5'
    fits=dict(source_revision=source,registration_sha256=registration,failed_attempts=[],
        charged_training_responses=32000,histories=80,matrix_fits=640,completed_fit_receipts=2,
        validated_completed_fits=[dict(index=i,cpu_seconds=2,wall_seconds=3) for i in (0,1)],completed_worlds=[])
    fp=root/'fit-metadata.json';fh=put(fp,fits)
    records=[dict(job_id_raw='33507418',job_id='33507418',state='COMPLETED',exit_code='0:0',
                  allocated_cpu_seconds_accrued=10,elapsed_seconds=10,allocated_cpus=1),
             dict(job_id_raw='33513116',job_id='33507421_15',state='RUNNING',exit_code='0:0',
                  allocated_cpu_seconds_accrued=5,elapsed_seconds=5,allocated_cpus=1)]
    scheduler=dict(at='fixture',source_revision=source,registration_sha256=registration,failed_scheduler_states=[],
        evaluation_scores_accessed=False,scheduler=records,current_chain_allocated_CPU_seconds_accrued=15,
        historical_failed_pilot_allocated_CPU_seconds=126,rounded_total_reserved_CPU_core_hours=86.86833333333334)
    sp=root/'scheduler-metadata.json';sh=put(sp,scheduler)
    mapping=dict(schema='delivery-slurm-identity-map-v1',source_revision=source,registration_sha256=registration,
        account='ucb736_asc1',captured_scheduler_receipt_sha256='a'*64,
        pairs=[{k:a[k] for k in ('job_id_raw','job_id')} for a in records],
        reserved_CPU_core_hours=86.86833333333334,reservation_basis='original full-registration total_requested_cpu_core_hours')
    mp=root/'identity-mapping.json';mh=put(mp,mapping)
    return dict(root=root,manifest_sha=manifest,accepted_costs=cp,accepted_costs_sha=ch,
        fit_metadata=fp,fit_metadata_sha=fh,scheduler_metadata=sp,scheduler_metadata_sha=sh,
        confirmation_worker=worker,scheduler_identity_mapping=mp,scheduler_identity_mapping_sha=mh)


class ComputeAccountingTests(unittest.TestCase):
    def test_separate_scopes_no_inferred_total_and_nonadditive_reservations(self):
        with tempfile.TemporaryDirectory() as directory:
            r=auditor.audit(**fixture(Path(directory).resolve()))
            self.assertIsNone(r['total_sprint_cpu_core_hours']);self.assertIsNone(r['F_confirmation']['process_cpu_seconds'])
            self.assertEqual(r['F_confirmation']['saved_refits'],36)
            self.assertEqual(r['F_confirmation']['refit_wall_seconds_sum'],72)
            self.assertEqual(r['F_confirmation']['final_nested_charged_reserved_wall_seconds'],25)
            self.assertEqual(r['B_current']['completed_fit_cpu_seconds_sum'],4)
            self.assertEqual(r['B_current']['completed_fit_wall_seconds_sum'],6)
            self.assertEqual(r['B_current']['original_chain_allocated_cpu_seconds_accrued'],15)
            self.assertEqual(r['B_current']['allocated_core_hours_including_pilots'],260/3600)
            self.assertFalse(r['B_scientific_outcomes_opened'])
            self.assertEqual(r['new_fits'],r['new_optimizer_updates'],r['new_responses'])

    def test_duplicate_or_unrelated_scheduler_and_bad_sum_reject(self):
        for variant in ('duplicate-raw','duplicate-job','unrelated','child','bad-sum','bool','bad-telemetry'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as directory:
                args=fixture(Path(directory).resolve());d=json.loads(args['scheduler_metadata'].read_text())
                if variant=='duplicate-raw': d['scheduler'].append(copy.deepcopy(d['scheduler'][0]))
                if variant=='duplicate-job': d['scheduler'][1]['job_id']=d['scheduler'][0]['job_id']
                if variant=='unrelated':d['scheduler'][1]['job_id']='99999999_15'
                if variant=='child':d['scheduler'][1]['job_id_raw']='33513116.batch'
                if variant=='bad-sum':d['current_chain_allocated_CPU_seconds_accrued']=16
                if variant=='bool':d['scheduler'][1]['elapsed_seconds']=True
                if variant=='bad-telemetry':d['scheduler'][1]['allocated_cpu_seconds_accrued']=6
                args['scheduler_metadata_sha']=put(args['scheduler_metadata'],d)
                with self.assertRaises(ValueError):auditor.audit(**args)

    def test_missing_or_changed_worker_and_nonfinite_json_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();args=fixture(root)
            changed=root/'worker.py';changed.write_bytes(b'# wrong timer')
            with self.assertRaisesRegex(ValueError,'digest differs'):
                auditor.audit(**dict(args,confirmation_worker=changed))
            with self.assertRaisesRegex(ValueError,'nonfinite'):
                auditor.decoded(b'{"seconds":NaN}')
            with self.assertRaisesRegex(ValueError,'duplicate'):
                auditor.decoded(b'{"seconds":1,"seconds":2}')

    def test_distinct_fit_indices_and_negative_measurements_reject(self):
        for variant in ('duplicate','negative','bool'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as directory:
                args=fixture(Path(directory).resolve());d=json.loads(args['fit_metadata'].read_text())
                if variant=='duplicate':d['validated_completed_fits'][1]['index']=0
                else:d['validated_completed_fits'][1]['cpu_seconds']=-1 if variant=='negative' else True
                args['fit_metadata_sha']=put(args['fit_metadata'],d)
                with self.assertRaises(ValueError):auditor.audit(**args)

    def test_foreign_raw_pair_and_array_aliases_or_overlap_reject(self):
        for variant in ('foreign-raw','padded','pending-overlap','range-overlap','parent-summary'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as directory:
                args=fixture(Path(directory).resolve());d=json.loads(args['scheduler_metadata'].read_text())
                mapping=json.loads(args['scheduler_identity_mapping'].read_text())
                if variant=='foreign-raw':d['scheduler'][1]['job_id_raw']='99999999'
                elif variant=='padded':
                    d['scheduler'].append(dict(d['scheduler'][1],job_id_raw='33513117',job_id='33507421_015'))
                    d['current_chain_allocated_CPU_seconds_accrued']=20
                else:
                    extra=dict(job_id_raw='33507420',job_id='33507420_[0-19%2]',state='PENDING',exit_code='0:0',
                        allocated_cpu_seconds_accrued=0,elapsed_seconds=0,allocated_cpus=0)
                    d['scheduler'].append(extra)
                    job={'pending-overlap':'33507420_5','range-overlap':'33507420_[4-8%2]',
                         'parent-summary':'33507420'}[variant]
                    d['scheduler'].append(dict(extra,job_id_raw='33513199',job_id=job))
                # Alias/coverage tests include even purportedly authorized pairs:
                # syntax and canonical membership still must reject them.
                if variant!='foreign-raw':
                    mapping['pairs']=[{k:a[k] for k in ('job_id_raw','job_id')} for a in d['scheduler']]
                    args['scheduler_identity_mapping_sha']=put(args['scheduler_identity_mapping'],mapping)
                args['scheduler_metadata_sha']=put(args['scheduler_metadata'],d)
                with self.assertRaises(ValueError):auditor.audit(**args)

    def test_valid_distinct_array_tasks_and_explicit_unknown_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            args=fixture(Path(directory).resolve());d=json.loads(args['scheduler_metadata'].read_text())
            d['scheduler'].append(dict(d['scheduler'][1],job_id_raw='33513117',job_id='33507421_16'))
            d['current_chain_allocated_CPU_seconds_accrued']=20
            d['rounded_total_reserved_CPU_core_hours']=None
            d['reserved_core_hours_unavailable_reason']='explicit fabricated missing reservation metadata'
            mapping=json.loads(args['scheduler_identity_mapping'].read_text())
            mapping['pairs']=[{k:a[k] for k in ('job_id_raw','job_id')} for a in d['scheduler']]
            args['scheduler_identity_mapping_sha']=put(args['scheduler_identity_mapping'],mapping)
            args['scheduler_metadata_sha']=put(args['scheduler_metadata'],d)
            r=auditor.audit(**args)
            self.assertEqual(r['B_current']['original_chain_allocated_cpu_seconds_accrued'],20)
            self.assertIsNone(r['B_current']['reserved_CPU_core_hours'])
            self.assertEqual(r['B_current']['reservation_unavailable_reason'],d['reserved_core_hours_unavailable_reason'])

    def test_invalid_reservations_reject(self):
        for v in (-100,'CPU measurement unknown',True,None,150.01):
            with self.subTest(v=v),tempfile.TemporaryDirectory() as directory:
                args=fixture(Path(directory).resolve());d=json.loads(args['scheduler_metadata'].read_text())
                d['rounded_total_reserved_CPU_core_hours']=v
                args['scheduler_metadata_sha']=put(args['scheduler_metadata'],d)
                with self.assertRaises(ValueError):auditor.audit(**args)


if __name__=='__main__':unittest.main()
