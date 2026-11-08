"""Only R1/R2 artificial metadata regressions; no actual artifact inputs."""
from pathlib import Path
import copy
import importlib.util
import sys
import unittest

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('review_delta_aid',ROOT/'audit_saved.py')
aid=importlib.util.module_from_spec(spec)
spec.loader.exec_module(aid)


class ReviewDelta(unittest.TestCase):
    def test_r1_source_dimensions_reject_self_consistent_wrong_width(self):
        exposure={'records':[{'record_id':rid,'action_ids':([None] if i%5==4 else [f'a{k}' for k in range(8)]+[None])}
                            for i,rid in enumerate(aid.ROSTER)]}
        def toy(width):
            return dict(feature_width=width,parameter_count=width*128+128+128*128+128+128+1,
                encoding='complete-padded-numeric-state-signed-log-v1',retry_allowed=False,
                feature_pins={r['record_id']:{'shape':[len(r['action_ids']),width],'float32_tensor_sha256':'1'*64}
                              for r in exposure['records']})
        self.assertEqual(aid.numerical_dimensions(),(8891,1154817))
        aid.audit_numerical_shape(toy(8891),exposure)
        for width in (1,8890,8892,20000):
            with self.subTest(self_consistent_wrong_width=width):
                with self.assertRaisesRegex(ValueError,'source-derived numerical width/parameter count'):
                    aid.audit_numerical_shape(toy(width),exposure)
        changed=toy(8891);changed['parameter_count']+=1
        with self.assertRaisesRegex(ValueError,'source-derived numerical width/parameter count'):
            aid.audit_numerical_shape(changed,exposure)
        changed=toy(8891);changed['feature_pins'][aid.ROSTER[0]]['shape'][1]=8890
        with self.assertRaisesRegex(ValueError,'numerical feature metadata'):
            aid.audit_numerical_shape(changed,exposure)
        changed=toy(8891);changed['feature_pins'][aid.ROSTER[-1]]['shape'][0]=9
        with self.assertRaisesRegex(ValueError,'numerical feature metadata'):
            aid.audit_numerical_shape(changed,exposure)

    def test_r2_worker_relative_phases_never_claim_parent_work_timing(self):
        facts=dict(plan_sha256='1'*64,numerical_plan_sha256='2'*64,exposure_sha256='3'*64,
            prepared_sha256='4'*64,numerical_plan_file_sha256='5'*64,records=160,retained_pairs=3584,sft_steps=160,dpo_steps=128)
        stage_raw=b'{}\n'
        worker=dict(schema='ace-scientific-training-preparation-cpu-v1',status='cpu_preparation_complete',packet_sha256=aid.INNER,
            runtime_stage_sha256=aid.sha(stage_raw),training_plan_sha256=facts['plan_sha256'],
            numerical_plan_sha256=facts['numerical_plan_sha256'],exposure_sha256=facts['exposure_sha256'],
            llm_prepared_file_sha256=facts['prepared_sha256'],numerical_plan_file_sha256=facts['numerical_plan_file_sha256'],
            records=160,retained_pairs=3584,sft_steps=160,dpo_steps=128,**aid.FLAGS)
        worker_raw=(aid.canonical(worker)+'\n').encode()
        cleanup=dict(group_still_exists=False,errors=[],returncode=0,kill_sent=False)
        parent=dict(schema=worker['schema'],status=worker['status'],packet_sha256=aid.INNER,
            worker_terminal_sha256=aid.sha(worker_raw),stage_sha256=aid.sha(stage_raw),cleanup=cleanup,elapsed_seconds=580.,**aid.FLAGS)
        resources=dict(cleanup=cleanup,errors=[],startup_observed=True,returncode=0,elapsed_including_cleanup_seconds=578.)
        started=dict(started_monotonic=1000.,pid=123,pgid=123,startup_deadline=1300.,deadline_monotonic=1550.,supervisor_deadline_monotonic=1590.)
        marker=dict(packet_sha256=aid.INNER,pid=123,monotonic=1250.)
        phases=[dict(phase=n,elapsed_seconds=t) for n,t in zip(
            ('service_cpu_import_start','service_cpu_import_complete','cpu_tokenizer_load_start','cpu_tokenization_complete'),
            (0.,10.,11.,370.))]
        def check(rows=phases,par=parent,st=started,mk=marker):
            return aid.audit_lifecycle(par,worker,cleanup,resources,st,mk,rows,worker_raw=worker_raw,stage_raw=stage_raw,facts=facts)
        # If journal origin were parent+200,370 would correspond to parent+570.
        # No origin is supplied/invented by the aid: these phases cannot prove550.
        result=check()
        self.assertEqual(result['phase_time_origin'],'worker_journal_started')
        self.assertTrue(result['phase_order_and_durations_consistent'])
        self.assertFalse(result['original_origin_work_deadline_verified'])
        self.assertFalse(result['exact_cleanup_duration_verified'])
        self.assertTrue(result['work_timing_requires_separate_authenticated_source_command_trace_audit'])
        self.assertFalse(result['acceptance_authorized'])
        later=copy.deepcopy(phases);later[-1]['elapsed_seconds']=560.
        self.assertFalse(check(later)['original_origin_work_deadline_verified'])
        for value in (-1.,float('nan'),float('inf'),True):
            with self.subTest(invalid_worker_duration=repr(value)):
                bad=copy.deepcopy(phases);bad[0]['elapsed_seconds']=value
                with self.assertRaises(ValueError):check(bad)
        bad=copy.deepcopy(phases);bad[-1]['elapsed_seconds']=5.
        with self.assertRaises(ValueError):check(bad)
        bad=copy.deepcopy(phases);bad.append(dict(phase='optimizer_step',elapsed_seconds=371.))
        with self.assertRaises(ValueError):check(bad)
        bad=dict(parent,elapsed_seconds=590.)
        with self.assertRaisesRegex(ValueError,'absolute completion bound'):check(par=bad)
        bad=dict(started,deadline_monotonic=1551.)
        with self.assertRaisesRegex(ValueError,'deadline origin reset'):check(st=bad)
        bad=dict(marker,monotonic=1300.)
        with self.assertRaisesRegex(ValueError,'startup marker identity'):check(mk=bad)
        self.assertFalse(any(n in sys.modules for n in ('torch','numpy','transformers','tokenizers')))


if __name__=='__main__':
    unittest.main(verbosity=2)
