"""New fabricated 24-world custody/reporting boundaries, no scientific execution."""
import copy
import itertools
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import finalize_llm_intervention_scientific_campaign_cpu as gate
import qualify_llm_intervention_campaign_gpu as inventory_api

class ScientificFinalizerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_revised_report_cap_scheduler_and_absolute_offsets(self):
        self.assertEqual(gate.RESOURCES['wall_seconds'], 1200)
        self.assertEqual((gate.STARTUP, gate.WORK, gate.CLEANUP, gate.HARD), (120,1140,1170,1190))
        text = ('JobId=17 JobState=RUNNING Account=ucb736_asc2 Partition=acpu QOS=cpu-normal '
                'NumCPUs=2 NumTasks=1 NumNodes=1 TimeLimit=00:20:00 Requeue=0 Restarts=0 '
                'UserId=paco0228(12) AllocTRES=cpu=2,node=1,mem=7680M ReqTRES=cpu=2,node=1,mem=7680M')
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(gate.allocation(text, '17')['TimeLimit'], '00:20:00')
            for invalid in ('00:05:00','00:19:59','00:20:01','00:30:00'):
                with self.subTest(limit=invalid), self.assertRaises(ValueError):
                    gate.allocation(text.replace('00:20:00',invalid),'17')
        self.assertFalse({'numpy','torch','transformers'} & set(sys.modules))

    def fixture(self):
        paths = {k: str(self.root / k) for k in ('cpu_inputs', 'cpu_output', 'gpu_inputs', 'gpu_output', 'output')}
        paths['campaign'] = str(Path(paths['cpu_output']) / 'worker/artifacts/campaign')
        for (k, v) in paths.items():
            if k != 'output':
                Path(v).mkdir(parents=True, exist_ok=True)
        (croot, cpuout, gpuout) = (Path(paths[k]) for k in ('campaign', 'cpu_output', 'gpu_output'))
        c = dict(scope='development', policy_ids=list(gate.POLICIES), sources=[dict(plan=dict(world=dict(world_id=w), acquisition_cap_units=24)) for w in gate.WORLD_IDS], scientific_provenance_qualified=False, retry_allowed=False)
        bundle = dict(campaign=c, campaign_sha256=gate.sha(gate.raw(c).rstrip(b'\n')), source_plan_pins=[dict(world_id=w, sha256='1' * 64) for w in gate.WORLD_IDS])
        gate.save(croot / 'campaign.json', c)
        gate.save(croot / 'source-plan-pins.json', bundle['source_plan_pins'])
        gate.save(croot / 'start.json', {'fabricated': True})
        attempts = {}
        for (i, pid) in enumerate(gate.POLICIES):
            folder = croot / f'policy-{i:02d}'
            folder.mkdir()
            journal = gate.save(folder / 'journal.jsonl', dict(fabricated=True, failed=False))
            attempt = dict(policy_id=pid, evidence_sha256={'journal.jsonl': journal})
            attempts[pid] = gate.save(folder / 'attempt.json', attempt)
            gate.save(folder / 'attempt.candidate.json', attempt)
            if i == 4:
                initial = inventory_api.inventory(croot)
        bp = gate.save(cpuout / 'bundle.json', bundle)
        ct = dict(status='cpu_phase_complete', completed_cpu_cells=120, declared_cells=192, attempt_pins={p: attempts[p] for p in gate.POLICIES[:5]}, campaign_sha256=bundle['campaign_sha256'], numeric_training_updates=161, llm_training_updates=0, gpu_services_started=0, private_scores_computed=False)
        cp = gate.save(cpuout / 'phase-terminal.json', ct)
        ca = dict(schema='ace-scientific-campaign-cpu-acceptance-v1', status='development_cpu_phase_accepted', cpu_packet_sha256='2' * 64, job_id='17', allocation=dict(account='ucb736_asc2', partition='acpu', qos='cpu-normal', cpus=2, memory_mib=7680, gpus=0, state='COMPLETED', exit_code='0:0'), cpu_output=paths['cpu_output'], campaign_root=paths['campaign'], cpu_output_files=inventory_api.inventory(cpuout, exclude='worker/artifacts/campaign'), cpu_output_excluded_directory='worker/artifacts/campaign', campaign_files=initial, bundle_path='bundle.json', bundle_sha256=bp, phase_terminal_path='phase-terminal.json', phase_terminal_sha256=cp, attempt_pins=ct['attempt_pins'], scientific_qualified=False)
        capin = gate.sha(gate.raw(ca))
        gt = dict(schema=inventory_api.VERSION, status='development_gpu_campaign_complete', packet_sha256='3' * 64, cpu_acceptance_sha256=capin, campaign_sha256=bundle['campaign_sha256'], attempt_pins={p: attempts[p] for p in gate.POLICIES[5:]}, new_training_updates=0, private_scores_computed=False, scientific_qualified=False)
        phase = dict(status='gpu_phase_complete', campaign_sha256=bundle['campaign_sha256'], attempt_pins=gt['attempt_pins'], signals_restored=True, failure=None, errors=[], unstarted_not_recorded=[])
        (gpuout / 'controller/phase').mkdir(parents=True)
        terminal_pins = {}
        for (role, path, value) in (('supervisor', 'terminal.json', gt), ('phase', 'controller/phase/terminal.json', phase)):
            terminal_pins[path] = gate.save(gpuout / path, value)
        ga = dict(schema='ace-scientific-campaign-gpu-acceptance-v1', status='development_gpu_phase_accepted', scope='development', gpu_packet_sha256='3' * 64, cpu_packet_sha256='2' * 64, cpu_acceptance_sha256=capin, job_id='18', allocation=dict(account='ucb736_asc2', partition='aa100', qos='gpu-testing', requested_cpus=4, allocated_cpus=4, memory_mib=15360, wall_seconds=2700, gpus=1, gpu_type='a100_3g.20gb', state='COMPLETED', exit_code='0:0'), gpu_output=paths['gpu_output'], campaign_root=paths['campaign'], gpu_output_files=inventory_api.inventory(gpuout), campaign_files=inventory_api.inventory(croot), bundle_sha256=bp, attempt_pins=attempts, accepted_checks=dict(scheduler=True, runtime=True, cleanup=True, custody=True), terminal_pins=terminal_pins, scientific_qualified=False)
        packet = dict(schema=gate.VERSION, scope='development', resources=gate.RESOURCES, paths=paths, cpu_packet_sha256='2' * 64, gpu_packet_sha256='3' * 64, cpu_acceptance_sha256=capin, gpu_acceptance_sha256=gate.sha(gate.raw(ga)), files={}, source_names=[])
        data = {'numerical-plan.json': gate.raw(dict(exposure=dict(counts=dict(sft_steps=160, dpo_steps=1)))), 'cpu-acceptance.json': gate.raw(ca), 'gpu-acceptance.json': gate.raw(ga)}
        ctx = dict(packet=packet, old=dict(cpu_output=paths['cpu_output']), data=data, api=types.SimpleNamespace(inventory=inventory_api.inventory), cpu_acceptance=ca, gpu_acceptance=ga, bundle=bundle)
        return ctx

    def admit(self,ctx):
        return gate.admit_evidence(ctx['packet'],ctx['old'],dict(schema=inventory_api.VERSION,
            resources=dict(wall_seconds=2700)),ctx['data'],inventory_api)

    def test_full_24_world_all_eight_attempts(self):
        ctx=self.fixture(); ca,ga,bundle=self.admit(ctx)
        self.assertEqual([s['plan']['world']['world_id'] for s in bundle['campaign']['sources']],list(gate.WORLD_IDS))
        self.assertEqual(len(ga['attempt_pins']),8)
        self.assertEqual(len(ca['attempt_pins']),5)
        self.assertFalse({'numpy','torch','transformers'} & set(sys.modules))

    def test_missing_acceptance_rejected_before_private_bundle_decode(self):
        ctx=self.fixture();ctx['cpu_acceptance']['status']='pending'
        ctx['data']['cpu-acceptance.json']=gate.raw(ctx['cpu_acceptance'])
        original=gate.decode
        def decode(raw):
            value=original(raw)
            if isinstance(value,dict) and 'campaign' in value:raise AssertionError('private bundle read too soon')
            return value
        with patch.object(gate,'decode',side_effect=decode),self.assertRaisesRegex(ValueError,'CPU acceptance'):
            self.admit(ctx)

    def test_changed_cpu_bytes_rejected_without_report(self):
        ctx=self.fixture();p=Path(ctx['packet']['paths']['cpu_output'])/'phase-terminal.json'
        p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'inventory differs'):self.admit(ctx)
        self.assertFalse((Path(ctx['packet']['paths']['campaign'])/'private').exists())

    def test_exact_matched_training_exposure(self):
        for s,d in ((159,1),(160,0),(160,129),(160,True)):
            with self.subTest(s=s,d=d),self.assertRaises(ValueError):
                gate.expected_numeric_steps({'numerical-plan.json':gate.raw(dict(exposure=dict(counts=dict(sft_steps=s,dpo_steps=d))))})
        self.assertEqual(gate.expected_numeric_steps({'numerical-plan.json':gate.raw(dict(exposure=dict(counts=dict(sft_steps=160,dpo_steps=128))))}),288)

    def report_fixture(self):
        ctx=self.fixture();root=Path(ctx['packet']['paths']['campaign']);(root/'private').mkdir()
        rows=[dict(world_id=w,policy_id=p,cell_status='failed',failed_or_unresolved=True)
              for w,p in itertools.product(gate.WORLD_IDS,gate.POLICIES)]
        summary=dict(scope='development',cell_count=192,rows=rows,
            policy_pairs=[dict(left=l,right=r,world_ids=list(gate.WORLD_IDS)) for l,r in itertools.combinations(gate.POLICIES,2)],
            scientific_acceptance=False,scientific_protocol_frozen=False,scientific_provenance_qualified=False,complete_case_filtering_allowed=False)
        pin=gate.save(root/'private/summary.json',summary)
        receipt=dict(summary_sha256=pin,campaign_sha256=ctx['bundle']['campaign_sha256'],attempt_pins=ctx['gpu_acceptance']['attempt_pins'],
            cell_count=192,scientific_acceptance=False,new_environment_responses=0,new_training_updates=0,retry_allowed=False)
        gate.save(root/'finalization.json',receipt);gate.save(root/'finalization.candidate.json',receipt)
        return ctx,summary

    def test_complete_192_cells_retains_every_failure(self):
        ctx,summary=self.report_fixture();r=gate.validate_report(ctx)
        self.assertEqual((r['cell_count'],r['descriptive_policy_comparisons'],r['failed_or_unresolved_cells']),(192,28,192))
        root=Path(ctx['packet']['paths']['campaign']);summary['rows'].pop()
        (root/'private/summary.json').write_bytes(gate.raw(summary))
        receipt=gate.decode((root/'finalization.json').read_bytes());receipt['summary_sha256']=gate.sha(gate.raw(summary))
        for n in ('finalization.json','finalization.candidate.json'):(root/n).write_bytes(gate.raw(receipt))
        with self.assertRaisesRegex(ValueError,'192-cell'):gate.validate_report(ctx)

    def test_cleanup_throw_preserves_first_failure_and_handle(self):
        proc=Mock(pid=8123);proc.poll.return_value=7
        owned=gate.OwnedChild(10)
        with patch.object(gate.time,'monotonic',return_value=11),patch.object(gate.subprocess,'Popen',return_value=proc), \
             patch.object(gate.signal,'signal'),patch.object(gate.signal,'setitimer'), \
             self.assertRaisesRegex(ValueError,'failed or never began'):
            gate.supervise(['fake'],{},self.root,10,'a'*64,Mock(side_effect=RuntimeError('cleanup failed')),owned)
        self.assertIs(owned.process,proc)
        r=json.loads((self.root/'worker.resources.json').read_text())
        self.assertIn('failed or never began',r['first_failure']['message'])
        self.assertEqual(r['cleanup_failure']['message'],'cleanup failed')

    def test_owned_hold_uses_original_absolute_deadline_then_retires(self):
        proc=Mock(pid=123);proc.poll.side_effect=[None,0,0]
        owned=gate.OwnedChild(10);owned.process=proc
        with patch.object(gate.os,'killpg',side_effect=[None,None,ProcessLookupError()]), \
             patch.object(gate.time,'monotonic',return_value=20),patch.object(gate.time,'sleep'), \
             patch.object(gate.signal,'signal'),patch.object(gate.signal,'setitimer') as timer:
            owned.hold_until_closed()
        timer.assert_called_once_with(signal.ITIMER_REAL,280)
        self.assertIsNone(owned.process)
        with patch.object(gate.os,'killpg') as kill:
            self.assertTrue(owned.confirm_closed())
        kill.assert_not_called()

    def test_full_24_report_read_does_not_modify_upstream(self):
        ctx,_=self.report_fixture();root=Path(ctx['packet']['paths']['campaign'])
        before=inventory_api.inventory(root);gate.validate_report(ctx)
        self.assertEqual(before,inventory_api.inventory(root))

    def test_scientific_inventory_exceeds_old_fixture_cap_and_rejects_links(self):
        data = self.root/'data'
        data.mkdir()
        for i in range(8193):
            (data/f'{i:05d}.txt').write_bytes(b'x')
        excluded = self.root/'excluded'
        excluded.mkdir()
        (excluded/'private.txt').write_bytes(b'kept upstream')
        result = gate.inventory(self.root, exclude='excluded')
        self.assertEqual(len(result),8193)
        self.assertEqual(set(result.values()),{gate.sha(b'x')})
        (self.root/'link').symlink_to(data/'00000.txt')
        with self.assertRaisesRegex(ValueError,'symlink'):
            gate.inventory(self.root, exclude='excluded')

if __name__=='__main__':unittest.main()
