"""Fabricated custody and mocked execution only; no numerical/model imports."""
import copy
import itertools
import os
from pathlib import Path
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import finalize_llm_intervention_campaign_cpu as gate
import qualify_llm_intervention_campaign_gpu as inventory_api


class FinalizerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self):
        paths = {k: str(self.root/k) for k in ('cpu_inputs', 'cpu_output', 'gpu_inputs',
                 'gpu_output', 'gpu_qualification', 'output')}
        paths['campaign'] = str(Path(paths['cpu_output'])/'worker/artifacts/campaign')
        for k, v in paths.items():
            if k != 'output':
                Path(v).mkdir(parents=True, exist_ok=True)
        croot, cpuout, gpuout = (Path(paths[k]) for k in ('campaign', 'cpu_output', 'gpu_output'))
        c = dict(scope='artificial', policy_ids=list(gate.POLICIES),
                 sources=[dict(plan=dict(world=dict(world_id=w), acquisition_cap_units=4)) for w in (9101, 9102)],
                 scientific_provenance_qualified=False, retry_allowed=False)
        bundle = dict(campaign=c, campaign_sha256=gate.sha(gate.raw(c).rstrip(b'\n')),
                      source_plan_pins=[dict(world_id=w, sha256='1'*64) for w in (9101, 9102)])
        gate.save(croot/'campaign.json', c)
        gate.save(croot/'source-plan-pins.json', bundle['source_plan_pins'])
        gate.save(croot/'start.json', {'fabricated': True})
        attempts = {}
        for i, pid in enumerate(gate.POLICIES):
            folder = croot/f'policy-{i:02d}'
            folder.mkdir()
            journal = gate.save(folder/'journal.jsonl', dict(fabricated=True, failed=False))
            attempt = dict(policy_id=pid, evidence_sha256={'journal.jsonl': journal})
            attempts[pid] = gate.save(folder/'attempt.json', attempt)
            gate.save(folder/'attempt.candidate.json', attempt)
            if i == 4:
                initial = inventory_api.inventory(croot)
        bp = gate.save(cpuout/'bundle.json', bundle)
        ct = dict(status='cpu_phase_complete', failed_policies=[], attempt_pins={p: attempts[p] for p in gate.POLICIES[:5]},
                  campaign_sha256=bundle['campaign_sha256'], numeric_training_updates=6,
                  llm_training_updates=0, gpu_services_started=0, private_scores_computed=False)
        cp = gate.save(cpuout/'phase-terminal.json', ct)
        ca = dict(schema='ace-paired-campaign-cpu-acceptance-v1', status='artificial_cpu_phase_accepted',
            cpu_packet_sha256='2'*64, job_id='17', allocation=dict(account='ucb736_asc2', partition='acpu',
            qos='cpu-normal', cpus=2, memory_mib=7680, gpus=0, state='COMPLETED', exit_code='0:0'),
            cpu_output=paths['cpu_output'], campaign_root=paths['campaign'],
            cpu_output_files=inventory_api.inventory(cpuout, exclude='worker/artifacts/campaign'),
            cpu_output_excluded_directory='worker/artifacts/campaign', campaign_files=initial,
            bundle_path='bundle.json', bundle_sha256=bp, phase_terminal_path='phase-terminal.json',
            phase_terminal_sha256=cp, attempt_pins=ct['attempt_pins'], scientific_qualified=False)
        capin = gate.sha(gate.raw(ca))
        gt = dict(schema=inventory_api.VERSION, status='artificial_gpu_campaign_complete', packet_sha256='3'*64,
            cpu_acceptance_sha256=capin, campaign_sha256=bundle['campaign_sha256'],
            attempt_pins={p: attempts[p] for p in gate.POLICIES[5:]}, new_training_updates=0,
            private_scores_computed=False, scientific_qualified=False)
        phase = dict(status='gpu_phase_complete', campaign_sha256=bundle['campaign_sha256'],
            attempt_pins=gt['attempt_pins'], signals_restored=True, failure=None, errors=[], unstarted_not_recorded=[])
        (gpuout/'controller/phase').mkdir(parents=True)
        terminal_pins = {}
        for role, path, value in (('supervisor', 'terminal.json', gt),
                                  ('phase', 'controller/phase/terminal.json', phase)):
            terminal_pins[role] = dict(root='gpu_output', path=path, sha256=gate.save(gpuout/path, value))
        for role in ('scheduler', 'runtime', 'cleanup'):
            terminal_pins[role] = dict(root='gpu_qualification', path=role+'.json',
                sha256=gate.save(Path(paths['gpu_qualification'])/(role+'.json'), dict(fabricated=True, accepted=True)))
        ga = dict(schema='ace-paired-campaign-gpu-acceptance-v1', status='artificial_gpu_phase_accepted', scope='artificial',
            gpu_packet_sha256='3'*64, cpu_packet_sha256='2'*64, cpu_acceptance_sha256=capin, job_id='18',
            allocation=dict(account='ucb736_asc2', partition='aa100', qos='gpu-testing', requested_cpus=4,
                allocated_cpus=4, memory_mib=15360, wall_seconds=600, gpus=1, gpu_type='a100_3g.20gb', state='COMPLETED', exit_code='0:0'),
            gpu_output=paths['gpu_output'], campaign_root=paths['campaign'], qualification_root=paths['gpu_qualification'],
            gpu_output_files=inventory_api.inventory(gpuout), campaign_files=inventory_api.inventory(croot),
            qualification_files=inventory_api.inventory(paths['gpu_qualification']), bundle_sha256=bp,
            attempt_pins=attempts, accepted_checks=dict(scheduler=True, runtime=True, cleanup=True, custody=True),
            terminal_pins=terminal_pins, scientific_qualified=False)
        packet = dict(schema=gate.VERSION, scope='artificial', resources=gate.RESOURCES, paths=paths,
            cpu_packet_sha256='2'*64, gpu_packet_sha256='3'*64, cpu_acceptance_sha256=capin,
            gpu_acceptance_sha256=gate.sha(gate.raw(ga)), files={}, source_names=[])
        data = {'cpu-acceptance.json': gate.raw(ca), 'gpu-acceptance.json': gate.raw(ga)}
        ctx = dict(packet=packet, old=dict(cpu_output=paths['cpu_output']), data=data,
            api=types.SimpleNamespace(inventory=inventory_api.inventory),
            cpu_acceptance=ca, gpu_acceptance=ga, bundle=bundle)
        return ctx

    def admit(self, ctx):
        return gate.admit_evidence(ctx['packet'], ctx['old'], {'schema': inventory_api.VERSION}, ctx['data'], inventory_api)

    def refresh_acceptance(self, ctx):
        ctx['data']['gpu-acceptance.json'] = gate.raw(ctx['gpu_acceptance'])

    def test_admits_complete_custody_without_runtime_imports(self):
        ctx = self.fixture()
        before = inventory_api.inventory(self.root)
        ca, ga, bundle = self.admit(ctx)
        self.assertEqual(bundle, ctx['bundle'])
        self.assertEqual(set(ga['attempt_pins']), set(gate.POLICIES))
        self.assertEqual(ca['attempt_pins'], ctx['cpu_acceptance']['attempt_pins'])
        self.assertEqual(before, inventory_api.inventory(self.root))
        self.assertFalse({'numpy', 'torch', 'transformers'} & set(sys.modules))

    def test_missing_or_unaccepted_authority_rejected_before_private_decode(self):
        ctx = self.fixture()
        original = gate.decode
        private = gate.raw(ctx['bundle'])
        for mutate in (
            lambda a: a.update(scope='scientific'),
            lambda a: a.update(status='gpu_phase_complete'),
            lambda a: a.update(scientific_qualified=True),
            lambda a: a['accepted_checks'].update(runtime=False),
            lambda a: a['accepted_checks'].update(cleanup=False),
            lambda a: a['accepted_checks'].update(custody=1),
            lambda a: a['allocation'].update(state='RUNNING'),
            lambda a: a['terminal_pins'].pop('scheduler'),
            lambda a: a['attempt_pins'].pop('dpo_llm'),
            lambda a: a.update(bundle_sha256='f'*64),
            lambda a: a.update(cpu_acceptance_sha256='f'*64),
        ):
            bad = copy.deepcopy(ctx)
            mutate(bad['gpu_acceptance'])
            self.refresh_acceptance(bad)
            def guarded(value):
                self.assertNotEqual(value, private, 'private bundle decoded before admission')
                return original(value)
            with self.subTest(mutation=mutate), patch.object(gate, 'decode', side_effect=guarded):
                with self.assertRaises(ValueError):
                    self.admit(bad)

    def test_cpu_acceptance_cannot_be_relabelled_or_omitted(self):
        ctx = self.fixture()
        for key, value in (('status', 'failed'), ('cpu_output_excluded_directory', 'worker'),
                           ('cpu_packet_sha256', 'f'*64), ('scientific_qualified', True)):
            bad = copy.deepcopy(ctx)
            bad['cpu_acceptance'][key] = value
            bad['data']['cpu-acceptance.json'] = gate.raw(bad['cpu_acceptance'])
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.admit(bad)

    def test_each_immutable_inventory_rejects_tampering(self):
        ctx = self.fixture()
        for key, name in (('campaign', 'policy-07/journal.jsonl'), ('cpu_output', 'phase-terminal.json'),
                          ('gpu_output', 'terminal.json'), ('gpu_qualification', 'runtime.json')):
            path = Path(ctx['packet']['paths'][key])/name
            original = path.read_bytes()
            path.write_bytes(original+b' ')
            try:
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'immutable inventory'):
                    self.admit(ctx)
            finally:
                path.write_bytes(original)

    def test_added_file_and_symlink_cannot_hide_in_custody(self):
        ctx = self.fixture()
        p = Path(ctx['packet']['paths']['campaign'])/'extra'
        p.write_text('unlisted')
        with self.assertRaisesRegex(ValueError, 'immutable inventory'):
            self.admit(ctx)
        p.unlink()
        p.symlink_to(Path(ctx['packet']['paths']['gpu_output'])/'terminal.json')
        with self.assertRaises(ValueError):
            self.admit(ctx)

    def test_all_eight_pins_and_publication_candidates_are_independent(self):
        ctx = self.fixture()
        for index, pid in enumerate(gate.POLICIES):
            bad = copy.deepcopy(ctx)
            bad['gpu_acceptance']['attempt_pins'][pid] = 'f'*64
            self.refresh_acceptance(bad)
            with self.subTest(policy=pid), self.assertRaises(ValueError):
                self.admit(bad)
        bad = copy.deepcopy(ctx)
        bad['gpu_acceptance']['campaign_files']['policy-07/attempt.candidate.json'] = 'e'*64
        self.refresh_acceptance(bad)
        with self.assertRaises(ValueError):
            self.admit(bad)

    def test_previous_finalization_intent_forbids_retry_even_if_repinned(self):
        ctx = self.fixture()
        root = Path(ctx['packet']['paths']['campaign'])
        gate.save(root/'finalization-intent.json', {'previous': 'failed'})
        ctx['gpu_acceptance']['campaign_files'] = inventory_api.inventory(root)
        self.refresh_acceptance(ctx)
        with self.assertRaises(ValueError):
            self.admit(ctx)

    def test_terminal_pins_cannot_escape_or_point_at_uninventoried_bytes(self):
        ctx = self.fixture()
        for value in ('../scheduler.json', '/tmp/scheduler.json', 'missing.json'):
            bad = copy.deepcopy(ctx)
            bad['gpu_acceptance']['terminal_pins']['scheduler']['path'] = value
            self.refresh_acceptance(bad)
            with self.subTest(path=value), self.assertRaises(ValueError):
                self.admit(bad)

    def test_successful_acceptance_does_not_override_failed_gpu_terminal(self):
        ctx = self.fixture()
        path = Path(ctx['packet']['paths']['gpu_output'])/'terminal.json'
        terminal = gate.decode(path.read_bytes())
        terminal['private_scores_computed'] = True
        path.write_bytes(gate.raw(terminal))
        a = ctx['gpu_acceptance']
        a['gpu_output_files'] = inventory_api.inventory(path.parent)
        a['terminal_pins']['supervisor']['sha256'] = gate.sha(path.read_bytes())
        self.refresh_acceptance(ctx)
        with self.assertRaisesRegex(ValueError, 'GPU terminal'):
            self.admit(ctx)

    def test_output_overlap_and_unpinned_destinations_are_rejected_without_writes(self):
        ctx = self.fixture()
        inp = self.root/'finalizer-inputs'
        inp.mkdir()
        old = dict(runtime_source=str(self.root/'runtime'), model_directory=str(self.root/'model'),
                   archive=dict(path=str(self.root/'archive')), manifest=dict(path=str(self.root/'manifest')))
        ctx['packet']['cpu_packet_sha256'] = gate.save(inp/'cpu-packet.json', old)
        packet = inp/'packet.json'
        pp = gate.save(packet, ctx['packet'])
        out = ctx['packet']['paths']['output']
        self.assertEqual(gate.packet_destination(packet, pp, out)[0], inp)
        baseline = inventory_api.inventory(self.root)
        for dest in (str(inp/'output'), ctx['packet']['paths']['campaign'], str(self.root/'wrong')):
            with self.subTest(dest=dest), self.assertRaises(ValueError):
                gate.packet_destination(packet, pp, dest)
        with self.assertRaises(ValueError):
            gate.packet_destination(packet, 'f'*64, out)
        bad = copy.deepcopy(ctx['packet'])
        bad['paths']['output'] = str(inp/'nested')
        wrong = inp/'wrong-packet.json'
        wp = gate.save(wrong, bad)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            gate.packet_destination(wrong, wp, bad['paths']['output'])
        wrong.unlink()
        self.assertEqual(baseline, inventory_api.inventory(self.root))

    def test_source_pin_failure_precedes_helper_execution(self):
        p = dict(files={gate.CPU_ENTRY: 'f'*64})
        with patch.object(gate, 'packet_destination', return_value=(self.root, p)), \
             patch.object(gate, 'module') as execute:
            with self.assertRaisesRegex(ValueError, 'helper/source'):
                gate.authenticate('packet', 'a'*64, 'out')
        execute.assert_not_called()

    def scheduler(self):
        return ('JobId=123 UserId=paco0228(7) JobState=RUNNING Account=ucb736_asc2 Partition=acpu QOS=cpu-normal '
                'NumCPUs=2 NumTasks=1 NumNodes=1 TimeLimit=00:05:00 Requeue=0 Restarts=0 '
                'AllocTRES=cpu=2,mem=7.50G,node=1,billing=2 ReqTRES=cpu=2,mem=7680M,node=1,billing=2')

    def test_exact_cpu_allocation_rejects_gpu_resources_requeues_and_wrong_limits(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(gate.allocation(self.scheduler(), '123')['NumCPUs'], '2')
            for old, new in (('00:05:00', '00:10:00'), ('cpu=2', 'cpu=4'), ('7.50G', '7.51G'),
                             ('Restarts=0', 'Restarts=1'), ('ucb736_asc2', 'other'),
                             ('billing=2', 'billing=2,gres/gpu=1'), ('ReqTRES=cpu=2', 'ReqTRES=cpu=2,cpu=2')):
                with self.subTest(new=new), self.assertRaises(ValueError):
                    gate.allocation(self.scheduler().replace(old, new), '123')
            with patch.dict(os.environ, {'SLURM_JOB_GPUS': '0'}), self.assertRaises(ValueError):
                gate.allocation(self.scheduler(), '123')

    def publish_report(self, ctx, failed=0):
        root = Path(ctx['packet']['paths']['campaign'])
        private = root/'private'
        private.mkdir()
        rows = []
        for i, (world, pid) in enumerate(itertools.product((9101, 9102), gate.POLICIES)):
            rows.append(dict(world_id=world, policy_id=pid, cell_status='reported', failed_or_unresolved=i < failed))
            folder = private/f'world-{i//8:02d}'/f'arm-{i%8:02d}'
            folder.mkdir(parents=True)
            gate.save(folder/'cost-report.json', dict(curve=[dict(cost=4, error=2)], curve_complete=i >= failed))
        summary = dict(scope='artificial', cell_count=16, rows=rows,
            policy_pairs=[dict(left=a, right=b, world_ids=[9101, 9102]) for a, b in itertools.combinations(gate.POLICIES, 2)],
            scientific_acceptance=False, scientific_protocol_frozen=False,
            scientific_provenance_qualified=False, complete_case_filtering_allowed=False)
        sp = gate.save(private/'summary.json', summary)
        receipt = dict(summary_sha256=sp, campaign_sha256=ctx['bundle']['campaign_sha256'],
            attempt_pins=ctx['gpu_acceptance']['attempt_pins'], cell_count=16, scientific_acceptance=False,
            new_environment_responses=0, new_training_updates=0, retry_allowed=False)
        gate.save(root/'finalization-intent.json', dict(fabricated=True))
        gate.save(root/'finalization.json', receipt)
        gate.save(root/'finalization.candidate.json', receipt)

    def test_report_retains_failed_cells_all_comparisons_and_original_journals(self):
        ctx = self.fixture()
        self.publish_report(ctx, failed=3)
        result = gate.validate_report(ctx)
        self.assertEqual((result['cell_count'], result['descriptive_policy_comparisons']), (16, 28))
        self.assertEqual(result['failed_or_unresolved_cells'], 3)
        self.assertFalse(result['scientific_acceptance'])
        self.assertTrue(set(ctx['gpu_acceptance']['campaign_files']) <= set(result['campaign_files']))

    def test_report_rejects_missing_cells_comparisons_claims_and_lost_curves(self):
        ctx = self.fixture()
        self.publish_report(ctx)
        root = Path(ctx['packet']['paths']['campaign'])
        summary_path = root/'private/summary.json'
        original = summary_path.read_bytes()
        # Repin the receipt too: envelope checks must be stronger than a hash.
        for mutate in (lambda s: s['rows'].pop(), lambda s: s['policy_pairs'].pop(),
                       lambda s: s.update(scientific_acceptance=True)):
            summary = gate.decode(original)
            mutate(summary)
            summary_path.write_bytes(gate.raw(summary))
            receipt = gate.decode((root/'finalization.json').read_bytes())
            receipt['summary_sha256'] = gate.sha(summary_path.read_bytes())
            for n in ('finalization.json', 'finalization.candidate.json'):
                (root/n).write_bytes(gate.raw(receipt))
            with self.assertRaisesRegex(ValueError, 'descriptive envelope'):
                gate.validate_report(ctx)
        summary_path.write_bytes(original)
        receipt['summary_sha256'] = gate.sha(original)
        for n in ('finalization.json', 'finalization.candidate.json'):
            (root/n).write_bytes(gate.raw(receipt))
        (root/'private/world-01/arm-07/cost-report.json').unlink()
        with self.assertRaises(FileNotFoundError):
            gate.validate_report(ctx)

    def test_report_never_accepts_changed_original_attempts(self):
        ctx = self.fixture()
        self.publish_report(ctx)
        (Path(ctx['packet']['paths']['campaign'])/'policy-00/journal.jsonl').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'original campaign evidence'):
            gate.validate_report(ctx)

    def process(self, rc=0):
        return Mock(pid=321, poll=Mock(return_value=rc))

    def clean(self, **kwargs):
        return dict(pid=321, pgid=321, group_still_exists=False, errors=[], returncode=0, kill_sent=False, **kwargs)

    def supervisor_setup(self):
        out = self.root/'supervisor'
        (out/'worker').mkdir(parents=True)
        gate.save(out/'worker/work-start.json', dict(packet_sha256='a'*64, pid=321, monotonic=105.))
        return out

    def test_supervisor_owns_one_group_and_closes_before_return(self):
        out = self.supervisor_setup()
        proc, kill = self.process(), Mock(return_value=self.clean())
        with patch.object(gate.subprocess, 'Popen', return_value=proc) as popen, \
             patch.object(gate.time, 'monotonic', return_value=110.), patch.object(gate.signal, 'setitimer'):
            result = gate.supervise(['captured', '-I', '-B'], {'CUDA_VISIBLE_DEVICES': ''}, out, 100., 'a'*64, kill)
        self.assertFalse(result['group_still_exists'])
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        kill.assert_called_once_with(proc)
        self.assertTrue((out/'worker.cleanup.json').exists())
        self.assertTrue((out/'worker.resources.json').exists())

    def test_startup_deadline_is_independent_of_work_deadline(self):
        out = self.supervisor_setup()
        (out/'worker/work-start.json').unlink()
        proc, kill = self.process(rc=None), Mock(return_value=self.clean())
        with patch.object(gate.subprocess, 'Popen', return_value=proc), \
             patch.object(gate.time, 'monotonic', side_effect=[110., 220., 221., 221., 221.]), \
             patch.object(gate.signal, 'setitimer'):
            with self.assertRaisesRegex(ValueError, 'startup/work deadline'):
                gate.supervise(['captured'], {}, out, 100., 'a'*64, kill)
        kill.assert_called_once_with(proc)

    def test_worker_exit_failure_still_cleans_and_never_relaunches(self):
        out = self.supervisor_setup()
        proc, kill = self.process(rc=1), Mock(return_value=self.clean())
        with patch.object(gate.subprocess, 'Popen', return_value=proc) as popen, \
             patch.object(gate.time, 'monotonic', return_value=110.), patch.object(gate.signal, 'setitimer'):
            with self.assertRaisesRegex(ValueError, 'worker failed'):
                gate.supervise(['captured'], {}, out, 100., 'a'*64, kill)
        self.assertEqual(popen.call_count, 1)
        kill.assert_called_once_with(proc)

    def test_unconfirmed_cleanup_is_held_and_cannot_be_relabelled_clean(self):
        out = self.supervisor_setup()
        first = self.clean()
        first.update(group_still_exists=True, kill_sent=True)
        proc, kill = self.process(), Mock(side_effect=[first, self.clean()])
        with patch.object(gate.subprocess, 'Popen', return_value=proc), \
             patch.object(gate.time, 'monotonic', return_value=110.), patch.object(gate.signal, 'setitimer'):
            with self.assertRaisesRegex(ValueError, 'natural worker closure'):
                gate.supervise(['captured'], {}, out, 100., 'a'*64, kill)
        self.assertEqual(kill.call_count, 2)
        self.assertTrue((out/'cleanup-hold-000.json').exists())

    def test_failed_authentication_records_terminal_attempt_and_forbids_blind_retry(self):
        ctx = self.fixture()
        out = Path(ctx['packet']['paths']['output'])
        argv = ['finalizer', '--packet', str(self.root/'packet.json'), '--packet-sha256', 'a'*64, '--output', str(out)]
        with patch.object(sys, 'argv', argv), patch.object(gate.signal, 'setitimer'), \
             patch.object(gate, 'packet_destination', return_value=(self.root, ctx['packet'])), \
             patch.object(gate, 'authenticate', side_effect=ValueError('unaccepted GPU')) as auth, \
             patch.object(gate, 'parent') as run:
            with self.assertRaisesRegex(ValueError, 'unaccepted GPU'):
                gate.main()
            before = inventory_api.inventory(out)
            with self.assertRaises(FileExistsError):
                gate.main()
            self.assertEqual(before, inventory_api.inventory(out))
        self.assertEqual(auth.call_count, 1)
        run.assert_not_called()
        self.assertEqual(gate.decode((out/'failure.json').read_bytes())['status'], 'terminal_failure')
        self.assertFalse(gate.decode((out/'attempt.json').read_bytes())['retry_allowed'])

    def test_worker_calls_original_engine_exactly_once_and_retains_exception(self):
        ctx = self.fixture()
        out = Path(ctx['packet']['paths']['output'])
        (out/'worker').mkdir(parents=True)
        stage = {'staged_prefix': '/fabricated/runtime'}
        stage_pin = gate.save(out/'stage.json', stage)
        handoff = dict(stage_sha256=stage_pin, parent_pid=os.getppid(), job_id='1', started_monotonic=100.)
        args = types.SimpleNamespace(output=str(out), packet_sha256='a'*64)
        ctx.update(root=self.root, cpu=types.SimpleNamespace(CapturedSources=Mock(return_value=object())))
        campaign = types.ModuleType('llm_intervention_campaign')
        campaign.finalize_campaign = Mock(side_effect=RuntimeError('preserved partial finalization'))
        with patch.dict(sys.modules, {'llm_intervention_campaign': campaign}), \
             patch.dict(os.environ, {'SLURM_JOB_ID': '1'}), patch.object(gate, 'runtime_record', return_value={}), \
             patch.object(gate.time, 'monotonic', return_value=110.), patch.object(gate.signal, 'setitimer'), \
             patch.object(gate, 'validate_report') as report:
            with self.assertRaisesRegex(RuntimeError, 'preserved partial'):
                gate.worker(args, ctx, handoff)
        campaign.finalize_campaign.assert_called_once_with(campaign=ctx['bundle']['campaign'],
            campaign_sha256=ctx['bundle']['campaign_sha256'], source_plan_pins=ctx['bundle']['source_plan_pins'],
            output=ctx['packet']['paths']['campaign'], attempt_pins=ctx['gpu_acceptance']['attempt_pins'])
        report.assert_not_called()
        self.assertTrue((out/'worker/work-start.json').exists())


if __name__ == '__main__':
    unittest.main()
