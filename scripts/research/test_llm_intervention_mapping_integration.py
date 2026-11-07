"""Artificial observer and saved-receipt integration. No device or model I/O."""
import copy
import json
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

import llm_intervention_gpu_mappings as mappings
import llm_intervention_service_runtime as runtime
import qualify_llm_intervention_model_service as gpu
import qualify_llm_intervention_runtime_metadata as diagnostic
import qualify_llm_intervention_service_cpu as cpu
import llm_intervention_model_service as service
import test_llm_intervention_gpu_mappings as map_fixtures
import test_llm_intervention_service_cpu as cpu_fixtures


class MappingIntegrationTests(unittest.TestCase):
    def observer(self, enabled=True):
        snapshot, inventory = map_fixtures.fixture()
        obj = object.__new__(runtime.ServiceRuntime)
        obj.prefix = Path('/copied')
        obj.loader = obj.captured = obj.entry_file = obj.entry_globals = None
        obj.provenance = obj.temp_root = obj.temp_stat = obj.baseline = None
        obj.allow_gpu_mappings = enabled
        obj.gpu_mappings = mappings
        obj.contract = service.RUNTIME
        records, order = {}, []
        def save(name, value):
            records[name] = copy.deepcopy(value)
            order.append(name)
        def capture(snapshot):
            self.assertIn('runtime-after-model-load.json', records)
            order.append('capture')
            return inventory
        obj.journal = types.SimpleNamespace(save=save)
        obj.diagnostic = types.SimpleNamespace(RUNTIME_SOURCE='/original',
            runtime_snapshot=Mock(return_value=snapshot), check_prefix=Mock())
        return obj, snapshot, inventory, records, order, capture

    def test_live_observer_preserves_raw_before_capture_and_checks_separate_view(self):
        obj, raw, inventory, records, order, capture = self.observer()
        before = copy.deepcopy(raw)
        with patch.object(runtime, 'runtime_modules', return_value=({}, {'fixture': {}})), \
             patch.object(mappings, 'capture', side_effect=capture):
            result = obj.observe('after-model-load', require_imports=False)
        self.assertEqual(raw, before | {'additional_package_versions': dict.fromkeys(runtime.EXTRA_PACKAGES)})
        self.assertEqual(result['mixed_native_paths'], sorted((mappings.SPECIAL_PATHS | {"/dev/nvidia0"})))
        self.assertEqual(order[:2], ['runtime-after-model-load.json', 'capture'])
        checked = obj.diagnostic.check_prefix.call_args.args[1]
        self.assertEqual(checked['mixed_native_paths'], [])
        self.assertEqual(records['runtime-after-model-load.json']['process_maps'], before['process_maps'])
        saved = records['gpu-mapping-validation-after-model-load.json']
        reconstructed = runtime.checked_gpu_snapshot(records['runtime-after-model-load.json'], inventory,
            saved, 'after-model-load', obj.prefix, obj.diagnostic)
        self.assertEqual(reconstructed['mixed_native_paths'], [])
        bad = copy.deepcopy(saved); bad['phase'] = 'after-requests'
        with self.assertRaises(ValueError):
            runtime.checked_gpu_snapshot(raw, inventory, bad, 'after-model-load', obj.prefix, obj.diagnostic)

    def test_cpu_path_never_captures_devices_and_cannot_enable_late_exemption(self):
        obj, raw, _, records, _, _ = self.observer(False)
        with patch.object(runtime, 'runtime_modules', return_value=({}, {})), \
             patch.object(mappings, 'capture') as capture:
            obj.observe('after-service-imports', require_imports=False)
            self.assertIs(obj.diagnostic.check_prefix.call_args.args[1], raw)
            with self.assertRaises(ValueError): obj.observe('after-model-load', require_imports=False)
            capture.assert_not_called()
        self.assertIn('runtime-after-model-load.json', records)

    def test_mapping_failure_retains_raw_inventory_without_accepted_proof(self):
        obj, raw, inventory, records, _, capture = self.observer()
        raw['process_maps'] += '5000-6000 r-xp 0 00:06 900 /tmp/foreign.so\n'
        with patch.object(runtime, 'runtime_modules', return_value=({}, {})), \
             patch.object(mappings, 'capture', side_effect=capture):
            with self.assertRaises(ValueError): obj.observe('after-model-load', require_imports=False)
        self.assertIn('/tmp/foreign.so', records['runtime-after-model-load.json']['process_maps'])
        self.assertEqual(records['gpu-mapping-inventory-after-model-load.json'], inventory)
        self.assertNotIn('gpu-mapping-validation-after-model-load.json', records)
        obj.diagnostic.check_prefix.assert_not_called()

    def test_gpu_receipts_reconstruct_both_phases_and_keep_cpu_guards(self):
        fixture = cpu_fixtures.CpuPreflightTests()
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        records = fixture.receipts()
        records['runtime-stage-binding.json']['gpu_mapping_enabled'] = True
        sample, inventory = map_fixtures.fixture()
        for phase in mappings.PHASES:
            raw = copy.deepcopy(records['runtime-after-cpu-tokenization.json'])
            raw.update(process_maps=sample['process_maps'], mixed_native_paths=sample['mixed_native_paths'])
            _, proof = mappings.checked_view(raw, inventory, phase,
                prefix=fixture.stage['staged_prefix'], original_runtime=diagnostic.RUNTIME_SOURCE)
            records['runtime-'+phase+'.json'] = raw
            records['gpu-mapping-inventory-'+phase+'.json'] = inventory
            records['gpu-mapping-validation-'+phase+'.json'] = proof
        out = fixture.root/'session'; worker = out/'worker'; worker.mkdir(parents=True)
        def validate(values):
            for path in worker.iterdir(): path.unlink()
            for name, value in values.items(): (worker/name).write_text(json.dumps(value))
            return gpu.validate_runtime(out, fixture.freeze, fixture.files, fixture.stage, 'c'*64,
                cpu, service, diagnostic, runtime, fixture.provenance)
        self.assertTrue(validate(records)['actual_imports_qualified'])
        mutations = [
            lambda r: r['runtime-stage-binding.json'].update(gpu_mapping_enabled=False),
            lambda r: r['gpu-mapping-validation-after-requests.json'].update(phase='wrong'),
            lambda r: r['gpu-mapping-inventory-after-model-load.json']['devices']['/dev/nvidia0'].update(inode=999),
            lambda r: r['runtime-after-requests.json']['packages']['numpy'].update(version='wrong'),
            lambda r: r['runtime-after-model-load.json'].update(mixed_module_paths=['/foreign.py']),
            lambda r: r['runtime-after-requests.json']['captured_project_modules'].pop('llm_intervention_gpu_mappings'),
            lambda r: r['runtime-before-imports.json'].update(prefix='/wrong')]
        for index, mutate in enumerate(mutations):
            bad = copy.deepcopy(records); mutate(bad)
            with self.subTest(index=index), self.assertRaises(ValueError): validate(bad)
        # The same bound flag is explicitly disallowed by the CPU-only acceptor.
        with self.assertRaises(ValueError): fixture.validate_records(records)


if __name__ == '__main__': unittest.main()
