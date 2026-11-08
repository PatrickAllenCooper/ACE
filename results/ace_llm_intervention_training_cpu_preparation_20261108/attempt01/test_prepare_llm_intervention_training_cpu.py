"""New boundary checks use saved bytes and fabricated tokens, never Qwen."""
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import prepare_llm_intervention_training_cpu as prep
import llm_intervention_model_service as service
import qualify_llm_intervention_service_cpu as cpu
import qualify_llm_intervention_runtime_metadata as diagnostic

REPO = Path(__file__).resolve().parents[2]
COLLECTION = REPO/'results/ace_llm_intervention_training_collection_preparation_20261107/attempt02/fixtures/test_two_world_complete_inventory_drives_exposure_and_exclusive_intent/collection'


def fixture():
    data = {name:(COLLECTION/old).read_bytes() for name,old in [
        ('collection-terminal.json','terminal.json'),('collection-inventory.json','inventory.json'),('collection-plan.json','plan.json')]}
    inventory = service.decode(data['collection-inventory.json'])
    for i,row in enumerate(inventory['records']):
        for kind in ['record','receipt']:
            data[f'{kind}-{i}.json'] = (COLLECTION/row['path']/(kind+'.json')).read_bytes()
    prior = service.decode((REPO/'results/ace_llm_intervention_service_cpu_20261107/launch04/preflight.json').read_bytes())
    sources = service.SOURCE_NAMES | prep.EXTRA_SOURCES
    packet = {'schema':prep.VERSION,'files':{n:'a'*64 for n in sources | prep.BATCH_FILES |
        {'probe.py','llm_intervention_runtime_archive.py'}},'source_names':sorted(sources),
        'scope':'artificial','record_roster':list(prep.ROSTER),'archive':prior['archive'],'manifest':prior['manifest'],
        'runtime_source':diagnostic.RUNTIME_SOURCE,'resources':cpu.RESOURCES,'runtime':service.RUNTIME,
        'model_directory':'/fabricated/model','model_files':{n:'b'*64 for n in service.MODEL_NAMES}}
    packet['files'].update({n:prep.sha(b) for n,b in data.items()})
    packet['files'].update({prep.CPU:prep.CPU_SHA,prep.DIAGNOSTIC:prep.DIAGNOSTIC_SHA,
        'probe.py':diagnostic.PROBE_SHA,'llm_intervention_runtime_archive.py':diagnostic.HELPER_SHA,
        'llm_intervention_gpu_mappings.py':service.RUNTIME['gpu_mapping_source_sha256']})
    # Start from the preserved artificial-tokenizer artifact, not a new bridge run.
    choices = list((REPO/'results/ace_llm_intervention_training_plan_preparation_20261108/attempt01').rglob('artificial-prepared-plan.json'))
    if len(choices)!=1:raise AssertionError('one saved artificial plan required')
    result = service.decode(choices[0].read_bytes())
    result['tokenizer_sha256'] = service.digest(packet['model_files'])
    plan = result['training_plan']
    plan['outer_provenance_receipt_sha256']='c'*64
    plan['checkpoint_steps']=[3,5]
    for rec in plan['records']:
        rec['pad_token_id'],rec['eos_token_id']=151643,151645
        rec['tokenizer_sha256']=result['tokenizer_sha256']
        for ids in rec['commands'].values():ids[-1]=151645
    rebind(result)
    return packet,data,result


def rebind(result):
    plan=result['training_plan']
    plan['record_pins']={r['record_id']:service.digest(r) for r in plan['records']}
    result['training_plan_sha256']=service.digest(plan)
    result['exposure_sha256']=service.digest(result['exposure'])


class TrainingCPUChecks(unittest.TestCase):
    def test_exact_roster_resource_source_and_original_collection_gate(self):
        packet,_,_=fixture()
        prep.packet_shape(packet,service,cpu,diagnostic)
        for change in [lambda p:p['files'].pop('record-3.json'),
                       lambda p:p['files'].update({'extra.json':'a'*64}),
                       lambda p:p['source_names'].pop(),
                       lambda p:p['resources'].update(gpus=1),
                       lambda p:p.update(scope='policy_train'),
                       lambda p:p['record_roster'].reverse(),
                       lambda p:p['files'].update({'collection-terminal.json':'0'*64})]:
            bad=copy.deepcopy(packet);change(bad)
            with self.assertRaises(ValueError):prep.packet_shape(bad,service,cpu,diagnostic)

    def test_saved_prepared_output_cannot_drop_reorder_or_relabel_rows(self):
        packet,data,result=fixture()
        self.assertEqual(prep.validate_prepared(result,packet,data,service,'c'*64)['records'],4)
        for change in [lambda p:p['records'].pop(),lambda p:p['records'].reverse(),
                       lambda p:p.update(outer_provenance_receipt_sha256='e'*64),
                       lambda p:p['records'][0].update(raw_record_utf8='{}'),
                       lambda p:p['records'][0]['commands'][next(iter(p['records'][0]['commands']))].__setitem__(-1,0),
                       lambda p:p.update(stored_tokens=p['stored_tokens']+1),
                       lambda p:p['schedule'][-1].update(pair_index=0)]:
            bad=copy.deepcopy(result);change(bad['training_plan']);rebind(bad)
            with self.assertRaises(ValueError):prep.validate_prepared(bad,packet,data,service,'c'*64)

    def test_capture_rejects_bad_pin_symlink_and_oversize(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'input';p.write_bytes(b'fixed')
            link=Path(td)/'link';link.symlink_to(p)
            self.assertEqual(prep.capture(p,prep.sha(b'fixed')),b'fixed')
            for path,pin,limit in [(p,'0'*64,10),(link,prep.sha(b'fixed'),10),(p,prep.sha(b'fixed'),3)]:
                with self.assertRaises((ValueError,OSError)):prep.capture(path,pin,limit)

    def test_captured_loader_uses_captured_bytes_and_rejects_cache(self):
        name='llm_intervention_new_boundary_fixture'
        raw=b'VALUE = 17\n'
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/(name+'.py')).write_text('raise RuntimeError("disk source executed")\n')
            loader=prep.CapturedSources({name+'.py':raw,prep.ENTRY:b'ignored'},root)
            sys.meta_path.insert(0,loader)
            try:
                m=importlib.import_module(name)
                self.assertEqual(m.VALUE,17)
                self.assertIs(loader.loaded[name],m)
                with self.assertRaises(ValueError):prep.CapturedSources({name+'.py':raw},root)
                with self.assertRaises(ImportError):loader.find_spec('llm_intervention_not_in_closure')
            finally:
                sys.meta_path.remove(loader);sys.modules.pop(name,None)

    def test_explicit_cpu_environment_overrides_hidden_threading(self):
        # Own process only. Child receives a full explicit environment mapping.
        with patch.dict(os.environ,{'PYTHONPATH':'/unwanted','MKL_SERVICE_FORCE_INTEL':'1','CUDA_VISIBLE_DEVICES':'2'}):
            os.putenv('MKL_THREADING_LAYER','INTEL')
            env=prep.cpu_environment(Path('/copied/runtime'),os.environ)
            self.assertEqual(env['MKL_THREADING_LAYER'],'GNU')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
            for n in ['PYTHONPATH','PYTHONHOME','LD_PRELOAD','MKL_SERVICE_FORCE_INTEL']:
                self.assertNotIn(n,env)
            import ctypes
            libc=ctypes.CDLL(None);libc.getenv.argtypes=[ctypes.c_char_p];libc.getenv.restype=ctypes.c_char_p
            self.assertEqual(libc.getenv(b'MKL_THREADING_LAYER'),b'GNU')
            self.assertIsNone(libc.getenv(b'MKL_SERVICE_FORCE_INTEL'))

    def test_entry_import_keeps_numerical_libraries_unloaded(self):
        self.assertFalse({'torch','numpy','scipy','transformers'}.intersection(sys.modules))


if __name__=='__main__':unittest.main()
