"""Artificial runtime-contract checks. No model or numerical runtime is loaded."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import llm_intervention_model_service as service
import llm_intervention_service_runtime as runtime


class ServiceRuntimeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="ace-service-runtime-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def test_prospective_contract_cannot_relabel_historical_runtime(self):
        sources={name:"a"*64 for name in service.SOURCE_NAMES}
        sources[runtime.DIAGNOSTIC+".py"] = runtime.DIAGNOSTIC_SHA
        freeze={"schema":service.VERSION,"scope":"artificial","model_directory":"/pinned/model",
            "model_files":{name:"b"*64 for name in service.MODEL_NAMES},"sources":sources,
            "runtime":copy.deepcopy(service.RUNTIME),"python_version":[3,10,19],"recipe":service.RECIPE,
            "treatment":"frozen","adapter":None,"limits":{"startup_seconds":180,
                "request_seconds":30,"idle_seconds":30,"total_seconds":240,"max_requests":4}}
        service.validate_freeze(freeze)
        for change in (lambda f:f.update(schema="ace-pinned-qwen-service-v1"),
                       lambda f:f["runtime"]["imported_versions"].update(numpy="2.2.6"),
                       lambda f:f["sources"].update({runtime.DIAGNOSTIC+".py":"0"*64})):
            bad=copy.deepcopy(freeze)
            change(bad)
            with self.assertRaises(ValueError):
                service.validate_freeze(bad)

    def test_stage_requires_original_pins_and_exact_inventory(self):
        stage = dict(version="ace-runtime-archive-v1", archive_sha256=service.RUNTIME["archive_sha256"],
            manifest_sha256=service.RUNTIME["manifest_sha256"], staged_prefix=str(self.root),
            objects=49966, file_bytes=8397575073, model_loaded=False, runtime_qualified=False,
            cache_source_sha256={})
        runtime.check_stage(stage, service.RUNTIME, self.root)
        for key, value in {"archive_sha256":"0"*64, "manifest_sha256":"0"*64,
                           "staged_prefix":"/unrelated", "objects":49965, "file_bytes":0,
                           "model_loaded":True, "runtime_qualified":True,
                           "cache_source_sha256":None, "version":"historical"}.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                runtime.check_stage({**stage, key:value}, service.RUNTIME, self.root)

    def test_metadata_default_does_not_replace_imported_version(self):
        inventory = {"distributions": [{"name":name, "version":v}
            for name, versions in service.RUNTIME["metadata_candidates"].items() for v in versions],
            "resolved_defaults":{"numpy":{"version":"2.2.6"}}}
        checked = []
        diagnostic = types.SimpleNamespace(check_metadata=lambda value, prefix: checked.append((value,prefix)))
        runtime.check_metadata(inventory, service.RUNTIME, diagnostic, self.root)
        self.assertEqual(checked, [(inventory, self.root)])
        for version in ("2.2.5", "2.2.6"):
            bad = copy.deepcopy(inventory)
            bad["distributions"] = [r for r in bad["distributions"] if not (r["name"]=="numpy" and r["version"]==version)]
            with self.assertRaises(ValueError):
                runtime.check_metadata(bad, service.RUNTIME, diagnostic, self.root)
        with self.assertRaises(ValueError):
            runtime.check_metadata({**inventory,"distributions":inventory["distributions"]+[inventory["distributions"][0]]},
                                   service.RUNTIME, diagnostic, self.root)

    def test_imported_versions_are_exact_and_missing_packages_fail(self):
        modules = {}
        for name in runtime.EXTRA_PACKAGES:
            m = types.ModuleType(name)
            m.__version__ = service.RUNTIME["imported_versions"][name]
            modules[name] = m
        snapshot = {"packages":{name:{"version":service.RUNTIME["imported_versions"][name]}
                                 for name in ("torch","transformers","numpy","scipy")}}
        self.assertEqual(runtime.imported_versions(snapshot, modules, service.RUNTIME), service.RUNTIME["imported_versions"])
        snapshot["packages"]["numpy"]["version"] = "2.2.6"
        with self.assertRaises(ValueError):
            runtime.imported_versions(snapshot, modules, service.RUNTIME)
        snapshot["packages"]["numpy"]["version"] = "2.2.5"
        modules.pop("tokenizers")
        with self.assertRaises(ValueError):
            runtime.imported_versions(snapshot, modules, service.RUNTIME)

    def test_additional_version_mismatch_survives_in_saved_evidence(self):
        observer=runtime.ServiceRuntime.__new__(runtime.ServiceRuntime)
        observer.contract=service.RUNTIME
        observer.journal=service.Journal(self.root/'observations')
        observer.prefix=self.root/'runtime'
        observer.loader=observer.captured=observer.entry_globals=None
        observer.entry_file=self.root/'service.py'
        observer.provenance=observer.temp_root=observer.temp_stat=observer.baseline=None
        modules={}
        for name in runtime.EXTRA_PACKAGES:
            modules[name]=types.ModuleType(name)
            modules[name].__version__=service.RUNTIME['imported_versions'][name]
        modules['jinja2'].__version__='unexpected'
        snapshot={'packages':{n:{'version':service.RUNTIME['imported_versions'][n]}
            for n in ('numpy','scipy','torch','transformers')},'generated_provenance':{'verified':True}}
        observer.diagnostic=types.SimpleNamespace(RUNTIME_SOURCE='/original',
            runtime_snapshot=lambda *a,**k:copy.deepcopy(snapshot),check_prefix=lambda *a:None)
        with patch.object(runtime,'runtime_modules',return_value=(modules,{})),self.assertRaises(ValueError):
            observer.observe('version-mismatch')
        record=json.loads((observer.journal.output/'runtime-version-mismatch.json').read_bytes())
        self.assertEqual(record['additional_package_versions']['jinja2'],'unexpected')
        self.assertEqual(record['packages']['numpy']['version'],'2.2.5')
    def loaded_fixture(self):
        name = "llm_intervention_arithmetic_runtime_fixture"
        captured = {name+".py":b"answer = 6 * 7\n", "llm_intervention_model_service.py":b"entry fixture\n"}
        loader = service.CapturedSources(captured, self.root)
        sys.meta_path.insert(0, loader)
        try:
            module = importlib.import_module(name)
        finally:
            sys.meta_path.remove(loader)
            sys.modules.pop(name, None)
        return name, module, loader, captured

    def test_only_captured_objects_are_excluded_from_runtime_paths(self):
        name, module, loader, captured = self.loaded_fixture()
        (self.root/(name+".py")).write_text("answer = 999\n")
        system = types.ModuleType("system_fixture")
        system.__file__ = "/runtime/stdlib.py"
        other = types.ModuleType("untrusted")
        other.__file__ = "/untrusted/payload.py"
        filtered, records = runtime.runtime_modules({name:module,"system_fixture":system,"untrusted":other},
            loader,captured,self.root/"llm_intervention_model_service.py",{})
        self.assertEqual(module.answer,42)
        self.assertEqual(set(filtered),{"system_fixture","untrusted"})
        self.assertEqual(records[name]["sha256"],hashlib.sha256(captured[name+".py"]).hexdigest())
        # Unknown outside-runtime modules stay visible to the downstream rejection.

    def test_project_alias_and_cached_counterfeit_are_rejected(self):
        name, module, loader, captured = self.loaded_fixture()
        entry = self.root/"llm_intervention_model_service.py"
        alias = types.ModuleType("alias")
        alias.__file__ = module.__file__
        for modules in ({"alias":alias}, {"llm_intervention_unlisted":types.ModuleType("x")},
                        {name:types.ModuleType(name)}):
            with self.assertRaises(ValueError):
                runtime.runtime_modules(modules,loader,captured,entry,{})
        original = module.__loader__
        module.__loader__ = object()
        with self.assertRaises(ValueError):
            runtime.runtime_modules({name:module},loader,captured,entry,{})
        module.__loader__ = original
        loader.sources[name] = b"answer=0\n"
        with self.assertRaises(ValueError):
            runtime.runtime_modules({name:module},loader,captured,entry,{})

    def test_entrypoint_exception_requires_same_globals_and_path(self):
        _,_,loader,captured = self.loaded_fixture()
        entry = types.ModuleType("__main__")
        entry.__file__ = str(self.root/"llm_intervention_model_service.py")
        filtered, records = runtime.runtime_modules({"__main__":entry,"__mp_main__":entry},loader,
            captured,Path(entry.__file__),vars(entry))
        self.assertEqual(filtered,{})
        self.assertEqual(set(records),{"__main__","__mp_main__"})
        with self.assertRaises(ValueError):
            runtime.runtime_modules({"__main__":entry},loader,captured,Path(entry.__file__),dict(vars(entry)))

    def test_fixture_roster_pins_and_no_decoding_corrupt_input(self):
        args=[]
        for name in sorted(runtime.FIXTURES):
            data=json.dumps({"fixture":name}).encode()
            (self.root/name).write_bytes(data)
            args.append(name+"="+hashlib.sha256(data).hexdigest())
        rows=runtime.fixture_inputs(args,self.root,service.read_pinned,service.decode)
        self.assertEqual(set(rows),runtime.FIXTURES)
        for bad in (args[:2],args+[args[0]],args+["../bad.json="+"0"*64],[] ):
            with self.assertRaises(ValueError):
                runtime.fixture_inputs(bad,self.root,service.read_pinned,service.decode)
        (self.root/"case0.json").write_bytes(b"corrupt")
        decoded=[]
        with self.assertRaises(ValueError):
            runtime.fixture_inputs(args,self.root,service.read_pinned,lambda raw:decoded.append(raw))
        self.assertEqual(decoded,[])

    def test_cpu_tokenization_records_all_public_fixtures_without_model_loader(self):
        transformers=types.ModuleType("transformers")
        calls=[]
        tokenizer=types.SimpleNamespace(eos_token_id=151645)
        transformers.AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:(calls.append((a,k)) or tokenizer))
        language=types.ModuleType("llm_intervention_language_policy")
        def prepare(envelope, tok, **kw):
            self.assertIs(tok,tokenizer)
            self.assertEqual(kw,{"max_tokens":8192,"max_new_tokens":32})
            return types.SimpleNamespace(prompt_ids=(1,2,3),completion_ids=((4,151645),)), {"envelope_sha256":service.digest(envelope)}
        language.prepare_task_tokens=prepare
        fixture={name:{"sha256":"a"*64,"envelope":{"name":name}} for name in runtime.FIXTURES}
        freeze={"scope":"artificial","model_directory":"/pinned/model","recipe":service.RECIPE}
        journal=service.Journal(self.root/"output")
        with patch.dict(sys.modules,{"transformers":transformers,"llm_intervention_language_policy":language}):
            runtime.verify_cpu_tokens(freeze,fixture,journal)
            with self.assertRaises(ValueError):
                runtime.verify_cpu_tokens({**freeze,"scope":"development"},fixture,journal)
        result=json.loads((journal.output/"cpu-tokenization.json").read_text())
        self.assertEqual(len(result["fixtures"]),3)
        self.assertEqual(calls,[(('/pinned/model',),{"local_files_only":True,"trust_remote_code":False})])
        self.assertFalse(result["model_weights_loaded"])
        self.assertFalse(result["explicit_cuda_api_called"])


if __name__ == "__main__":
    unittest.main()
