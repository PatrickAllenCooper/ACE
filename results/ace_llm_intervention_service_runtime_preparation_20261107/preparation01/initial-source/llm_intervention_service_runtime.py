"""Prospective service adapter for the accepted captured-runtime diagnostic.

The trusted outer launcher must stage the archive and independently pin its
receipt. This module checks that receipt, observed imports and captured project
modules. It does not authenticate an arbitrary runtime from receipt text alone.
No model weights, CUDA APIs, responses or training are used by this adapter.
"""
from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
import re
import sys
import tempfile
import types


DIAGNOSTIC = "qualify_llm_intervention_runtime_metadata"
DIAGNOSTIC_SHA = "31bb109e2ce9603e72fe97e1cb3fcfdff2b1a72b3f7eaf74431ca92f1cd7c329"
EXTRA_PACKAGES = ("safetensors", "huggingface_hub", "tokenizers", "jinja2")
FIXTURES = frozenset({"case0.json", "case3.json", "stress.json"})


def need(condition, message):
    if not condition:
        raise ValueError(message)


def check_stage(stage, contract, prefix):
    """Receipt binding only. The external archive stager supplies byte proof."""
    need(type(stage) is dict and stage.get("version") == "ace-runtime-archive-v1",
         "unknown archive staging receipt")
    need(stage.get("archive_sha256") == contract["archive_sha256"]
         and stage.get("manifest_sha256") == contract["manifest_sha256"]
         and stage.get("staged_prefix") == str(prefix)
         and stage.get("objects") == 49966 and stage.get("file_bytes") == 8397575073
         and stage.get("model_loaded") is False and stage.get("runtime_qualified") is False,
         "archive stage binding differs")
    need(type(stage.get("cache_source_sha256")) is dict, "cache/source pins absent")


def check_metadata(inventory, contract, diagnostic, prefix):
    diagnostic.check_metadata(inventory, prefix)
    candidates = {}
    for row in inventory["distributions"]:
        key = re.sub(r"[-_.]+", "-", row["name"]).lower()
        candidates.setdefault(key, []).append(row["version"])
    need(all(sorted(candidates.get(k, [])) == v
             for k, v in contract["metadata_candidates"].items()),
         "captured metadata candidates differ")
    # resolved_defaults are retained as observations, never imported versions.


def runtime_modules(modules, loader, captured, entry_file, entry_globals):
    """Exclude only exact captured project objects before the runtime check.

    Executed entrypoint bytes are authenticated by the outer bootstrap. Its
    globals can be distinct from sys.modules['__main__'] under exec bootstrap.
    This is a trusted-process consistency check, not hostile-code attestation.
    """
    filtered, records = dict(modules), {}
    project_paths = {str((loader.root/name).resolve()) for name in captured}
    for name, module in list(modules.items()):
        if not isinstance(module, types.ModuleType):
            continue
        data = vars(module)
        file = data.get("__file__")
        if name in loader.sources:
            expected = str(loader.root/(name+".py"))
            spec = data.get("__spec__")
            need(loader.loaded.get(name) is module and data.get("__loader__") is loader
                 and spec is not None and spec.loader is loader and spec.name == name
                 and spec.origin == expected and file == expected,
                 "uncaptured project module: "+name)
            raw = captured[name+".py"]
            need(loader.sources[name] == raw, "captured project bytes changed: "+name)
            records[name] = {"path": expected, "sha256": hashlib.sha256(raw).hexdigest(),
                             "loader": "authenticated_captured_bytes"}
            filtered.pop(name)
        elif type(file) is str and not file.startswith("<") and str(Path(file).resolve()) in project_paths:
            need(name in ("__main__", "__mp_main__") and data is entry_globals
                 and file == str(entry_file), "unverified project alias: "+name)
            records[name] = {"path": file,
                "sha256": hashlib.sha256(captured[entry_file.name]).hexdigest(),
                "loader": "outer_authenticated_entrypoint"}
            filtered.pop(name)
        elif name.startswith("llm_intervention_") or name == DIAGNOSTIC:
            raise ValueError("project module absent from captured closure: "+name)
    return filtered, records


def imported_versions(snapshot, modules, contract):
    observed = {k: v["version"] for k, v in snapshot["packages"].items()}
    for name in EXTRA_PACKAGES:
        module = modules.get(name)
        need(isinstance(module, types.ModuleType), "expected package not loaded: "+name)
        observed[name] = vars(module).get("__version__")
    need(observed == contract["imported_versions"], "imported package versions differ")
    return observed


def fixture_inputs(arguments, root, read_pinned, decode):
    pins = {}
    for item in arguments:
        name, separator, sha = item.partition("=")
        need(separator and name in FIXTURES and name not in pins,
             "exact three unique CPU fixture pins required")
        pins[name] = sha
    need(set(pins) == FIXTURES, "exact CPU fixture roster required")
    return {name: {"sha256": sha, "envelope": decode(read_pinned(root/name, sha, capture=True))}
            for name, sha in sorted(pins.items())}


class ServiceRuntime:
    def __init__(self, freeze, captured, loader, entry_file, entry_globals,
                 stage_path, stage_sha256, journal, read_pinned, decode):
        self.contract, self.journal = freeze["runtime"], journal
        self.loader, self.captured = loader, captured
        self.entry_file, self.entry_globals = Path(entry_file), entry_globals
        need(self.contract["diagnostic_source_sha256"] == DIAGNOSTIC_SHA
             and hashlib.sha256(captured[DIAGNOSTIC+".py"]).hexdigest() == DIAGNOSTIC_SHA,
             "accepted diagnostic source differs")
        need(sys.flags.isolated == 1 and sys.dont_write_bytecode,
             "isolated interpreter with bytecode writes disabled required")
        self.diagnostic = importlib.import_module(DIAGNOSTIC)
        d = self.diagnostic
        self.prefix = Path(sys.prefix).resolve(strict=True)
        need(not d.inside(self.prefix, d.RUNTIME_SOURCE), "original environment cannot be the stage")
        stage = decode(read_pinned(stage_path, stage_sha256, capture=True))
        check_stage(stage, self.contract, self.prefix)
        need(set(stage["cache_source_sha256"]) == {p for pair in d.CACHE_PAIRS for p in pair},
             "exact three cache/source pairs required")
        self.temp_root = self.prefix.parent/("service-tmp-"+str(os.getpid()))
        self.temp_root.mkdir(mode=0o700, exist_ok=False)
        self.temp_stat = d.owned_stat(self.temp_root, directory=True, private=True)
        need(not d.inside(self.temp_root, self.prefix), "temporary directory overlaps runtime")
        os.environ.update({k: str(self.temp_root) for k in ("TMPDIR", "TMP", "TEMP")})
        # This fresh isolated process has not imported numerical/model modules.
        need(not {"torch", "numpy", "scipy", "transformers"}.intersection(sys.modules),
             "numerical runtime imported before provenance initialization")
        tempfile.tempdir = None
        need(tempfile.gettempdir() == str(self.temp_root) and not list(self.temp_root.iterdir()),
             "exclusive temporary root differs")
        journal.save("runtime-stage-binding.json", {"stage_sha256": stage_sha256, "stage": stage,
            "temporary_root": str(self.temp_root), "temporary_root_stat": self.temp_stat,
            "contract": self.contract, "outer_archive_execution_required": True})
        self.provenance = d.provenance_contract(self.prefix)
        journal.save("runtime-provenance-sources.json", self.provenance)
        self.baseline = None
        before = self.observe("before-imports", require_imports=False)
        self.baseline = list(before["sys_path"])
        need(before["python_version"] == "3.10.19", "actual Python version differs")
        d.capture_caches(self.prefix, stage["cache_source_sha256"], journal.output)
        metadata = d.metadata_inventory()
        journal.save("runtime-metadata.json", metadata)
        check_metadata(metadata, self.contract, d, self.prefix)

    def observe(self, label, *, require_imports=True):
        d = self.diagnostic
        modules, projects = runtime_modules(sys.modules, self.loader, self.captured,
                                            self.entry_file, self.entry_globals)
        snapshot = d.runtime_snapshot(self.prefix, d.RUNTIME_SOURCE, modules=modules,
            contract=self.provenance, temp_root=self.temp_root, temp_stat=self.temp_stat)
        # Keep complete evidence before validation, including failures.
        self.journal.save("runtime-"+label+".json", {**snapshot, "captured_project_modules": projects})
        d.check_prefix(self.prefix, snapshot, self.baseline)
        if require_imports:
            need(snapshot["generated_provenance"]["verified"], "generated Torch provenance absent")
            imported_versions(snapshot, modules, self.contract)
        return snapshot

    def import_cpu_closure(self):
        self.journal.event("service_cpu_import_start")
        self.diagnostic.original_imports(self.journal.event)
        for name in EXTRA_PACKAGES:
            importlib.import_module(name)
        # Exercise the entire captured library closure without a model or CUDA.
        for name in sorted(self.loader.sources):
            importlib.import_module(name)
        self.observe("after-service-imports")
        self.journal.event("service_cpu_import_complete")


def verify_cpu_tokens(freeze, fixtures, journal):
    """Tokenize only pinned public artificial envelopes, with no model weights."""
    from transformers import AutoTokenizer
    from llm_intervention_language_policy import prepare_task_tokens
    need(freeze["scope"] == "artificial" and set(fixtures) == FIXTURES,
         "CPU qualification needs the complete artificial fixture roster")
    journal.event("cpu_tokenizer_load_start")
    tokenizer = AutoTokenizer.from_pretrained(freeze["model_directory"],
        local_files_only=True, trust_remote_code=False)
    need(tokenizer.eos_token_id == 151645, "tokenizer EOS differs")
    rows = []
    for name, fixture in sorted(fixtures.items()):
        grammar, metadata = prepare_task_tokens(fixture["envelope"], tokenizer,
            max_tokens=freeze["recipe"]["max_tokens"], max_new_tokens=freeze["recipe"]["max_new_tokens"])
        rows.append({"name": name, "fixture_sha256": fixture["sha256"], **metadata,
                     "prompt_ids": list(grammar.prompt_ids),
                     "completion_ids": [list(x) for x in grammar.completion_ids]})
    journal.save("cpu-tokenization.json", {"fixtures": rows, "model_weights_loaded": False,
        "adapter_deserialized": False, "explicit_cuda_api_called": False,
        "new_training_updates": 0, "environment_responses": 0})
    journal.event("cpu_tokenization_complete", fixtures=len(rows))
