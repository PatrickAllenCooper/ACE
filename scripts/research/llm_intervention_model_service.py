#!/usr/bin/env python3
"""Pinned, inference-only Qwen service for the persistent public-state protocol.

This entrypoint authenticates files and executes captured project source bytes.
It does not establish training provenance, experimental usefulness, scheduler
allocation or protection from malicious same-user code. External supervision
and a separately reviewed scientific launch remain necessary.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time
import traceback


VERSION = "ace-pinned-qwen-service-v2"
SOURCE_NAMES = frozenset({"llm_intervention_model_service.py",
    "llm_intervention_persistent_policy.py", "llm_intervention_contract.py",
    "llm_intervention_episode.py", "llm_intervention_task.py", "llm_intervention_scm.py",
    "llm_intervention_numerical_policy.py", "llm_intervention_language_policy.py",
    "llm_intervention_learning.py", "llm_intervention_grammar.py",
    "llm_intervention_canonical_scoring.py", "llm_intervention_action_scoring.py",
    "llm_intervention_precision.py", "llm_intervention_service_runtime.py",
    "qualify_llm_intervention_runtime_metadata.py"})
MODEL_NAMES = frozenset({"model.safetensors", "config.json", "generation_config.json",
    "tokenizer_config.json", "tokenizer.json", "merges.txt", "vocab.json"})
RUNTIME = {
    "schema": "ace-staged-service-runtime-v1",
    "archive_sha256": "6815ab9784b6025b8bb4439e40a1a012b9697d434bfc83bd5f36e27abc262d98",
    "manifest_sha256": "b44610f638f0873da8588f1cf23039638203a855959a6ed12cf0ec6c2551d63a",
    "diagnostic_source_sha256": "31bb109e2ce9603e72fe97e1cb3fcfdff2b1a72b3f7eaf74431ca92f1cd7c329",
    "diagnostic_acceptance_sha256": "6e557150edb12f0db45c3b0baa9c72eeb63d763af769c062d3c8a556b633f578",
    "imported_versions": {"torch": "2.9.1+cu128", "transformers": "4.57.3", "numpy": "2.2.5",
        "scipy": "1.15.3", "safetensors": "0.7.0", "huggingface_hub": "0.36.0",
        "tokenizers": "0.22.1", "jinja2": "3.1.6"},
    "metadata_candidates": {"torch": ["2.9.1"], "transformers": ["4.57.3"],
        "numpy": ["2.2.5", "2.2.6"], "scipy": ["1.15.3"], "safetensors": ["0.7.0"],
        "huggingface-hub": ["0.36.0"], "tokenizers": ["0.22.1"], "jinja2": ["3.1.6"]}}
RECIPE = {"base_dtype": "bfloat16", "adapter_dtype": "float32", "rank": 8,
    "alpha": 16., "attention": "sdpa", "sdpa_backend": "flash_attention_only",
    "strict_determinism": True, "float32_matmul": "highest", "tf32": False,
    "cublas_workspace_config": ":4096:8", "max_tokens": 8192, "max_new_tokens": 32,
    "output_projection": "fixed_row_fp32_action_reduction_v1", "thinking": False,
    "decoding": "greedy_single_prompt_fresh_cache", "new_training_updates": 0}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def pin(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def decode(raw):
    def obj(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def bad(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=obj, parse_constant=bad)


def keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("unexpected manifest fields")


def validate_freeze(freeze):
    keys(freeze, {"schema", "scope", "model_directory", "model_files", "sources",
        "runtime", "python_version", "recipe", "treatment", "adapter", "limits"})
    if (freeze["schema"] != VERSION or freeze["scope"] not in ("artificial", "development")
            or freeze["runtime"] != RUNTIME or freeze["python_version"] != [3, 10, 19]
            or freeze["recipe"] != RECIPE or not Path(freeze["model_directory"]).is_absolute()):
        raise ValueError("unqualified scope/runtime/recipe/model location")
    for section, names in (("sources", SOURCE_NAMES), ("model_files", MODEL_NAMES)):
        keys(freeze[section], names)
        if not all(pin(p) for p in freeze[section].values()):
            raise ValueError("invalid file pin")
    if freeze["sources"]["qualify_llm_intervention_runtime_metadata.py"] != RUNTIME["diagnostic_source_sha256"]:
        raise ValueError("accepted diagnostic source pin differs")
    limits = freeze["limits"]
    keys(limits, {"startup_seconds", "request_seconds", "idle_seconds", "total_seconds", "max_requests"})
    for name, cap in (("startup_seconds", 180), ("request_seconds", 60),
                      ("idle_seconds", 60), ("total_seconds", 3600), ("max_requests", 1024)):
        if type(limits[name]) is not int or not 1 <= limits[name] <= cap:
            raise ValueError("invalid bounded service limit")
    if max(limits[k] for k in ("startup_seconds", "request_seconds", "idle_seconds")) > limits["total_seconds"]:
        raise ValueError("stage deadline exceeds session deadline")
    if freeze["treatment"] == "frozen":
        if freeze["adapter"] is not None:
            raise ValueError("frozen treatment cannot load an adapter")
    elif freeze["treatment"] in ("sft", "dpo", "artificial_adapter"):
        adapter = freeze["adapter"]
        keys(adapter, {"file", "sha256", "training_receipt", "training_receipt_sha256", "training_scope"})
        if (not pin(adapter["sha256"]) or not pin(adapter["training_receipt_sha256"])
                or adapter["training_scope"] not in ("artificial", "policy_train")
                or (freeze["treatment"] == "artificial_adapter") != (adapter["training_scope"] == "artificial")
                or (freeze["scope"] == "development" and adapter["training_scope"] != "policy_train")):
            raise ValueError("adapter scope or pins invalid")
        for name in (adapter["file"], adapter["training_receipt"]):
            if type(name) is not str or Path(name).name != name or name in ("", ".", ".."):
                raise ValueError("adapter inputs must be plain filenames beside freeze")
    else:
        raise ValueError("unknown treatment")
    return freeze


def read_pinned(path, expected, *, capture=False):
    """No-follow regular-file read, hashing the exact captured input bytes.

    Filesystem tree and model loading are trusted against concurrent hostile
    same-user replacement. Project modules use these captured bytes instead of
    reopening source or accepting a preexisting bytecode cache.
    """
    if not pin(expected):
        raise ValueError("independent SHA-256 required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    h, blocks = hashlib.sha256(), []
    with os.fdopen(fd, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("regular input file required")
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
            if capture:
                blocks.append(block)
        after = os.fstat(handle.fileno())
    if ((before.st_size, before.st_mtime_ns, before.st_ino) !=
            (after.st_size, after.st_mtime_ns, after.st_ino) or h.hexdigest() != expected):
        raise ValueError("input changed or digest differs: " + str(path))
    return b"".join(blocks) if capture else {"bytes": before.st_size, "sha256": expected}


def authenticate(freeze_path, freeze_sha256, *, runtime=True, on_manifest=None):
    freeze_path = Path(freeze_path).absolute()
    freeze = validate_freeze(decode(read_pinned(freeze_path, freeze_sha256, capture=True)))
    if on_manifest is not None:
        on_manifest(freeze)  # Tighten deadlines before any remaining input read.
    source = freeze_path.parent
    model = Path(freeze["model_directory"])
    if model.is_symlink() or not model.is_dir() or {p.name for p in model.iterdir()} != MODEL_NAMES:
        raise ValueError("model directory membership differs")
    captured = {name: read_pinned(source/name, sha, capture=True) for name, sha in freeze["sources"].items()}
    models = {name: read_pinned(model/name, sha) for name, sha in freeze["model_files"].items()}
    adapter_raw = None
    if freeze["adapter"] is not None:
        a = freeze["adapter"]
        adapter_raw = read_pinned(source/a["file"], a["sha256"], capture=True)
        # Bind receipt bytes. A separately reviewed launch must establish that
        # this receipt actually qualifies the scientific training history.
        read_pinned(source/a["training_receipt"], a["training_receipt_sha256"])
    # Imported versions and provenance are checked by ServiceRuntime after
    # captured-source installation. Metadata defaults cannot establish them.
    if runtime and list(sys.version_info[:3]) != freeze["python_version"]:
        raise ValueError("actual Python version differs")
    return freeze, captured, adapter_raw, models


def identity(freeze):
    adapter = freeze["adapter"]
    return {"model_sha256": digest(freeze["model_files"]), "source_sha256": digest(freeze["sources"]),
        "adapter_sha256": adapter["sha256"] if adapter else digest({"treatment": "frozen", "adapter": None}),
        "numeric_recipe": freeze["recipe"], "numeric_recipe_sha256": digest(freeze["recipe"])}


class CapturedSources(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import only authenticated in-memory project sources, with no pyc cache."""
    def __init__(self, captured, root):
        self.sources = {Path(name).stem: raw for name, raw in captured.items()
                        if name != "llm_intervention_model_service.py"}
        self.root = Path(root)
        self.loaded = {}
        if set(self.sources) & set(sys.modules):
            raise ValueError("project module was imported before authentication")

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.sources:
            return importlib.util.spec_from_loader(fullname, self, origin=str(self.root/(fullname+".py")))
        if fullname.startswith("llm_intervention_"):
            raise ImportError("project import absent from pinned closure: " + fullname)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = str(self.root/(module.__name__+".py"))
        exec(compile(self.sources[module.__name__], module.__file__, "exec"), module.__dict__)
        self.loaded[module.__name__] = module


class Journal:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(exist_ok=False)
        self.started, self.cpu, self.events = time.monotonic(), time.process_time(), 0

    def save(self, name, value):
        with (self.output/name).open("xb") as handle:
            handle.write((canonical(value)+"\n").encode())
            handle.flush()
            os.fsync(handle.fileno())

    def event(self, phase, **fields):
        self.save("phase-%03d.json" % self.events, {"phase": phase,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": time.monotonic()-self.started,
            "process_cpu_seconds": time.process_time()-self.cpu, **fields})
        self.events += 1


@contextlib.contextmanager
def capture_generation(model, journal, ordinal):
    """Capture bounded returned tokens before the shared helper validates them.

    A sequential, temporary instance wrapper preserves the exact return value
    and restores the original dispatch. It is not a generation implementation.
    """
    import torch
    had_instance = "generate" in model.__dict__
    previous_instance = model.__dict__.get("generate")
    original = model.generate
    calls = 0
    def captured(*args, **kwargs):
        nonlocal calls
        if calls:
            raise ValueError("one generation call per public request required")
        calls += 1
        journal.save(f"raw-generation-{ordinal:04d}.started.json",
                     {"tokens_available": False, "status": "generation_invoked"})
        value = original(*args, **kwargs)
        record = {"status": "returned_unaccepted", "tokens_available": True,
                  "returned_type": type(value).__name__}
        if type(value) is torch.Tensor and value.layout == torch.strided:
            # Copy only a bounded prefix even when the returned shape is invalid.
            detached = value.detach()
            count = min(value.numel(), 16384)
            if detached.is_contiguous():
                prefix = detached.view(-1)[:count]
            else:
                # Advanced indexing gathers a bounded logical row-major prefix.
                # Flattening a transposed return first could copy it in full.
                offsets = torch.arange(count, dtype=torch.long, device=value.device)
                coordinates = []
                for size in reversed(value.shape):
                    coordinates.append(offsets.remainder(size))
                    offsets = torch.div(offsets, size, rounding_mode="floor")
                prefix = detached[tuple(reversed(coordinates))]
            prefix = prefix.contiguous().cpu()
            raw = prefix.view(torch.uint8).numpy().tobytes()
            with (journal.output/f"raw-generation-{ordinal:04d}.bin").open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            record.update(shape=list(value.shape), dtype=str(value.dtype), numel=value.numel(),
                captured_elements=prefix.numel(), truncated=prefix.numel() != value.numel(),
                raw_sha256=hashlib.sha256(raw).hexdigest())
            if value.dtype == torch.long:
                record["token_prefix"] = prefix.tolist()
        else:
            record["tokens_available"] = False
            record["reason"] = "return was not a strided token tensor"
        journal.save(f"raw-generation-{ordinal:04d}.json", record)
        return value
    model.generate = captured
    try:
        yield
    finally:
        if had_instance:
            model.generate = previous_instance
        else:
            del model.generate


def initialize_qwen(freeze, adapter_raw, journal, arm_deadline):
    """Load one authenticated base, then return a stateless-request decision closure."""
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from llm_intervention_language_policy import restore_adapter, generate_task_decision
    from llm_intervention_canonical_scoring import install_canonical_projection
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError("exactly one BF16-compatible visible GPU required")
    if torch.version.cuda != "12.8":
        raise ValueError("CUDA build differs from qualified recipe")
    model_dir = freeze["model_directory"]
    journal.event("model_cpu_load_start")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if (type(model) is not Qwen3ForCausalLM or model.config._attn_implementation != "sdpa"
            or model.config.num_hidden_layers != 28 or model.config.hidden_size != 1024
            or model.config.vocab_size != 151936 or tokenizer.eos_token_id != 151645):
        raise ValueError("actual model architecture/tokenizer differs")
    if adapter_raw is not None:
        names = tuple(f"model.layers.{i}.self_attn.{part}" for i in range(28) for part in ("q_proj", "v_proj"))
        a = freeze["adapter"]
        journal.save("adapter_restoration.json", restore_adapter(model, adapter_raw,
            adapter_sha256=a["sha256"], training_receipt_sha256=a["training_receipt_sha256"],
            expected_projection_names=names, rank=8, alpha=16.))
    model.requires_grad_(False)
    model.eval()
    journal.event("model_cpu_load_complete")
    model.to("cuda")
    journal.save("projection.json", install_canonical_projection(model))
    model.eval()
    torch.cuda.synchronize()
    def versions():
        return tuple((n, id(t), t.data_ptr(), t._version, tuple(t.shape), str(t.dtype), str(t.device))
                     for n, t in tuple(model.named_parameters())+tuple(model.named_buffers()))
    baseline = versions()
    count = 0
    journal.event("model_ready", device=torch.cuda.get_device_name(0), cuda_build=torch.version.cuda,
        gpu_allocated_bytes=torch.cuda.memory_allocated(), identity=identity(freeze))
    arm_deadline(freeze["limits"]["idle_seconds"])

    def decide(envelope):
        nonlocal count
        arm_deadline(freeze["limits"]["request_seconds"])
        if versions() != baseline or any(p.requires_grad for p in model.parameters()) or model.training:
            raise ValueError("inference model state changed")
        ordinal = count
        count += 1
        journal.save(f"request-{ordinal:04d}.json", {"ordinal": ordinal,
            "envelope_sha256": digest(envelope), "identity": identity(freeze),
            "tokens_available_at_request_start": False})
        start = time.monotonic()
        first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats()
        first.record()
        # Each invocation constructs a new input tensor, generation config and
        # KV cache. No previous history, sequence or cache is an input here.
        with capture_generation(model, journal, ordinal), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            completion, record = generate_task_decision(model, tokenizer, envelope,
                max_tokens=8192, max_new_tokens=32)
        journal.save(f"generated-{ordinal:04d}.unaccepted.json", record)
        last.record()
        torch.cuda.synchronize()
        if versions() != baseline:
            raise ValueError("inference mutated model tensors")
        journal.save(f"generation-{ordinal:04d}.json", {**record, "ordinal": ordinal,
            "elapsed_seconds": time.monotonic()-start, "cuda_interval_ms": first.elapsed_time(last),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "tensor_versions_unchanged": True, "cross_request_cache_supplied": False})
        journal.event("generation_complete", ordinal=ordinal, envelope_sha256=digest(envelope))
        arm_deadline(freeze["limits"]["idle_seconds"])
        return completion
    return decide


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime-stage", required=True)
    parser.add_argument("--runtime-stage-sha256", required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--fixture", action="append", default=[], help="CPU-only NAME=SHA256 public fixture")
    args = parser.parse_args()
    journal = Journal(args.output)
    loader, runtime_observer = None, None
    def timeout(signum, frame):
        raise TimeoutError("bounded model-service deadline reached")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 180.)
    total_end = journal.started+180.
    def arm_deadline(seconds):
        remaining = total_end-time.monotonic()
        if remaining <= 0:
            raise TimeoutError("total model-service deadline reached")
        signal.setitimer(signal.ITIMER_REAL, min(seconds, remaining))
    try:
        journal.event("authentication_start")
        def tighten(authenticated):
            nonlocal total_end
            total_end = journal.started+authenticated["limits"]["total_seconds"]
            startup_remaining = authenticated["limits"]["startup_seconds"]-(time.monotonic()-journal.started)
            if startup_remaining <= 0:
                raise TimeoutError("freeze verification exceeded startup deadline")
            arm_deadline(startup_remaining)
        freeze, captured, adapter_raw, models = authenticate(args.freeze, args.freeze_sha256,
                                                              on_manifest=tighten)
        if Path(__file__).absolute() != Path(args.freeze).absolute().parent/"llm_intervention_model_service.py":
            raise ValueError("entrypoint must be the frozen source beside the manifest")
        total_end = journal.started+freeze["limits"]["total_seconds"]
        startup_remaining = freeze["limits"]["startup_seconds"]-(time.monotonic()-journal.started)
        if startup_remaining <= 0:
            raise TimeoutError("authentication exceeded startup deadline")
        arm_deadline(startup_remaining)
        journal.save("authenticated.json", {"freeze_sha256": args.freeze_sha256,
            "identity": identity(freeze), "models": models, "runtime": RUNTIME,
            "runtime_scope": "declared prospective contract; runtime observations follow separately",
            "training_provenance_qualified": False, "scientific_execution_qualified": False})
        journal.event("authentication_complete")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        if args.verify_only:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        loader = CapturedSources(captured, Path(args.freeze).absolute().parent)
        sys.meta_path.insert(0, loader)
        with contextlib.redirect_stdout(sys.stderr):
            from llm_intervention_service_runtime import ServiceRuntime, fixture_inputs, verify_cpu_tokens
            if args.fixture and not args.verify_only:
                raise ValueError("CPU fixture arguments are not service requests")
            fixtures = fixture_inputs(args.fixture, Path(args.freeze).absolute().parent,
                                      read_pinned, decode) if args.verify_only else None
            runtime_observer = ServiceRuntime(freeze, captured, loader, Path(__file__).absolute(), globals(),
                args.runtime_stage, args.runtime_stage_sha256, journal, read_pinned, decode)
            runtime_observer.import_cpu_closure()
            if args.verify_only:
                verify_cpu_tokens(freeze, fixtures, journal)
                runtime_observer.observe("after-cpu-tokenization")
                journal.save("terminal.json", {"status": "cpu_preflight_complete",
                    "freeze_sha256": args.freeze_sha256, "runtime_stage_sha256": args.runtime_stage_sha256,
                    "model_loaded": False, "adapter_deserialized": False, "explicit_cuda_api_called": False,
                    "new_training_updates": 0, "environment_responses": 0,
                    "service_qualified": False, "scientific_execution_qualified": False})
                return
            from llm_intervention_persistent_policy import serve_policy
        def initialize():
            decide = initialize_qwen(freeze, adapter_raw, journal, arm_deadline)
            runtime_observer.observe("after-model-load")
            return decide
        serve_policy(initialize, identity=identity(freeze), max_requests=freeze["limits"]["max_requests"])
        runtime_observer.observe("after-requests")
        journal.save("terminal.json", {"status": "complete", "freeze_sha256": args.freeze_sha256,
            "runtime_stage_sha256": args.runtime_stage_sha256,
            "generation_count": len(list(journal.output.glob("generation-*.json"))),
            "new_training_updates": 0, "environment_responses": 0,
            "scheduler_cleanup_and_scientific_acceptance_qualified": False})
    except BaseException as exc:
        journal.save("failure.json", {"type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(), "new_training_updates": 0,
            "environment_responses": 0, "retry_allowed": False,
            "available_raw_generation_records": sorted(p.name for p in journal.output.glob("raw-generation-*.json")),
            "accepted_generation_records": sorted(p.name for p in journal.output.glob("generation-*.json"))})
        # Best-effort observation cannot replace the primary failure or qualify
        # execution. The external supervisor still owns its absolute deadline.
        if runtime_observer is not None:
            try:
                runtime_observer.observe("failure", require_imports=False)
            except BaseException as observation_error:
                journal.save("runtime-failure-observation-error.json", {
                    "type": type(observation_error).__name__, "message": str(observation_error)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        if loader is not None and loader in sys.meta_path:
            sys.meta_path.remove(loader)


if __name__ == "__main__":
    main()
