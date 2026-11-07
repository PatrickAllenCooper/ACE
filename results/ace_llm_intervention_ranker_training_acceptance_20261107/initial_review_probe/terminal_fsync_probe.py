"""One new standard-library persistence probe; no trainer/model/test imports.

Execute only the unchanged AST bodies of canonical and _save. Reproduce the
terminal-write exception followed by a failure write, as in fit's except branch.
This is not a fit invocation and does not establish checkpoint/model behavior.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
started = time.monotonic()
cpu_before = resource.getrusage(resource.RUSAGE_SELF)
sources = {
    "canonical": REPO / "scripts/research/llm_intervention_contract.py",
    "_save": REPO / "scripts/research/llm_intervention_ranker_training.py",
}
expected_engine = "5ba9d41c9df4c35e81d77131ae735f92b452eefce6bdcd27d45b5b1e7f1d63c5"
raw = {name: path.read_bytes() for name, path in sources.items()}
assert hashlib.sha256(raw["_save"]).hexdigest() == expected_engine
nodes = []
for name, content in raw.items():
    nodes.append(next(n for n in ast.parse(content).body
                      if isinstance(n, ast.FunctionDef) and n.name == name))

calls = []
def fsync_once_failed(fd):
    calls.append(fd)
    if len(calls) == 1:
        raise OSError("focused injected terminal fsync failure")
    return os.fsync(fd)

namespace = {"json": json, "os": SimpleNamespace(fsync=fsync_once_failed)}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "<pinned persistence functions>", "exec"), namespace)
out = ROOT / "terminal_fsync_once"
out.mkdir(exist_ok=False)
terminal = {"status": "complete", "scope": "isolated persistence helper only; no fit"}
raised = None
try:
    namespace["_save"](out / "terminal.json", terminal)
except OSError as exc:
    raised = str(exc)
    namespace["_save"](out / "failure.json", {
        "status": "failed", "message": str(exc),
        "scope": "synthetic failure write matching fit's exception disposition",
    })
observed_terminal = json.loads((out / "terminal.json").read_text())
observed_failure = json.loads((out / "failure.json").read_text())
assert raised is not None
assert observed_terminal["status"] == "complete"
assert observed_failure["status"] == "failed"
assert len(calls) == 2
cpu_after = resource.getrusage(resource.RUSAGE_SELF)
result = {
    "utc": datetime.now(timezone.utc).isoformat(),
    "command": [sys.executable, "-B", str(Path(__file__).resolve())],
    "cwd": str(Path.cwd()),
    "scope": "AST-extracted persistence functions only; no fit, Torch, models or old tests",
    "source_sha256": {str(sources[n]): hashlib.sha256(b).hexdigest() for n, b in raw.items()},
    "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "injected_failure": raised,
    "fsync_calls": len(calls),
    "terminal_status_after_exception": observed_terminal["status"],
    "failure_status_after_exception": observed_failure["status"],
    "both_files_parse_as_complete_json": True,
    "elapsed_seconds_since_probe_start": time.monotonic() - started,
    "cpu_seconds_since_probe_start": (
        cpu_after.ru_utime + cpu_after.ru_stime - cpu_before.ru_utime - cpu_before.ru_stime),
    "measurement_scope": "probe body only; excludes Python startup/imports",
    "artifacts": {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in sorted(out.iterdir())},
}
with (ROOT / "terminal_fsync_result.json").open("x") as f:
    json.dump(result, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps(result, sort_keys=True))
