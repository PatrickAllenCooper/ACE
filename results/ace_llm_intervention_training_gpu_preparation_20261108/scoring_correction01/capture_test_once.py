from pathlib import Path
import datetime, hashlib, json, os, resource, subprocess, sys, time
root = Path(__file__).resolve().parent
mirror = Path("/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-training-gpu-scoring-correction-20261108-01")
invocation = json.loads((root/"invocation.json").read_text())
# Reserve all result paths exclusively before starting the single test child.
handles = {(out, name): (out/name).open("xb") for out in (root, mirror)
           for name in ("stdout.txt", "stderr.txt", "metrics.json")}
started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
started = time.monotonic()
result = subprocess.run(invocation["argv"], cwd=invocation["cwd"],
                        env={**os.environ, **invocation["environment_overrides"]}, capture_output=True)
elapsed = time.monotonic() - started
usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
metrics = {"started_utc": started_utc,
           "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "elapsed_seconds": elapsed,
           "child_user_cpu_seconds": usage_after.ru_utime - usage_before.ru_utime,
           "child_system_cpu_seconds": usage_after.ru_stime - usage_before.ru_stime,
           "child_max_rss_bytes": usage_after.ru_maxrss if sys.platform == "darwin" else usage_after.ru_maxrss*1024,
           "resource_measurement": "resource.getrusage(RUSAGE_CHILDREN); exactly one test child",
           "exit_code": result.returncode, "stdout_bytes": len(result.stdout), "stderr_bytes": len(result.stderr),
           "test_runs": 1, "new_test_methods": 6, "prior_closed_tests_run": 0,
           "gpu_seconds": 0, "remote_operations": 0, "scientific_responses": 0,
           "preparation_overhead_cpu_seconds": None,
           "preparation_overhead_note": "Only the test subprocess is metered; editing, reads and evidence capture overhead are not measured."}
for (out, name), handle in handles.items():
    handle.write(result.stdout if name == "stdout.txt" else result.stderr if name == "stderr.txt"
                 else (json.dumps(metrics, indent=2, sort_keys=True)+"\n").encode())
    handle.close()
print(result.stdout.decode(), end="")
print(result.stderr.decode(), end="", file=sys.stderr)
print(json.dumps(metrics, indent=2, sort_keys=True))
raise SystemExit(result.returncode)
