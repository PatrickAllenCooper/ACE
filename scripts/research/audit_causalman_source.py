#!/usr/bin/env python3
"""Read-only pinned-source audit. Does not import or execute upstream code."""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request

REVISION = "17529dad5ec8b8c691494c617b9af4533aa44bf8"
REPOSITORY = "boschresearch/CausalMan"
FILES = (
    "README.md", "LICENSE", "pyproject.toml", "requirements.txt",
    "causalman/causalman.py", "causalman/fcm.py", "causalman/sample_batch.py",
    "causalman/utils/sampling.py", "causalman/utils/graph.py",
    "causalman/utils/data.py", "causalman/utils/serialization.py",
    "causalman/dataset_objects/causalman_micro/causalman_micro_1_batch_info.csv",
)


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def blob_hash(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.cache.mkdir(parents=True, exist_ok=True)
    tree_path = args.cache / "upstream_tree.json"
    if not tree_path.exists():
        tree_path.write_bytes(download(
            f"https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1"))
    tree = json.loads(tree_path.read_bytes())
    if tree.get("truncated"):
        raise ValueError("incomplete upstream inventory")
    blobs = {item["path"]: item for item in tree["tree"] if item["type"] == "blob"}
    inventory = []
    sources = {}
    for name in FILES:
        path = args.cache / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(download(
                f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{name}"))
        data = path.read_bytes()
        expected = blobs[name]
        if blob_hash(data) != expected["sha"] or len(data) != expected["size"]:
            raise ValueError(f"upstream blob mismatch: {name}")
        inventory.append({"path": name, "bytes": len(data), "git_blob_sha1": expected["sha"],
                          "sha256": hashlib.sha256(data).hexdigest()})
        sources[name] = data.decode()

    groups = collections.defaultdict(lambda: {"files": 0, "bytes": 0})
    for name, item in blobs.items():
        if name.startswith("causalman/dataset_objects/") and len(name.split("/")) > 3:
            group = groups[name.split("/")[2]]
            group["files"] += 1
            group["bytes"] += item["size"]

    module = ast.parse(sources["causalman/causalman.py"])
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "CausalMan")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    sample_returns = [n for n in ast.walk(methods["sample"]) if isinstance(n, ast.Return)]
    returned = sample_returns[-1].value
    assert isinstance(returned, ast.Tuple) and len(returned.elts) == 6
    apply = methods["apply_interventions"]
    assert any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
               and not n.value.keys for n in apply.body)
    rows = list(csv.DictReader(io.StringIO(sources[FILES[-1]])))
    assert len({row["subbatch_ID_unique"] for row in rows}) == len(rows)
    checks = {
        "fixed_source_seed_path": "seed=random_state" in sources["causalman/utils/graph.py"],
        "loads_path_specific_graphs": 'f"dag_level_1_{path_idx}.pkl"' in sources["causalman/sample_batch.py"],
        "seed_changes_sampling": "random_state=random_state_seed" in sources["causalman/sample_batch.py"],
        "one_micro_product": 'return ["causalman_micro_1"]' in sources["causalman/causalman.py"],
        "observability_action_dependent": "node not in intervention_targets" in sources["causalman/causalman.py"],
        "last_graph_returned": "dag_level_2," in sources["causalman/utils/sampling.py"],
        "string_intervention_eval": "value = eval(value)" in sources["causalman/fcm.py"],
    }
    if not all(checks.values()):
        raise ValueError(f"source assumptions changed: {checks}")
    report = {
        "repository": REPOSITORY, "upstream_revision": REVISION,
        "audit_type": "static_source_only", "source_checks": checks,
        "dataset_inventory": dict(sorted(groups.items())),
        "sample_return_order": [ast.unparse(x) for x in returned.elts],
        "micro_configured_subbatches": len(rows),
        "micro_sum_integer_batch_size_run": sum(int(float(row["batch_size_run"])) for row in rows),
        "row_count_status": "Configured count only; actual generation uses saved path dataframe lengths and is unmeasured.",
        "independence": "Seed varies random sampling and row selection, not the bundled structural equations. Micro is one product with batch/path configurations, not independent seeded SCMs.",
        "runtime_status": "Not executed; pair targets, public columns, peak RAM and latency still need a bounded runtime gate.",
        "decision": "Do not launch a multi-seed confirmation through the high-level mixture API. Audit one fixed batch/path using the unchanged lower-level sampler before a custody smoke.",
        "upstream_tree_sha256": hashlib.sha256(tree_path.read_bytes()).hexdigest(),
        "simulator_queries": 0, "model_calls": 0, "gpu_allocations": 0,
    }
    args.output.mkdir(parents=True)
    for name, content in (("source_inventory.json", inventory), ("audit.json", report)):
        (args.output / name).write_text(json.dumps(content, sort_keys=True, indent=2) + "\n")
    receipt = {"upstream_revision": REVISION, "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "files": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                         for name in ("source_inventory.json", "audit.json")},
               "source_blobs_verified": len(inventory), "simulator_queries": 0, "model_calls": 0}
    (args.output / "complete.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"blobs_verified": len(inventory), "micro": groups["causalman_micro"],
                      "runtime": "not executed"}))


if __name__ == "__main__":
    main()
