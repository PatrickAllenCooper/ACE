"""Read-only: parse action_log of Mission-1 LLM-arm report.yaml files (saved bundles)."""
import json, os, sys, collections, re, yaml
try:
    Loader = yaml.CSafeLoader
except AttributeError:
    Loader = yaml.SafeLoader
ROOT = sys.argv[1]
arms = {"peter-2026-10-A": "A", "peter-2026-10-B": "B", "peter-2026-10-Bp": "Bp", "peter-2026-10-C": "C"}
runs = []
for slot in ["slot0", "slot1", "c0", "c1"]:
    p = os.path.join(ROOT, slot, "ace_results", "runs.local.ndjson")
    for l in open(p):
        l = l.strip()
        if not l: continue
        r = json.loads(l)
        if r.get("batch_id") in arms:
            runs.append((arms[r["batch_id"]], r["seed"], os.path.join(ROOT, slot, "ace_results", r["run_id"])))
runs.sort()
VALID = ["leo_satellite_count","leo_dynamic_satellite_count","leo_altitude_km","leo_antenna_power","meo_satellite_count","meo_altitude_km","meo_antenna_power","jamming_percent"]
do_re = re.compile(r"(?i)\bDO\s+([A-Za-z_][A-Za-z0-9_]*)\s*=")
agg = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
out = {}
for arm, seed, d in runs:
    if arm == "C": continue
    text = open(os.path.join(d, "report.yaml")).read()
    i = text.find("\naction_log:")
    assert i >= 0
    # take the action_log block up to the next top-level key
    rest = text[i+1:]
    m = re.search(r"\n(?=[A-Za-z_][A-Za-z0-9_]*:)", rest[len("action_log:"):])
    block = rest if m is None else rest[: len("action_log:") + m.start() + 1]
    al = yaml.load(block, Loader=Loader)["action_log"]
    ps = collections.Counter(a.get("parse_status") for a in al)
    tv = collections.Counter((a.get("intervention_target"), a.get("intervention_value")) for a in al)
    # raw output analysis
    raw_first = collections.Counter()
    raw_empty = 0
    for a in al:
        raw = a.get("llm_raw_output") or ""
        if not raw.strip():
            raw_empty += 1; continue
        mm = do_re.search(raw)
        if mm is None: raw_first["no_DO_in_generation"] += 1
        elif mm.group(1) in VALID: raw_first["first_DO_names_legal_target"] += 1
        else: raw_first["first_DO_names_illegal_target"] += 1
    # by parse status, share of the constant command
    const = ("leo_dynamic_satellite_count", 200.0)
    by_ps_const = collections.Counter(); by_ps = collections.Counter()
    for a in al:
        by_ps[a.get("parse_status")] += 1
        if (a.get("intervention_target"), a.get("intervention_value")) == const:
            by_ps_const[a.get("parse_status")] += 1
    agg[arm]["ps"].update(ps); agg[arm]["tv"].update(tv); agg[arm]["raw"].update(raw_first)
    agg[arm]["raw"]["empty_raw_output"] += raw_empty
    agg[arm]["by_ps"].update(by_ps); agg[arm]["by_ps_const"].update(by_ps_const)
    agg[arm]["n"]["steps"] += len(al)
    print(arm, seed, "steps", len(al), dict(ps), "const_selected", tv.get(const, 0), "raw", dict(raw_first), "empty", raw_empty)
print()
for arm in ["A", "B", "Bp"]:
    n = agg[arm]["n"]["steps"]
    print("ARM", arm, "action_log steps", n)
    print("  parse_status:", [(k, v, round(v/n, 4)) for k, v in agg[arm]["ps"].most_common()])
    print("  top selected (target,value):", [(k, v, round(v/n, 4)) for k, v in agg[arm]["tv"].most_common(4)])
    print("  raw-output classes:", dict(agg[arm]["raw"]))
    print("  const share by parse_status:", {k: (agg[arm]["by_ps_const"][k], v) for k, v in agg[arm]["by_ps"].items()})
