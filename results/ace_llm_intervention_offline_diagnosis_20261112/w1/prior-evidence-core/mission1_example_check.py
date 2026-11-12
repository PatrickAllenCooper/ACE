"""Read-only: for Mission-1 LLM-arm selected 'lenient' rows whose raw generation has no legal DO command,
what command was actually executed? Also per-step presence of the modal command among the K=4 probes."""
import json, os, sys, collections, re, yaml
Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
ROOT = sys.argv[1]
arms = {"peter-2026-10-A": "A", "peter-2026-10-B": "B", "peter-2026-10-Bp": "Bp", "peter-2026-10-C": "C"}
runs = []
for slot in ["slot0", "slot1", "c0", "c1"]:
    for l in open(os.path.join(ROOT, slot, "ace_results", "runs.local.ndjson")):
        l = l.strip()
        if not l: continue
        r = json.loads(l)
        if r.get("batch_id") in arms:
            runs.append((arms[r["batch_id"]], r["seed"], os.path.join(ROOT, slot, "ace_results", r["run_id"])))
runs.sort()
VALID = ["leo_satellite_count","leo_dynamic_satellite_count","leo_altitude_km","leo_antenna_power","meo_satellite_count","meo_altitude_km","meo_antenna_power","jamming_percent"]
do_re = re.compile(r"(?i)DO\s+([a-zA-Z0-9_]+)\s*=\s*(-?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)")
CONST = ("leo_dynamic_satellite_count", 200.0)
res = collections.defaultdict(collections.Counter)
steps_with_const = collections.Counter(); steps_total = collections.Counter(); const_per_step = collections.defaultdict(collections.Counter)
first_feature = None
for arm, seed, d in runs:
    # per-step presence among probes
    bystep = collections.defaultdict(list)
    for l in open(os.path.join(d, "observations.ndjson")):
        r = json.loads(l)
        if r["role"] == "lookahead":
            bystep[r["step"]].append((r["intervention_target"], r["intervention_value"]))
    for s, lst in bystep.items():
        steps_total[arm] += 1
        c = sum(1 for x in lst if x == CONST)
        const_per_step[arm][c] += 1
        if c: steps_with_const[arm] += 1
    if arm == "C": continue
    text = open(os.path.join(d, "report.yaml")).read()
    i = text.find("\naction_log:"); rest = text[i+1:]
    m = re.search(r"\n(?=[A-Za-z_][A-Za-z0-9_]*:)", rest[len("action_log:"):])
    block = rest if m is None else rest[: len("action_log:") + m.start() + 1]
    al = yaml.load(block, Loader=Loader)["action_log"]
    for a in al:
        if a.get("parse_status") != "lenient": continue
        raw = a.get("llm_raw_output") or ""
        mm = do_re.search(raw)
        executed = (a.get("intervention_target"), a.get("intervention_value"))
        if mm is None:
            cls = "gen_has_no_parsable_DO"
        elif mm.group(1) in VALID:
            cls = "gen_first_DO_legal"
        else:
            cls = "gen_first_DO_illegal_target"
        res[(arm, cls)]["n"] += 1
        res[(arm, cls)]["executed_is_modal_cmd"] += int(executed == CONST)
        if cls == "gen_first_DO_legal":
            res[(arm, cls)]["executed_target_equals_generated_target"] += int(executed[0] == mm.group(1))
            res[(arm, cls)]["generated_target_is_leo_dynamic"] += int(mm.group(1) == CONST[0])
for k in sorted(res): print(k, dict(res[k]))
print()
for arm in ["A", "B", "Bp", "C"]:
    print(arm, "steps", steps_total[arm], "steps with modal cmd among K probes", steps_with_const[arm],
          round(steps_with_const[arm] / steps_total[arm], 4), "count-per-step hist", dict(const_per_step[arm]))
