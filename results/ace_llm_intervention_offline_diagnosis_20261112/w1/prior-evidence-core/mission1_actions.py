"""Read-only tabulation of Mission-1 (2026-10 peter-baseline) proposal behaviour.
Reads saved run bundles only. No model, no simulator."""
import json, os, sys, collections, re
ROOT = sys.argv[1]
arms = {"peter-2026-10-A": "A", "peter-2026-10-B": "B", "peter-2026-10-Bp": "Bp", "peter-2026-10-C": "C"}
runs = []
for slot in ["slot0", "slot1", "c0", "c1"]:
    p = os.path.join(ROOT, slot, "ace_results", "runs.local.ndjson")
    for l in open(p):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        if r.get("batch_id") in arms:
            runs.append((arms[r["batch_id"]], r["seed"], os.path.join(ROOT, slot, "ace_results", r["run_id"])))
runs.sort()
print("runs", collections.Counter(a for a, _, _ in runs))
agg_probe = collections.defaultdict(collections.Counter)   # arm -> (target,value) over all lookahead probes
agg_sel = collections.defaultdict(collections.Counter)     # arm -> (target,value) over selected
agg_parse = collections.defaultdict(collections.Counter)   # arm -> parse_status over action_log rows
per_run = []
ps_re = re.compile(r"^  parse_status: (\S+)\s*$")
tgt_re = re.compile(r"^  intervention_target: (\S+)\s*$")
val_re = re.compile(r"^  intervention_value: (\S+)\s*$")
for arm, seed, d in runs:
    probes = collections.Counter(); sel = collections.Counter(); nlook = 0; nsel = 0
    for l in open(os.path.join(d, "observations.ndjson")):
        r = json.loads(l)
        if r["role"] != "lookahead":
            continue
        key = (r["intervention_target"], r["intervention_value"])
        probes[key] += 1; nlook += 1
        if r["selected"]:
            sel[key] += 1; nsel += 1
    agg_probe[arm].update(probes); agg_sel[arm].update(sel)
    # action log parse statuses (text scan of report.yaml; action_log is the last top-level block we need)
    inlog = False; statuses = collections.Counter(); al = []
    cur = {}
    for l in open(os.path.join(d, "report.yaml")):
        if l.startswith("action_log:"):
            inlog = True; continue
        if inlog:
            if l and not l.startswith((" ", "-")):
                inlog = False; continue
            if l.startswith("- step:"):
                if cur: al.append(cur)
                cur = {}
            m = ps_re.match(l)
            if m: cur["ps"] = m.group(1)
            m = tgt_re.match(l)
            if m: cur["t"] = m.group(1)
            m = val_re.match(l)
            if m: cur["v"] = m.group(1)
    if cur: al.append(cur)
    for a in al:
        statuses[a.get("ps")] += 1
    agg_parse[arm].update(statuses)
    top = probes.most_common(1)[0] if probes else None
    tops = sel.most_common(1)[0] if sel else None
    per_run.append((arm, seed, nlook, nsel, top, tops, dict(statuses), len(al)))
for row in per_run:
    print(row)
print()
for arm in ["A", "B", "Bp", "C"]:
    tot = sum(agg_probe[arm].values()); tots = sum(agg_sel[arm].values())
    print("ARM", arm, "probes", tot, "selected", tots)
    print("  top probe (target,value):", [(k, v, round(v / tot, 4)) for k, v in agg_probe[arm].most_common(5)])
    print("  top selected:", [(k, v, round(v / tots, 4)) for k, v in agg_sel[arm].most_common(5)])
    tcount = collections.Counter()
    for (t, v), c in agg_probe[arm].items(): tcount[t] += c
    print("  probe target share:", [(k, v, round(v / tot, 4)) for k, v in tcount.most_common()])
    print("  distinct (target,value) among probes:", len(agg_probe[arm]))
    ps = agg_parse[arm]; pt = sum(ps.values())
    print("  action_log parse_status:", [(k, v, round(v / pt, 4)) for k, v in ps.most_common()], "n", pt)
