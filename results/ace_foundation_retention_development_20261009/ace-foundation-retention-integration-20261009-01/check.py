import hashlib,itertools,json,sys,time,resource
from pathlib import Path
sys.path.insert(0,"/Users/pat/code/ACE/scripts/research")
import test_foundation_retention_runner as fixture
import foundation_retention_pilot as w
import summarize_foundation_retention as r
root=Path(__file__).parent/"stub-study"
root.mkdir()
fixture.setup_worker()
plan=[{"seed":223456,"variant":v,"method":m} for v,m in itertools.product(w.VARIANTS,w.METHODS)]
w.write(root/"plan.json",{"mode":"fixture","cells":plan})
w.run(root,Path("/unused"),"fixture")
complete=json.loads((root/"complete.json").read_text())
terminal={"mode":"fixture","status":"complete","reason":"exited","exit_code":0,"error":None,"planned_cells":36,"cells":complete["cells"],**dict.fromkeys(r.RESOURCE_KEYS,0)}
freeze={"schema":"ace-retention-freeze-v1","mode":"fixture","sources":{r.REPORTER_KEY:r.sha(r.__file__)}}
summary=r.verify(root,freeze,terminal)
assert len(summary["cells"])==36
assert all(x["status"]=="complete" for x in summary["cells"])
assert len(summary["comparisons"])==96 and len(summary["direct_comparisons"])==72
assert len(summary["local_harm"])==64
assert summary["response_accounting"]["training"]["validated_returned"]==160
assert summary["response_accounting"]["private"]["validated_returned"]==3072
w.write(root/"verified_stub_summary.json",summary)
print(json.dumps({"passed":True,"cells":36,"endpoints":108,"main_summaries":96,"direct_summaries":72,"local_harm":64,"scope":"stub learners; fabricated terminal/freezemetadata; real worker serialization and independent reporter; not actual runtime qualification"}))
