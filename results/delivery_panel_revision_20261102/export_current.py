"""Exclusive publication-only export using previously authenticated local fonts."""
from pathlib import Path
import sys,json,hashlib,shutil,subprocess,time,resource
root=Path('/Users/pat/code/ACE')
base=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
name=sys.argv[1]
assert name.isdigit() and len(name)==2
out=base/f'delivery-paper-panel-layout-20261102-{name}'
fonts=json.loads((root/'results/delivery_panel_revision_20261102/export02/font_assets.json').read_text())
source=base/'delivery-paper-panel-layout-20261102-02'
payload=[]
for f in fonts['files']:
 p=source/f['name'];raw=p.read_bytes()
 assert len(raw)==f['bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256']
 payload.append((f['name'],raw))
out.mkdir(mode=0o700)
for n,b in payload:(out/n).write_bytes(b)
shutil.copyfile(root/'paper/aistats_ace_2027/paper.tex',out/'paper.tex')
t=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
r=subprocess.run(['/opt/homebrew/bin/tectonic','--only-cached','--untrusted','--keep-logs','--keep-intermediates','--outdir',str(out),str(out/'paper.tex')],cwd=out,capture_output=True,text=True)
after=resource.getrusage(resource.RUSAGE_CHILDREN)
(out/'compile.stdout.log').write_text(r.stdout);(out/'compile.stderr.log').write_text(r.stderr)
d={'returncode':r.returncode,'elapsed_seconds':time.monotonic()-t,'child_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,'peak_child_rss_bytes':after.ru_maxrss,'source_sha256':hashlib.sha256((out/'paper.tex').read_bytes()).hexdigest(),'scope':'publication layout compilation only, not scientific compute','fonts_inventory_sha256':hashlib.sha256((root/'results/delivery_panel_revision_20261102/export02/font_assets.json').read_bytes()).hexdigest()}
(out/'compile.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({'path':str(out),**d}));print(r.stderr[-2000:]);sys.exit(r.returncode)
