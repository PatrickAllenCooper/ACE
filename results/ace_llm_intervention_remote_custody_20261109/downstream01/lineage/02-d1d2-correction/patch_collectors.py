from pathlib import Path
import ast,hashlib
R=Path('/tmp/ace-downstream-compact-d1d2-1d_57cvu');I=R/'implementation'
def indent(text,n=4):return ''.join(' '*n+line if line.strip() else line for line in text.splitlines(True))
# GPU: retain its original two-lock try/finally, move all transport-dependent work inside it.
p=I/'gpu/pull_terminal.py';s=p.read_text()
start=s.index("    original_files=ctx['observations']['cpu']['files']")
var=s.index('    locks=[];claimed=[];',start)
source=s.index("    source='\\n'.join",var)
trypos=s.index('    try:\n        for name in (\'ace-delivery-controller.lock\'',source)
saveintent=s.index("        save('collector-intent.json'",trypos)
pre=s[:start];gates=s[start:var];functions=s[var:source];transport_defs=s[source:trypos]
lockclaim=s[trypos:saveintent]
line="            write(dest/'remote_custody.py',helper_source);write(dest/'storage-backend-review.md',helper_review)\n"
assert line in lockclaim;lockclaim=lockclaim.replace(line,'')
lockclaim=lockclaim.replace("        need(all(not os.path.lexists(p) for p in (LOCAL,EXT))", "        need(all(not os.path.lexists(p) for p in (LOCAL,EXT,EXT.with_name(EXT.name+'-storage-collector-operations')))")
# Parentheses unchanged: replacement retains comma/message portion of need.
start_evidence="""        b.start_storage_operations(ctx,EXT.with_name(EXT.name+'-storage-collector-operations'),(EXT,LOCAL),
            dict(phase='collector',job_id=a.job_id,packet_sha256=a.packet_sha256,
                 collector_sha256=a.collector_sha256,packet_commit=a.packet_commit))
"""
helper="        for dest in claimed:\n            write(dest/'remote_custody.py',helper_source);write(dest/'storage-backend-review.md',helper_review)\n"
s=pre+functions+lockclaim+start_evidence+indent(gates)+indent(transport_defs)+helper+s[saveintent:]
# Initial failures can precede the nested copy helper; it is only needed once transport definitions have completed.
s=s.replace("BUILDER_SHA='8fe8d6affe97bdd3f6cde7b84dd263bbec8b90e764158c50668bacb252e0b5fd'",'BUILDER_SHA='+repr(hashlib.sha256((I/'gpu/build_packet.py').read_bytes()).hexdigest()))
ast.parse(s);p.write_text(s)
# Finalizer: reuse its original lock scope, existing destination guard and failure handler.
p=I/'finalizer/pull_terminal.py';s=p.read_text()
gatestart=s.index('    gate=b.configure_packet_storage(')
varspos=s.index('    claimed=[];started=',gatestart)
observedef=s.index('    def observe(name):',varspos)
lockpos=s.index('    with b.coordinator_locks():',observedef)
intentpos=s.index("            save('collector-intent.json'",lockpos)
prefix=s[:gatestart];gates=s[gatestart:varspos];savedefs=s[varspos:observedef];observes=s[observedef:lockpos]
lockclaim=s[lockpos:intentpos]
assert line.replace('            ','                ',1) in lockclaim
lockclaim=lockclaim.replace(line.replace('            ','                ',1),'')
lockclaim=lockclaim.replace("not any(os.path.lexists(p) for p in (EXT,LOCAL))", "not any(os.path.lexists(p) for p in (EXT,LOCAL,EXT.with_name(EXT.name+'-storage-collector-operations')))")
evidence="""            b.start_storage_operations(ctx,EXT.with_name(EXT.name+'-storage-collector-operations'),(EXT,LOCAL),
                dict(phase='collector',job_id=a.job_id,packet_sha256=a.packet_sha256,
                     collector_sha256=a.collector_sha256,packet_commit=a.packet_commit))
"""
helper="            for dest in claimed:\n                write(dest/'remote_custody.py',helper_source);write(dest/'storage-backend-review.md',helper_review)\n"
s=prefix+savedefs+lockclaim+evidence+indent(gates,8)+indent(observes,8)+helper+s[intentpos:]
ast.parse(s);p.write_text(s)
print('collector locks and attempt claims now precede every storage operation')
