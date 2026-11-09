from pathlib import Path
import hashlib,ast
R=Path('/tmp/ace-downstream-compact-d1d2-1d_57cvu');I=R/'implementation'
s=(I/'compact_inputs.py').read_text()
s=s.replace("    evidence.record('binding-complete',details={'predecessor':predecessor,'receipt_sha256':receipt_pin,'ancestry':chain})\n",'')
(I/'compact_inputs.py').write_text(s);pin=hashlib.sha256(s.encode()).hexdigest()
helper='''def start_storage_operations(ctx, root, reference_roots, identity):
    """Register durable caller evidence before the first archive verification."""
    c=compact_adapter();roots=tuple(Path(p).resolve(strict=True) for p in reference_roots)
    def publish(name,value):
        errors=[]
        for target in roots:
            try:write(target/name,raw(value))
            except BaseException as exc:errors.append(type(exc).__name__+': '+str(exc))
        need(not errors,'storage operation reference publication failed: '+repr(errors))
    identity={**identity,'coordinator_sha256':sha(read(Path(__file__))),
              'source_commit':SOURCE_COMMIT,'reference_roots':[str(p) for p in roots]}
    ctx['storage_operation_evidence']=c.OperationEvidence(root,identity,publish)
    return ctx['storage_operation_evidence']


'''
for stage in ('gpu','finalizer'):
 p=I/stage/'build_packet.py';t=p.read_text().replace("COMPACT_ADAPTER_SHA='634979bb79542c0adc970f464cd445588a866815f11387d43ff64c664c12acb8'",'COMPACT_ADAPTER_SHA='+repr(pin))
 t=t.replace('def configure_storage(ctx,supplied,capsule,mirror):',helper+'def configure_storage(ctx,supplied,capsule,mirror):')
 t=t.replace("    gate=c.bind(ctx,supplied,predecessor=", "    gate=ctx['storage_operation_evidence'].run('binding',lambda:c.bind(ctx,supplied,predecessor=")
 t=t.replace('committed=committed,helper_path=CURC_HELPER)\n    ctx[', 'committed=committed,helper_path=CURC_HELPER))\n    ctx[')
 t=t.replace("    return {**ctx['storage_supplied'],'compact_inputs.py':", "    return {**ctx['storage_supplied'],\n        'storage-preparation-intent.json':raw(ctx['storage_operation_evidence'].intent_reference),\n        'storage-preparation-reference.json':raw(ctx['storage_operation_evidence'].last_reference),\n        'compact_inputs.py':")
 anchor="        storage=compact_adapter().capture_config(args.storage_config,args.storage_config_sha256)\n"
 insert='''        need(not any(os.path.lexists(p) for p in (LOCAL,EXT)),'exclusive new launch destinations required before storage use')
        preparations=(EXT.with_name(EXT.name+'-storage-build-preparation'),LOCAL.parent/'storage-build-preparation01')
        need(not any(os.path.lexists(p) for p in preparations),'existing storage preparation; never retry')
        for path in preparations:path.mkdir(mode=0o700,exist_ok=False)
        start_storage_operations(ctx,preparations[0]/'records',preparations,
            dict(phase='build',storage_config_sha256=args.storage_config_sha256,
                 intended_packet_roots=[str(EXT),str(LOCAL)]))
'''
 assert t.count(anchor)==1;t=t.replace(anchor,anchor+insert)
 # Original packet stdout supplements its pins with the final durable operation reference.
 t=t.replace("print(raw(dict(packet_sha256=sha(files['packet.json']),", "print(raw(dict(storage_operations=ctx['storage_operation_evidence'].last_reference,packet_sha256=sha(files['packet.json']),")
 ast.parse(t);p.write_text(t);(I/stage/'compact_inputs.py').write_text(s)
 p=I/stage/'launch_once.py';t=p.read_text()
 t=t.replace("('controller.json','launcher_failure.json','build-failure.json','submission.intent.json','submission.result.json')", "('controller.json','launcher_failure.json','build-failure.json','submission.intent.json','submission.result.json','storage-use.intent.json')")
 anchor='        b.configure_packet_storage(ctx,local,pins,packet)\n'
 insert="""        started=True
        b.start_storage_operations(ctx,b.EXT.with_name(b.EXT.name+'-storage-launch-operations'),(b.EXT,b.LOCAL),
            dict(phase='launch',packet_sha256=a.packet_sha256,packet_commit=a.packet_commit))
"""
 assert t.count(anchor)==1;t=t.replace(anchor,insert+anchor)
 t=t.replace("operations=str(ctx['compact_inputs'].staging),acceptance=False", "operation_reference=ctx['storage_operation_evidence'].last_reference,acceptance=False")
 ast.parse(t);p.write_text(t)
print(pin)
