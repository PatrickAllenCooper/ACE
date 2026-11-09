from pathlib import Path
import hashlib,ast,difflib,shutil
R=Path('/tmp/ace-downstream-compact-adapter-6r23szeq');I=R/'implementation'
def replace(s,a,b):
 assert s.count(a)==1,(s.count(a),a[:120]);return s.replace(a,b)
def cutfuncs(s,names):
 tree=ast.parse(s);lines=s.splitlines(keepends=True)
 for n in sorted([n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Assign)) and (getattr(n,'name',None) in names or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)))],key=lambda x:x.lineno,reverse=True):
  del lines[n.lineno-1:n.end_lineno]
 return ''.join(lines)
for stage,oldpath in [('gpu','gpu/pull_terminal.py'),('finalizer','finalizer/collector/pull_terminal.py')]:
 old=(R/'predecessor'/oldpath).read_text();s=old
 s=cutfuncs(s,{'STREAM_HELPER_SHA','STREAM_REVIEW_SHA','load_stream_helper','stream_transfer','stream_request','authenticate_stream_files','committed_sidecar','mirror_attempt'})
 if stage=='gpu':
  s=replace(s,"BUILDER_SHA='140374c5e561cc3078b5a5aa79c0b75094d4631d55d554a182348ecebe997ff9'","BUILDER_SHA="+repr(hashlib.sha256((I/stage/'build_packet.py').read_bytes()).hexdigest()))
  s=replace(s,"'stream-helper-repository-path','stream-helper-review-repository-path'","'backup-root'")
  start=s.index('    # Full accepted CPU capsule is independently pinned');end=s.index('    # This existing pure byte validator',start)
  s=s[:start]+'''    # Bound compact CPU capsule; original full archive and failed ancestors remain explicit.
    gate=b.configure_packet_storage(ctx,LAUNCH,pins,packet)
    need(intent['storage_binding']==packet['storage_binding'],'original storage intent differs')
    cpu_custody=Path(a.cpu_terminal_custody).resolve(strict=True)
    need(cpu_custody==gate.capsules[0] and a.cpu_custody_inventory_sha256==pins['storage-seal.json'],
         'independent original selected CPU capsule/seal differs')
    cpu_custody_before=inventory(cpu_custody)
''' + s[end:]
  start=s.index('    # Helper and independent review are standalone');end=s.index('    recovery_bindings=',start)
  s=s[:start]+'''    # Backend and review are independently captured/Git-bound outer storage members.
    backend=gate.backend;helper_source=gate.supplied['remote_custody.py']
    helper_review=gate.supplied['storage-backend-review.md']
''' +s[end:]
  s=replace(s,"write(dest/'stream_custody.py',helper_source);write(dest/'stream_custody_review.md',helper_review)","write(dest/'remote_custody.py',helper_source);write(dest/'storage-backend-review.md',helper_review)")
  s=replace(s,"            stream_helper_sha256=STREAM_HELPER_SHA,stream_helper_review_sha256=STREAM_REVIEW_SHA,\n            stream_helper_repository_path=a.stream_helper_repository_path,stream_helper_review_repository_path=a.stream_helper_review_repository_path,", "            backup_root=a.backup_root,backup_helper_sha256=gate.helper_sha256,\n            storage_binding=packet['storage_binding'],complete_local_raw_copy=False,")
  start=s.index('        shutil.copytree(cpu_custody,');end=s.index('        local_claim_present=',start)
  s=s[:start]+'''        save('predecessor-reference.json',dict(path=str(cpu_custody),
            custody_inventory_sha256=a.cpu_custody_inventory_sha256,
            disposition_sha256=pins['cpu-disposition.json'],raw_inventory_sha256=pins['cpu-inventory.json'],
            complete_existing_capsule_authenticated=True,recursive_copy_created=False,original_preserved=True,
            custody_mode='curc_projects_full_backup_with_compact_local_evidence'))
        coordinator={}
        for label,original in (('local-launch',LAUNCH),('external-launch',LAUNCH_EXT)):
            view=inventory(original);coordinator[label]=view
            save('coordinator/'+label+'-reference.json',dict(path=str(original),inventory=view,
                recursive_copy_created=False,original_preserved=True))
''' +s[end:]
  start=s.index('        helper=load_stream_helper');end=s.index('        errors=[]',start)
  s=s[:start]+'''        all_roots=dict(launch=ROOT,**UPSTREAM_ROOTS,**{'namespace.json':REMOTE_NAMESPACE,'original-namespace.json':OLD_REMOTE_NAMESPACE})
        roots={label:path for label,path in all_roots.items() if label in before['directories'] or label in before['files']}
        actual,after,backup_receipt=b.compact_adapter().collect_selected(backend=backend,
            helper_sha256=gate.helper_sha256,before=before,roots=roots,backup_root=a.backup_root,
            external=EXT,local=LOCAL,helper_path=HELPER,save=save,observe=observe,
            compare=compare_observations,copy_file=copy_file)
        b.recheck_storage(ctx)
        for label,original in (('local-launch',LAUNCH),('external-launch',LAUNCH_EXT)):
            need(inventory(original)==coordinator[label],'original coordinator history changed')
''' +s[end:]
  s=replace(s,"            hardlink_custody='All original names copied as independent local bytes; original alias metadata retained.',", "            custody_mode='curc_projects_full_backup_with_compact_local_evidence',\n            remote_custody_receipt_sha256=sha(raw(backup_receipt)),backup_helper_sha256=gate.helper_sha256,\n            complete_local_raw_copy=False,local_raw_files=len(backup_receipt['local_inventory']['files']),\n            local_raw_bytes=sum(v['bytes'] for v in backup_receipt['local_inventory']['files'].values()),\n            hardlink_custody='All original names verified in CURC USTAR archive. Local evidence is explicitly selected; original alias metadata retained.',")
  s=replace(s,"for sub in ('remote','coordinator','upstream-cpu-custody','stream-attempt'):","for sub in ('remote','coordinator','backup-operations'):")
 else:
  s=replace(s,"'stream-helper-repository-path','stream-review-repository-path'","'backup-root'")
  s=replace(s,"    b.admit_gpu_bytes(ctx,packet['gpu_terminal_custody'],packet['gpu_terminal_mirror'])", "    gate=b.configure_packet_storage(ctx,LAUNCH,pins,packet)\n    need(intent['storage_binding']==packet['storage_binding'],'original storage intent differs')\n    b.admit_gpu_bytes(ctx,packet['gpu_terminal_custody'],packet['gpu_terminal_mirror'])") if s.count("    b.admit_gpu_bytes(ctx,packet['gpu_terminal_custody'],packet['gpu_terminal_mirror'])")==1 else s.replace("    ctx=b.authenticate(supplied);m=ctx['metadata'];base=m['gpu-baseline']", "    ctx=b.authenticate(supplied);m=ctx['metadata'];base=m['gpu-baseline']\n    gate=b.configure_packet_storage(ctx,LAUNCH,pins,packet)\n    need(intent['storage_binding']==packet['storage_binding'],'original storage intent differs')")
  s=replace(s,"    helper_source,helper_review=authenticate_stream_files(a.packet_commit,a.stream_helper_repository_path,a.stream_review_repository_path)", "    backend=gate.backend;helper_source=gate.supplied['remote_custody.py'];helper_review=gate.supplied['storage-backend-review.md']")
  s=replace(s,"write(dest/'stream_custody.py',helper_source);write(dest/'stream_custody_review.md',helper_review)","write(dest/'remote_custody.py',helper_source);write(dest/'storage-backend-review.md',helper_review)")
  s=replace(s,"                roots=ROOTS,acceptance=False,retry_allowed=False))", "                roots=ROOTS,backup_root=a.backup_root,backup_helper_sha256=gate.helper_sha256,\n                storage_binding=packet['storage_binding'],complete_local_raw_copy=False,acceptance=False,retry_allowed=False))")
  start=s.index('            copies={}');end=s.index('            local_claims={}',start)
  s=s[:start]+'''            save('predecessor-reference.json',dict(path=str(ctx['custody_paths'][0]),
                custody_inventory_sha256=ctx['pins']['gpu-custody-inventory'],
                disposition_sha256=ctx['pins']['gpu-disposition'],raw_inventory_sha256=ctx['pins']['gpu-inventory'],
                complete_existing_capsule_authenticated=True,recursive_copy_created=False,original_preserved=True,
                custody_mode='curc_projects_full_backup_with_compact_local_evidence'))
            copies={}
            for label,original in [('coordinator-local',LAUNCH),('coordinator-external',LAUNCH_EXT)]:
                view=inventory(original);copies[label]=(Path(original),view)
                save(label+'-reference.json',dict(path=str(original),inventory=view,
                    recursive_copy_created=False,original_preserved=True))
''' +s[end:]
  start=s.index('            helper=load_stream_helper');end=s.index('            # Recheck both independent',start)
  s=s[:start]+'''            all_roots={**ROOTS,'namespace.json':REMOTE_NAMESPACE,'original-namespace.json':OLD_REMOTE_NAMESPACE}
            roots={label:path for label,path in all_roots.items() if label in before['directories'] or label in before['files']}
            actual,after,backup_receipt=b.compact_adapter().collect_selected(backend=backend,
                helper_sha256=gate.helper_sha256,before=before,roots=roots,backup_root=a.backup_root,
                external=EXT,local=LOCAL,helper_path=HELPER,save=save,observe=observe,
                compare=compare_observations,copy_file=copy_file)
            for label,(original,view) in copies.items():
                need(inventory(original)==view,'original coordinator changed')
''' +s[end:]
  s=replace(s,"                raw_files=len(actual['files']),raw_bytes=sum(v['bytes'] for v in actual['files'].values()),", "                raw_files=len(actual['files']),raw_bytes=sum(v['bytes'] for v in actual['files'].values()),\n                custody_mode='curc_projects_full_backup_with_compact_local_evidence',\n                remote_custody_receipt_sha256=sha(raw(backup_receipt)),backup_helper_sha256=gate.helper_sha256,\n                complete_local_raw_copy=False,local_raw_files=len(backup_receipt['local_inventory']['files']),\n                local_raw_bytes=sum(v['bytes'] for v in backup_receipt['local_inventory']['files'].values()),")
 ast.parse(s)
 (I/stage/'pull_terminal.py').write_text(s)
 (I/stage/'pull_terminal.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='predecessor/pull_terminal.py',tofile='pull_terminal.py')))
# Required original history sidecar, captured unchanged.
src=Path('/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_gpu_preparation_20261108/token02-history01/authoring/cpu_history_contract.py')
assert hashlib.sha256(src.read_bytes()).hexdigest()=='3893a30ec2e652dac78529602564a4a295ddf9258461530ec7c74557df809025'
shutil.copyfile(src,I/'gpu/cpu_history_contract.py')
print('wrote two collector derivatives; no collector executed')
