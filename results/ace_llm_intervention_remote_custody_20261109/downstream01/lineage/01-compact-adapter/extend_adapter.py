from pathlib import Path
import shutil
root=Path('/tmp/ace-downstream-compact-adapter-6r23szeq')
snapshot=root/'draft-before-collector-wiring';snapshot.mkdir()
shutil.copytree(root/'implementation',snapshot/'implementation')
p=root/'implementation/compact_inputs.py';s=p.read_text()
s=s.replace("path=Path(path);data=path.read_bytes()\n    need(len(data)<=128*1024 and sha(data)==expected_sha256, 'independent small storage config pin required')", "path=Path(path)\n    data=read_exact(path,{'bytes':path.stat().st_size,'sha256':expected_sha256},128*1024)")
s=s.replace("backend_spec=config['members']['backend']", "backend_spec=config['members']['backend']\n    for spec in (backend_spec,config['members']['backend_review']):\n        relative(spec['repository_path'])")
s+='''

def collect_selected(*, backend, helper_sha256, before, roots, backup_root,
                     external, local, helper_path, save, observe, compare, copy_file):
    """Transport-only wrapper. Full immutable archive plus two selected views.

    Caller retains both locks, independent source/Git/adoption bindings, original
    input/resource/accounting gates and exception/failure publication. This never
    copies a predecessor capsule or claims that a selected view is a full tree.
    """
    import shutil
    external,local=Path(external),Path(local)
    need(external.is_dir() and local.is_dir() and external!=local and
         external not in local.parents and local not in external.parents,
         'two exclusive disjoint capsule roots required')
    need(not any(os.path.lexists(p/'remote') or os.path.lexists(p/'backup-operations')
                 for p in (external,local)), 'selected/operation destinations must be absent; no retry')
    expected={'files':before['files'],'directories':sorted(before['directories'])}
    inventory_shape(expected)
    selected,_=backend.selected_inventory(backend.normalized(before))
    inventory_shape(selected)
    selected_bytes=sum(v['bytes'] for v in selected['files'].values())
    # Conservatively charge both copies on each device, including bounded logs
    # and observation/capsule metadata. Existing references need no new storage.
    required=2*selected_bytes+32*len(canonical(before))+512*1024**2
    capacity={}
    for dest in (external,local):
        fs=os.statvfs(dest);available=fs.f_bavail*fs.f_frsize
        need(available>required,'insufficient aggregate selected-copy capacity')
        capacity[str(dest)]={'available_bytes':available,'required_increment_bytes':required,
                             'device':dest.stat().st_dev,'selected_bytes_per_copy':selected_bytes}
    save('local-capacity.json',capacity)
    receipt=backend.collect_backup(observation=before,roots=roots,backup_root=backup_root,
        local_root=external/'remote',helper_path=helper_path,log_dir=external/'backup-operations')
    need(receipt['schema']=='ace-remote-custody-v1' and receipt['full_backup_verified'] is True and
         receipt['acceptance'] is False and receipt['retry_allowed'] is False and
         receipt['full_local_copy'] is False and receipt['backup_kind']=='ustar' and
         receipt['helper_sha256']==helper_sha256 and receipt['roots']==roots and
         receipt['backup_root']==backup_root and receipt['local_root']==str(external/'remote') and
         receipt['selection_policy']=='current-launch-text-and-namespaces-v1' and
         receipt['full_inventory']==expected and receipt['local_inventory']==selected,
         'complete archive/selected receipt binding differs')
    omitted={n:v for n,v in expected['files'].items() if n not in selected['files']}
    need(all(expected['files'].get(n)==v for n,v in selected['files'].items()) and
         receipt['omitted_files']==omitted and receipt['omitted_bytes']==sum(v['bytes'] for v in omitted.values()) and
         selected_inventory(external/'remote')==selected,'exact selected/omitted partition differs')
    save('remote-custody.json',receipt)
    after=observe('inventory_after_transport.json')
    compare(before,after)  # Unchanged stage-specific scheduler/journal/root gates.
    for sub in ('remote','backup-operations'):
        shutil.copytree(external/sub,local/sub,copy_function=copy_file)
        need(selected_inventory(external/sub)==selected_inventory(local/sub),'independent selected/operation mirror differs')
        for name in selected_inventory(external/sub)['files']:
            need(not os.path.samestat((external/sub/name).stat(),(local/sub/name).stat()),'selected mirror shares inode')
    return expected,after,receipt
'''
p.write_text(s)
# Correct the one new root expression; use exact already-authenticated predecessor constant.
f=root/'wire_builders.py';t=f.read_text().replace('failed01=FAILED_COLLECTION_ROOT','failed01=load_cpu_history().FAILED_COLLECTION_ROOT');f.write_text(t)
