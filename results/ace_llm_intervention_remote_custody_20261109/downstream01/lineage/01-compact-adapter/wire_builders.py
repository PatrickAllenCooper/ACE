from pathlib import Path
import ast,hashlib,difflib
BASE=Path(__file__).resolve().parent;IM=BASE/'implementation';adapter=(IM/'compact_inputs.py').read_bytes();PIN=hashlib.sha256(adapter).hexdigest()
COMMON='''
# Explicit outer storage adapter; never added to frozen scientific child inputs.
COMPACT_ADAPTER_SHA='__PIN__'
CURC_HELPER='/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'

def compact_adapter():
    import types
    path=Path(__file__).resolve().with_name('compact_inputs.py')
    source=read(path,COMPACT_ADAPTER_SHA)
    mod=types.ModuleType('captured_compact_inputs');mod.__file__=str(path)
    exec(compile(source,str(path),'exec'),vars(mod))
    return mod


def storage_members():
    c=compact_adapter()
    return ('storage-config.json',*c.STORAGE_FILES.values(),'compact_inputs.py')


def configure_storage(ctx,supplied,capsule,mirror):
    c=compact_adapter()
    gate=c.bind(ctx,supplied,predecessor='__PREDECESSOR__',capsule=capsule,mirror=mirror,
        expected_roots=__ROOTS__,committed=committed,helper_path=CURC_HELPER)
    ctx['storage_supplied']=supplied
    return gate


def recheck_storage(ctx):
    compact_adapter().recheck_capsules(ctx['compact_inputs'])


def storage_packet_files(ctx):
    return {**ctx['storage_supplied'],'compact_inputs.py':read(Path(__file__).resolve().with_name('compact_inputs.py'),COMPACT_ADAPTER_SHA)}


def storage_packet_binding(ctx):
    gate=ctx['compact_inputs']
    return dict(mode='curc_projects_full_backup_with_compact_local_evidence',
        predecessor_remote_custody_sha256=gate.receipt_sha256,
        storage_config_sha256=sha(ctx['storage_supplied']['storage-config.json']),
        backend_sha256=gate.helper_sha256,compact_adapter_sha256=COMPACT_ADAPTER_SHA,
        original_ancestor_references=gate.ancestry,complete_local_raw_copy=False)


def configure_packet_storage(ctx,local,pins,packet):
    c=compact_adapter();supplied=c.stored_config(local,pins)
    gate=configure_storage(ctx,supplied,__CAPSULE__,__MIRROR__)
    need(packet['storage_binding']==storage_packet_binding(ctx),'independent compact storage packet binding differs')
    return gate

'''

def replace(s,old,new):
 assert s.count(old)==1,(old,s.count(old));return s.replace(old,new)
for stage,pred,roots,cap,mir in (
 ('gpu','cpu',"dict(launch=CPU_ROOT,training=TRAIN_ROOT,cpu=TOKEN_ROOT,**{'failed-token01':FAILED_TOKEN_ROOT},common=COLLECTION_ROOT,failed01=load_cpu_history().FAILED_COLLECTION_ROOT,**{'namespace.json':REMOTE_REGISTRY,'original-namespace.json':ORIGINAL_REMOTE_REGISTRY})","Path(packet['cpu_remote_custody']).parent","packet['cpu_terminal_mirror']"),
 ('finalizer','gpu',"{**RAW_ROOTS,'namespace.json':REMOTE_REGISTRY,'original-namespace.json':ORIGINAL_REMOTE_REGISTRY}","packet['gpu_terminal_custody']","packet['gpu_terminal_mirror']")):
 old=(BASE/'predecessor'/stage/'build_packet.py').read_text();s=old
 helper=COMMON.replace('__PIN__',PIN).replace('__PREDECESSOR__',pred).replace('__ROOTS__',roots).replace('__CAPSULE__',cap).replace('__MIRROR__',mir)
 s=s.replace('\ndef main():','\n'+helper+'\ndef main():')
 if stage=='gpu':
  s=replace(s,"    observed=inventory(raw_root);need(observed==files,'complete original CPU raw custody differs')", "    gate=ctx['compact_inputs'];need(Path(raw_root).resolve(strict=True) in gate.local_roots,'bound selected CPU view required')\n    observed=gate.full['files'];need(observed==files,'complete remote CPU raw inventory differs')")
  s=replace(s,"        data[n]=read(Path(raw_root)/'launch/inputs'/n,h,256*1024**2)","        data[n]=gate.capture('launch/inputs/'+n,h,256*1024**2)")
  s=replace(s,"    need(inventory(raw_root)==observed,'complete original custody changed during capture')","    recheck_storage(ctx)")
  s=replace(s,"    ap.add_argument('--cpu-remote-custody',required=True,help='complete accepted CPU terminal remote/ tree, opaque bytes only')", "    ap.add_argument('--cpu-remote-custody',required=True,help='selected accepted CPU terminal remote/ view, opaque bytes only')\n    ap.add_argument('--cpu-terminal-mirror',required=True,help='independent selected CPU terminal capsule')")
  s=replace(s,"        custody=Path(args.cpu_remote_custody).resolve(strict=True);cpu_data=admit_cpu_bytes(ctx,custody)","        custody=Path(args.cpu_remote_custody).resolve(strict=True)\n        storage=compact_adapter().capture_config(args.storage_config,args.storage_config_sha256)\n        configure_storage(ctx,storage,custody.parent,args.cpu_terminal_mirror)\n        cpu_data=admit_cpu_bytes(ctx,custody)")
  s=replace(s,"            paths=stage['paths'],cpu_remote_custody=str(custody),prior_jobs=46", "            paths=stage['paths'],cpu_remote_custody=str(custody),cpu_terminal_mirror=str(Path(args.cpu_terminal_mirror).resolve(strict=True)),\n            storage_binding=storage_packet_binding(ctx),prior_jobs=46")
  s=replace(s,"            need(inventory(custody)==ctx['observations']['cpu']['files'],'original CPU custody changed during build')", "            recheck_storage(ctx)")
 else:
  a=s.index('    copies=[]\n',s.index('def admit_gpu_bytes'))
  z=s.index("    ctx['upstream_rosters']=accepted_rosters(ctx)",a)
  s=s[:a]+"    gate=ctx['compact_inputs']\n    need(gate.capsules==(capsule,mirror),'bound compact capsule identities differ')\n    need(sha(gate.supplied['storage-seal.json'])==pins['gpu-custody-inventory'] and\n         sha(gate.supplied['storage-after.json'])==pins['gpu-inventory-after'],'original independently pinned selected custody metadata differs')\n    recheck_storage(ctx)\n"+s[z:]
  s=replace(s,"        data[n]=read(capsule/'remote/launch/inputs'/n,h,256*1024**2)","        data[n]=gate.capture('launch/inputs/'+n,h,256*1024**2)")
  s=replace(s,"        gpu_data=admit_gpu_bytes(ctx,args.gpu_terminal_custody,args.gpu_terminal_mirror)","        storage=compact_adapter().capture_config(args.storage_config,args.storage_config_sha256)\n        configure_storage(ctx,storage,args.gpu_terminal_custody,args.gpu_terminal_mirror)\n        gpu_data=admit_gpu_bytes(ctx,args.gpu_terminal_custody,args.gpu_terminal_mirror)")
  s=replace(s,"            gpu_terminal_mirror=str(ctx['custody_paths'][1]),prior_jobs=47", "            gpu_terminal_mirror=str(ctx['custody_paths'][1]),storage_binding=storage_packet_binding(ctx),prior_jobs=47")
  s=replace(s,"            for root in ctx['custody_paths']:\n                now=tree(root);del now['files']['custody-inventory.json']\n                need(now==ctx['metadata']['gpu-custody-inventory']['copies'][str(root)], 'original full custody changed during capture')", "            recheck_storage(ctx)")
 s=replace(s,"    args=ap.parse_args()", "    ap.add_argument('--storage-config',required=True)\n    ap.add_argument('--storage-config-sha256',required=True)\n    args=ap.parse_args()")
 anchor="        for n,v in supplied.items():files[metadata_name(n)]=v"
 s=replace(s,anchor,anchor+"\n        files.update(storage_packet_files(ctx))")
 ast.parse(s);(IM/stage/'build_packet.py').write_text(s);(IM/stage/'compact_inputs.py').write_bytes(adapter)
 (IM/stage/'build_packet.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='predecessor/build_packet.py',tofile='build_packet.py')))
 old=(BASE/'predecessor'/stage/'launch_once.py').read_text();s=old
 if stage=='gpu':
  anchor="        cpu_data=b.admit_cpu_bytes(ctx,Path(packet['cpu_remote_custody']))"
 else:anchor="        gpu_data=b.admit_gpu_bytes(ctx,packet['gpu_terminal_custody'],packet['gpu_terminal_mirror'])"
 s=replace(s,anchor,"        b.configure_packet_storage(ctx,local,pins,packet)\n"+anchor)
 # Existing remote/source/lock/submission gates remain unchanged; bind outer storage in both intents.
 anchor="intent=dict(utc=b.utc(),packet_commit=a.packet_commit,source_commit=b.SOURCE_COMMIT,"
 s=replace(s,anchor,anchor+"storage_binding=b.storage_packet_binding(ctx),")
 # Preserve explicit storage use evidence outside immutable packet files.
 anchor="        save('controller.json',dict(pid=os.getpid(),utc=b.utc(),held=[str(b.BASE/n) for n in"
 s=replace(s,anchor,"        save('storage-launch-use.json',dict(binding=b.storage_packet_binding(ctx),operations=str(ctx['compact_inputs'].staging),acceptance=False))\n"+anchor)
 ast.parse(s);(IM/stage/'launch_once.py').write_text(s)
 (IM/stage/'launch_once.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='predecessor/launch_once.py',tofile='launch_once.py')))
print(PIN)
