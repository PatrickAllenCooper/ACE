from pathlib import Path
import ast,hashlib,shutil,difflib
R=Path('/tmp/ace-downstream-compact-adapter-6r23szeq');I=R/'implementation'
S=R/'draft-before-final-boundary-delta';S.mkdir();shutil.copytree(I,S/'implementation')
for stage in ('gpu','finalizer'):
 p=I/stage/'launch_once.py';s=p.read_text()
 line="        b.configure_packet_storage(ctx,local,pins,packet)\n"
 data="        cpu_data=b.admit_cpu_bytes(ctx,Path(packet['cpu_remote_custody']))\n" if stage=='gpu' else "        gpu_data=b.admit_gpu_bytes(ctx,packet['gpu_terminal_custody'],packet['gpu_terminal_mirror'])\n"
 assert s.count(line+data)==1;s=s.replace(line+data,'')
 anchor="               'committed independently pinned packet review required')\n" if stage=='gpu' else "               'committed independently pinned packet review required')\n"
 assert s.count(anchor)==1
 s=s.replace(anchor,anchor+line+data);p.write_text(s)
 p=I/stage/'pull_terminal.py';s=p.read_text()
 s=s.replace('def read_metadata(path,expected=None):','def read_metadata(path,expected=None,maximum=32*1024**2):')
 s=s.replace("need(meta['bytes']<=32*1024**2,'bounded coordinator metadata required')", "need(type(maximum) is int and 0<=maximum<=128*1024**2 and meta['bytes']<=maximum,'bounded coordinator metadata required')")
 if stage=='gpu':
  s=s.replace('data=read_metadata(LAUNCH/n,h)',"data=read_metadata(LAUNCH/n,h,128*1024**2 if n=='storage-seal.json' else 32*1024**2)")
  s=s.replace('read_metadata(LAUNCH_EXT/n,h)',"read_metadata(LAUNCH_EXT/n,h,128*1024**2 if n=='storage-seal.json' else 32*1024**2)")
 else:
  s=s.replace("read_metadata(LAUNCH/b.metadata_name(n),pins[b.metadata_name(n)]) for n in b.METADATA", "read_metadata(LAUNCH/b.metadata_name(n),pins[b.metadata_name(n)],128*1024**2 if n=='gpu-custody-inventory' else 32*1024**2) for n in b.METADATA")
 ast.parse(s);p.write_text(s)
 for filename in ('build_packet.py','launch_once.py','pull_terminal.py'):
  old=R/'predecessor'/stage/('collector/pull_terminal.py' if stage=='finalizer' and filename=='pull_terminal.py' else filename)
  new=I/stage/filename
  (I/stage/filename.replace('.py','.diff')).write_text(''.join(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile='predecessor/'+filename,tofile=filename)))
print('new local boundary delta authored; no coordinator run')
