"""Reproduce only the role-specific reader delta in this exclusive preparation."""
from pathlib import Path
import hashlib,json,difflib
ROOT=Path(__file__).resolve().parent
sha=lambda data:hashlib.sha256(data).hexdigest()
ROLES=('cpu-inventory','cpu-disposition','cpu-acceptance','gpu-inventory','gpu-inventory-after','gpu-custody-inventory','gpu-disposition','gpu-acceptance')
MEMBERS=tuple(n+'.json' for n in ROLES)+('storage-receipt.json','storage-seal.json','storage-after.json','inputs/cpu-acceptance.json','inputs/gpu-acceptance.json')
def change(text,old,new,count=1):
 if text.count(old)!=count:raise ValueError('unexpected replacement count '+repr(old))
 return text.replace(old,new)
role_block="FULL_MAP_METADATA = frozenset("+repr(ROLES)+")\n\ndef metadata_maximum(name):\n    return 128*1024**2 if name in FULL_MAP_METADATA else 32*1024**2\n\n\n"
member_block="FULL_MAP_PACKET_MEMBERS = frozenset("+repr(MEMBERS)+")\n\ndef packet_member_maximum(name):\n    return 128*1024**2 if name in FULL_MAP_PACKET_MEMBERS else 32*1024**2\n\n\n"
for stage in ('gpu','finalizer'):
 p=ROOT/'implementation'/stage/'compact_inputs.py';s=p.read_text()
 s=change(s,"maximum=128*1024**2 if key=='seal' else 32*1024**2","maximum=128*1024**2 if key in ('receipt','seal','after') else 32*1024**2")
 s=change(s,"maximum=128*1024**2 if name=='storage-seal.json' else 32*1024**2","maximum=128*1024**2 if name in ('storage-receipt.json','storage-seal.json','storage-after.json') else 32*1024**2")
 p.write_text(s)
 newadapter=sha(p.read_bytes())
 p=ROOT/'implementation'/stage/'build_packet.py';s=p.read_text()
 s=change(s,'def need(ok,message):',role_block+'def need(ok,message):')
 old="read(getattr(args,n.replace('-','_')),getattr(args,n.replace('-','_')+'_sha256')"+(",256*1024**2)" if stage=='finalizer' else ")")
 s=change(s,old,"read(getattr(args,n.replace('-','_')),getattr(args,n.replace('-','_')+'_sha256'),metadata_maximum(n))")
 s=change(s,"COMPACT_ADAPTER_SHA='6dc4fe9b95cb0452c801a9be3f0a2102de9bc6ea76e777725b823501e82dda46'","COMPACT_ADAPTER_SHA='"+newadapter+"'")
 p.write_text(s)
 newbuilder=sha(p.read_bytes())
 p=ROOT/'implementation'/stage/'launch_once.py';s=p.read_text()
 old="pins[b.metadata_name(name)]"+(",256*1024**2)" if stage=='finalizer' else ")")+" for name in b.METADATA}"
 s=change(s,old,"pins[b.metadata_name(name)],b.metadata_maximum(name)) for name in b.METADATA}")
 p.write_text(s)
 p=ROOT/'implementation'/stage/'pull_terminal.py';s=p.read_text()
 if stage=='gpu':
  s=change(s,'def read_metadata(path,expected=None,maximum=32*1024**2):',member_block+'def read_metadata(path,expected=None,maximum=32*1024**2):')
  s=change(s,"128*1024**2 if n=='storage-seal.json' else 32*1024**2","packet_member_maximum(n)",2)
  s=change(s,"read_metadata(LAUNCH/b.metadata_name(n),pins[b.metadata_name(n)]) for n in b.METADATA}","read_metadata(LAUNCH/b.metadata_name(n),pins[b.metadata_name(n)],b.metadata_maximum(n)) for n in b.METADATA}")
  s=change(s,"BUILDER_SHA='e41b787fe0ede1191408b0941e7f9d806b9d4d38fcd0cf5718aebc2bfa16f8e1'","BUILDER_SHA='"+newbuilder+"'")
 else:
  s=change(s,"128*1024**2 if n=='gpu-custody-inventory' else 32*1024**2","b.metadata_maximum(n)")
 p.write_text(s)
# CLI is a successor, with exact corresponding new path/source pins and named bounds.
p=ROOT/'implementation/cli/known-inputs.json';known=json.loads(p.read_bytes())
known['builder_directory']=str(ROOT/'implementation/gpu')
known['builder_sources']={n:sha((ROOT/'implementation/gpu'/n).read_bytes()) for n in known['builder_sources']}
known['full_map_metadata']=list(ROLES)
p.write_text(json.dumps(known,sort_keys=True,indent=2)+'\n')
newknown=sha(p.read_bytes())
p=ROOT/'implementation/cli/assemble_gpu_cli.py';s=p.read_text()
s=change(s,"KNOWN_SHA256 = '07b77331eaf27bd70d0cbb2d03a837ea6c3181261eb593932701f9a9477524a2'","KNOWN_SHA256 = '"+newknown+"'")
s=change(s,'def need(ok, message):',role_block+'def need(ok, message):')
s=change(s,"    for entry in metadata.values():\n        capture(entry['path'], entry['sha256'])", "    need(set(known['full_map_metadata']) == set(FULL_MAP_METADATA), 'known role bounds differ')\n    for name, entry in metadata.items():\n        capture(entry['path'], entry['sha256'], metadata_maximum(name))")
s=change(s,"pending[name] = capture(**dict(path=entry['path'], expected=entry['sha256']))", "pending[name] = capture(entry['path'], entry['sha256'], metadata_maximum(name))")
s=change(s,"128*MIB if name == 'seal' else 32*MIB", "128*MIB if name in ('receipt', 'seal', 'after') else 32*MIB")
p.write_text(s)
pins={}
for p in sorted((ROOT/'implementation').rglob('*')):
 if not p.is_file():continue
 rel=p.relative_to(ROOT/'implementation');before=(ROOT/'predecessor'/rel).read_bytes();after=p.read_bytes()
 pins[rel.as_posix()]={'before_sha256':sha(before),'after_sha256':sha(after),'changed':before!=after}
 if before!=after:
  d=ROOT/'diffs'/(rel.as_posix().replace('/','__')+'.diff')
  d.write_text(''.join(difflib.unified_diff(before.decode().splitlines(True),after.decode().splitlines(True),fromfile='predecessor/'+rel.as_posix(),tofile='successor/'+rel.as_posix())))
(ROOT/'source-pins.json').write_text(json.dumps(pins,sort_keys=True,indent=2)+'\n')
(ROOT/'role-bound-map.json').write_text(json.dumps({'metadata_128_mib':list(ROLES),'storage_128_mib':['receipt','seal','after'],'gpu_collector_packet_members_128_mib':list(MEMBERS),'ordinary_default_bytes':32*1024**2,'full_map_maximum_bytes':128*1024**2,'storage_config_maximum_bytes':128*1024,'opaque_fetch_maximum_bytes_unchanged':256*1024**2,'backend_and_batch02_unchanged':True,'metadata_finalizer_admission_note':'Replaces old broad256MiB initial metadata reads with named128MiB/ordinary32MiB; later existing256MiB exact-byte mirror/opaque-input comparisons remain unchanged.','unchanged_lateral_reads':'Finalizer collector all-input loop already streams fingerprints and uses byte readback without a32MiB reader; no lateral32MiB cap there. Existing seal reference reads remain128MiB.'},sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v['after_sha256'] for k,v in pins.items() if v['changed']},indent=2))
