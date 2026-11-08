"""Bounded read-only seed reconciliation. Emits hashes/seed provenance, never scores.

Only explicitly supplied research roots are visited; symlinks are not followed.
Exact proposed-seed mentions are conservative review candidates, not automatic
proof that a prior history exists. JSON/CSV seed fields and source declarations
are supplementary; unknown binary or oversized registries remain uncovered.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time

EXTENSIONS={'.json','.jsonl','.csv','.tsv','.yaml','.yml','.toml','.py','.sh','.md','.txt'}
SKIP_DIRS={'.git','.venv','.venv311','node_modules','__pycache__','.agents','.codex','frontend','dashboard','runner_delivery_registry_reconciliation_20261004'}
MAX_BYTES=2*1024*1024

def scan(roots,seeds,seconds=90,registry_only=False):
    deadline=time.monotonic()+seconds
    result={'roots':roots,'files':[],'proposed_mentions':[],'known_seed_values':[],
            'unread':[],'symlinks_not_followed':[],'max_file_bytes':MAX_BYTES,'seconds_cap':seconds,'registry_only':registry_only}
    known=set(); proposed=set(seeds)
    pattern=re.compile(r'(?<![0-9])('+ '|'.join(map(str,seeds))+r')(?![0-9])')
    def harvest(value,key=''):
        if isinstance(value,dict):
            for k,v in value.items():harvest(v,str(k))
        elif isinstance(value,list):
            for v in value:harvest(v,key)
        elif re.search(r'(^|_)seeds?($|_)',key.lower()) and type(value) is int:
            known.add(value)
    for raw in roots:
        root=Path(raw)
        if not root.is_dir():result['unread'].append({'path':raw,'reason':'root_missing'});continue
        for base,dirs,names in os.walk(root,followlinks=False):
            kept=[]
            for name in sorted(dirs):
                p=Path(base)/name
                if p.is_symlink():result['symlinks_not_followed'].append(str(p))
                elif name not in SKIP_DIRS:kept.append(name)
            dirs[:]=kept
            for name in sorted(names):
                p=Path(base)/name
                path_hits=sorted({int(m.group()) for m in pattern.finditer(str(p))})
                if path_hits:result['proposed_mentions'].append({'path':str(p),'seeds':path_hits,'kind':'path'})
                if p.suffix.lower() not in EXTENSIONS:continue
                if registry_only and not re.search(r'seed|meta|manifest|receipt|complete|config|registration|budget|aggregate|summary|ledger|runbook',name,re.I):continue
                if time.monotonic()>deadline:
                    result['unread'].append({'path':str(p),'reason':'deadline_remaining_scope_unvisited'})
                    result['known_seed_values']=sorted(known);return result
                if p.is_symlink():result['symlinks_not_followed'].append(str(p));continue
                try:
                    if p.stat().st_size>MAX_BYTES:
                        result['unread'].append({'path':str(p),'reason':'oversized'});continue
                    data=p.read_bytes(); text=data.decode('utf-8')
                    result['files'].append({'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
                    hits=sorted({int(m.group()) for m in pattern.finditer(text)})
                    if hits:result['proposed_mentions'].append({'path':str(p),'seeds':hits})
                    if p.suffix=='.json':
                        try:harvest(json.loads(text))
                        except json.JSONDecodeError:result['unread'].append({'path':str(p),'reason':'json_parse_failure_seed_fields'})
                    elif p.suffix in {'.csv','.tsv'}:
                        reader=csv.DictReader(io.StringIO(text),delimiter='\t' if p.suffix=='.tsv' else ',')
                        for row in reader:
                            for key,value in row.items():
                                if key and re.search(r'(^|_)seeds?($|_)',key.lower()) and value and re.fullmatch(r'\d+',value.strip()):known.add(int(value))
                    for match in re.finditer(r'\b(?:SEEDS?|seeds?)\s*[:=]\s*[\[\("\x27]*([0-9][0-9, \t]*)',text):
                        known.update(map(int,re.findall(r'\d+',match.group(1))))
                except (OSError,UnicodeError) as error:result['unread'].append({'path':str(p),'reason':type(error).__name__})
    result['known_seed_values']=sorted(known)
    result['structured_or_declared_intersection']=sorted(proposed & known)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--roots',nargs='+',required=True);parser.add_argument('--seeds',nargs='+',type=int,required=True);parser.add_argument('--seconds',type=int,default=90)
    parser.add_argument('--registry-only',action='store_true')
    args=parser.parse_args();print(json.dumps(scan(args.roots,args.seeds,args.seconds,args.registry_only),indent=2))
