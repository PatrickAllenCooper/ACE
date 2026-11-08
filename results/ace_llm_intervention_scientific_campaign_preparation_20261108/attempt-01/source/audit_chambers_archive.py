"""Inspect action metadata only; no outcome statistics or model fitting."""
import argparse,csv,hashlib,io,json,pathlib,zipfile

def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
 b=a.archive.read_bytes();assert hashlib.md5(b).hexdigest()=='cc49b95d85410e0b5ea3bcd1479428e3'
 cells=[]
 with zipfile.ZipFile(io.BytesIO(b)) as z:
  for name in sorted(z.namelist()):
   if not name.endswith('.csv'):continue
   raw=z.read(name); reader=csv.DictReader(io.StringIO(raw.decode()));fields=reader.fieldnames
   keys=['timestamp','counter','pol_1','pol_2','red','green','blue','flag','intervention']
   rows=[{k:r[k] for k in keys} for r in reader]
   pairs=[(float(r['pol_1']),float(r['pol_2'])) for r in rows]
   times=[float(r['timestamp']) for r in rows]
   cells.append({'file':name,'sha256':hashlib.sha256(raw).hexdigest(),'rows':len(rows),'columns':fields,'metadata_missing':sum(v=='' for r in rows for v in r.values()),'unique_pairs':len(set(pairs)),'angle_ranges':[[min(x[i] for x in pairs),max(x[i] for x in pairs)] for i in range(2)],'strictly_increasing_timestamps':all(y>x for x,y in zip(times,times[1:])),'rgb_values':{k:sorted(set(r[k] for r in rows)) for k in ['red','green','blue']},'flags':sorted(set(r['flag'] for r in rows)),'intervention_values':sorted(set(r['intervention'] for r in rows))})
 a.output.mkdir(parents=True,exist_ok=False)
 result={'archive_sha256':hashlib.sha256(b).hexdigest(),'archive_bytes':len(b),'published_md5_verified':True,'cells':cells,'total_rows':sum(c['rows'] for c in cells),'outcome_statistics_computed':False,'model_calls':0,'new_experimental_queries':0}
 data=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode();(a.output/'audit.json').write_bytes(data)
 (a.output/'complete.json').write_text(json.dumps({'audit_sha256':hashlib.sha256(data).hexdigest(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='cells'}));print([(c['file'],c['rows'],c['unique_pairs'],c['metadata_missing'],c['strictly_increasing_timestamps']) for c in cells])
if __name__=='__main__':main()
