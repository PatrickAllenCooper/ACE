import os,pathlib,stat,signal,json,datetime
signal.alarm(40)
root=pathlib.Path('/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108_01')
count=total=selected=selected_count=0;largest=[]
for d,dirs,files in os.walk(root,followlinks=False):
 for n in dirs:
  if pathlib.Path(d,n).is_symlink():raise ValueError('unexpected directory link')
 for n in files:
  f=pathlib.Path(d,n);st=f.lstat()
  if not stat.S_ISREG(st.st_mode):raise ValueError('nonregular file')
  count+=1;total+=st.st_size
  if count>100000:raise ValueError('bounded live file count exceeded')
  if f.suffix in {'.json','.jsonl','.log','.utf8','.py','.md','.txt','.sh','.sbatch'}:
   selected+=st.st_size;selected_count+=1
  largest.append((st.st_size,f.relative_to(root).as_posix()))
  if len(largest)>20:largest=sorted(largest,reverse=True)[:10]
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'35482135','in_progress':True,'files':count,'logical_bytes':total,'illustrative_text_files':selected_count,'illustrative_text_bytes':selected,'largest':sorted(largest,reverse=True)[:10],'not_full_collector_selection':True,'acceptance':False,'payload_decoded':False}))
