import json, sys, collections
p='/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/acceptance01/baseline.json'
d=json.load(open(p))
rows=d['related_accounting']
# submission UTC timestamps transcribed from the execution ledger (docs/development/guidance/ace_llm_intervention_execution_2026-11-02.md)
sub={
'35200690':'11-02 20:03','35202401':'11-02 20:46','35202931':'11-02 21:00','35203022':'11-02 21:05','35204167':'11-02 21:36',
'35204547':'11-02 21:44','35207064':'11-02 22:42','35208932':'11-02 23:07','35209536':'11-02 23:28','35209701':'11-02 23:42',
'35209904':'11-02 23:56','35210147':'11-03 00:14','35252306':'11-04 16:32','35341862':'11-07 16:25','35343088':'11-07 18:02',
'35343957':'11-07 18:36','35345616':'11-07 19:46','35347001':'11-07 20:09','35348012':'11-07 20:53','35348366':'11-07 21:06',
'35348887':'11-07 21:24','35349555':'11-07 21:39','35349776':'11-07 21:47','35350736':'11-07 22:08','35351262':'11-07 22:21',
'35353861':'11-07 22:36','35372844':'11-08 04:24','35374034':'11-08 ~04:5x','35377234':'11-08 05:42','35378508':'11-08 06:02',
'35379434':'11-08 06:19','35385156':'11-08 07:37','35385876':'11-08 07:48','35387068':'11-08 08:06','35388249':'11-08 08:30',
'35389391':'11-08 08:51','35391340':'11-08 09:28','35393751':'11-08 10:14','35402725':'11-08 12:20','35406081':'11-08 13:01',
'35407428':'11-08 13:26','35409614':'11-08 13:59','35415334':'11-08 15:11','35419195':'11-08 16:24','35420219':'11-08 17:01',
'35482135':'11-09 17:50','35498218':'11-09 22:13','35516125':'11-10 02:42','35533604':'11-10 06:32'}
def phase(i):
    if i<=12: return 'A_component_runtime_qualification(Nov2-3)'
    if i<=27: return 'B_service_runtime_qualification(Nov4-8)'
    if i<=38: return 'C_artificial_training_and_campaign_integration(Nov8)'
    return 'D_scientific_cycle(Nov8-10)'
agg=collections.OrderedDict()
tot=collections.Counter()
print('idx job name state cpus elapsed cpu_s gpu_s submitted')
for i,r in enumerate(rows,1):
    job,name,state,ec,cpus,el,acct,tres=r
    cpus=int(cpus); el=int(el)
    gpu = el if 'gres/gpu' in tres else 0
    cpu_s=cpus*el
    ph=phase(i)
    a=agg.setdefault(ph,collections.Counter())
    a['jobs']+=1; a['cpu_s']+=cpu_s; a['gpu_s']+=gpu
    a['failed']+= (state!='COMPLETED'); a['completed']+=(state=='COMPLETED')
    a['gpu_jobs']+= (gpu>0); a['gpu_failed']+= (gpu>0 and state!='COMPLETED')
    a['failed_cpu_s']+= cpu_s if state!='COMPLETED' else 0
    a['failed_gpu_s']+= gpu if state!='COMPLETED' else 0
    tot.update({k:v for k,v in [('jobs',1),('cpu_s',cpu_s),('gpu_s',gpu),('failed',state!='COMPLETED'),('completed',state=='COMPLETED'),('gpu_jobs',gpu>0),('gpu_failed',(gpu>0 and state!='COMPLETED')),('failed_cpu_s',cpu_s if state!='COMPLETED' else 0),('failed_gpu_s',gpu if state!='COMPLETED' else 0)]})
    print(i,job,name.replace('ace_llm_',''),state,cpus,el,cpu_s,gpu,sub[job],acct)
print()
for ph,a in agg.items():
    print(ph, dict(a))
print('TOTAL',dict(tot))
print('ledger closed', d['closed_cpu_seconds'], d['closed_gpu_slice_seconds'], 'admission', d['cpu_admission_seconds'], 'caps', d['cpu_cap_seconds'], d['gpu_cap_seconds'])
print('remaining vs cap: cpu', d['cpu_cap_seconds']-d['cpu_admission_seconds'], 'gpu', d['gpu_cap_seconds']-d['closed_gpu_slice_seconds'])
print('core-hours cpu closed', d['closed_cpu_seconds']/3600, 'gpu slice-hours', d['closed_gpu_slice_seconds']/3600)
# scientific-stage breakdown
sci={ 'realization':[39,40],'collection':[41,42],'tokenization':[43,44],'training':[45],'cpu_campaign':[46],'gpu_campaign':[47,48],'finalization':[49]}
print()
for st,idxs in sci.items():
    c=g=0; f=0
    for i in idxs:
        job,name,state,ec,cpus,el,acct,tres=rows[i-1]
        c+=int(cpus)*int(el); g+= int(el) if 'gres/gpu' in tres else 0; f+= state!='COMPLETED'
    print(st, 'jobs',len(idxs),'failed_jobs',f,'cpu_s',c,'gpu_s',g)
