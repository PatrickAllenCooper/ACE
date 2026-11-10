"""New operational metadata deltas only; never import a producer or invoke main."""
from pathlib import Path
import ast,copy,hashlib,json,os,re,stat,tempfile,types,unittest

HERE=Path(__file__).resolve().parent
ENTRY='finalize_llm_intervention_scientific_campaign_cpu.py'
def sha(b):return hashlib.sha256(b).hexdigest()
def raw(v):return (json.dumps(v,sort_keys=True,separators=(',',':'))+'\n').encode()
def need(ok,message):
    if not ok:raise ValueError(message)
def extract(name,functions):
    tree=ast.parse((HERE/'implementation'/name).read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in functions]
    assert len(nodes)==len(functions)
    ns=dict(Path=Path,hashlib=hashlib,json=json,os=os,re=re,stat=stat,need=need)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),name,'exec'),ns)
    return ns
def literal(name,key):
    tree=ast.parse((HERE/'implementation'/name).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==key for t in n.targets))
    return ast.literal_eval(node.value)

def child_fixture():
    ns=extract(ENTRY,{'admit_recovery','need','sha','pin','raw','decode'})
    for k in ('PREFLIGHT','HISTORY','RECOVERY_FILES','GPU_CONTROL_ROOT','GPU_ORIGINAL_ROOT','HISTORICAL_MARKER'):
        ns[k]=literal(ENTRY,k)
    h='1'*64;files={f'fixture{i:02}.py':h for i in range(56)};sources=dict(files)
    failed=dict(schema='ace-scientific-campaign-gpu-failure-acceptance-v1',status='accepted_failed_prepolicy_attempt',
                acceptance=True,required_findings=0,job_id='35498218')
    data={'gpu-failure-acceptance.json':raw(failed)}
    files['gpu-failure-acceptance.json']=sha(data['gpu-failure-acceptance.json'])
    prior=dict(terminal_jobs=47,submission_journals=96,input_pins=dict(acceptance=files['gpu-failure-acceptance.json']))
    data['gpu-failure-baseline.json']=raw(prior);files['gpu-failure-baseline.json']=sha(raw(prior))
    baseline=dict(terminal_jobs=48,submission_journals=98,outstanding_reservation=0,
                  input_pins=dict(predecessor=files['gpu-failure-baseline.json'],acceptance=h))
    data['gpu-success-baseline.json']=raw(baseline);files['gpu-success-baseline.json']=sha(raw(baseline))
    authority=dict(schema='ace-gpu-cpu-history-recovery-authority-v1',status='operationally_adopted',
        historical_postcustody_required=True,unchanged_scientific_inputs=True,
        control_root=ns['GPU_CONTROL_ROOT'],failed_gpu01_root=ns['GPU_ORIGINAL_ROOT'],
        allowed_new_output=ns['GPU_ORIGINAL_ROOT']+'/gpu',cpu_acceptance_sha256=h,
        original_source_commit='a'*40,original_entry_sha256=h,source_review_sha256=h,
        operational_source_files=sources,original_gpu01_job_id='35498218',
        failed_gpu01_acceptance_sha256=files['gpu-failure-acceptance.json'],
        baseline=dict(sha256=files['gpu-failure-baseline.json'],terminal_jobs=47,submission_journals=96))
    expected=dict(execution_context=dict(source_files_sha256=sha(raw(sources).rstrip(b'\n'))))
    for role,value in [('authority',authority),('context',expected),('receipt',{}),('read_set',{})]:
        n=ns['RECOVERY_FILES'][role];data[n]=raw(value);files[n]=sha(data[n])
    ns['RECOVERY_AUTHORITY_SHA']=files[ns['RECOVERY_FILES']['authority']]
    gp=dict(recovery_files=ns['RECOVERY_FILES'],files=dict(files),source_names=sorted(sources),
            paths=dict(gpu_output=ns['GPU_ORIGINAL_ROOT']+'/gpu',gpu_inputs=ns['GPU_CONTROL_ROOT']+'/inputs'))
    packet=dict(files=files,source_commit='a'*40,cpu_acceptance_sha256=h,gpu_acceptance_sha256=h,
                gpu_packet_sha256=h,paths=gp['paths'])
    scoped=dict(full_external_postcustody_required=True,acceptance=False)
    marker=dict(preflight=scoped,authority_sha256=ns['RECOVERY_AUTHORITY_SHA'],
                historical_postcheck_performed_on_gpu=False,full_external_postcustody_required=True,acceptance=False)
    marker_raw=raw(marker);marker_pin=sha(marker_raw)
    gpu=dict(job_id='42',campaign_root='/fixture/campaign',terminal_pins={'terminal.json':h,'controller/phase/terminal.json':h},
             gpu_output_files={ns['HISTORICAL_MARKER']:marker_pin})
    terminal=dict(operational_recovery_authority_sha256=ns['RECOVERY_AUTHORITY_SHA'],
        historical_preflight_receipt_sha256=files[ns['RECOVERY_FILES']['receipt']],
        historical_postcheck_performed_on_gpu=False,full_external_postcustody_required=True)
    s=dict(schema='ace-scientific-campaign-gpu-recovery-acceptance-v1',status='accepted',scope='development',
        acceptance=True,required_findings=0,scientific_qualified=False,historical_postcustody_verified=True,originals_preserved=True,
        job_id='42',gpu_acceptance_sha256=h,gpu_packet_sha256=h,cpu_acceptance_sha256=h,
        authority_sha256=ns['RECOVERY_AUTHORITY_SHA'],preflight_context_sha256=files[ns['RECOVERY_FILES']['context']],
        preflight_receipt_sha256=files[ns['RECOVERY_FILES']['receipt']],readset_sha256=files[ns['RECOVERY_FILES']['read_set']],
        failure_acceptance_sha256=files['gpu-failure-acceptance.json'],failure_baseline_sha256=files['gpu-failure-baseline.json'],
        baseline_sha256=files['gpu-success-baseline.json'],source_files_sha256=expected['execution_context']['source_files_sha256'],
        gpu_terminal_sha256=h,phase_terminal_sha256=h,control_root=ns['GPU_CONTROL_ROOT'],
        output_root=ns['GPU_ORIGINAL_ROOT']+'/gpu',raw_root_labels=dict(control=ns['GPU_CONTROL_ROOT'],launch=ns['GPU_ORIGINAL_ROOT']),
        cpu_campaign_root=gpu['campaign_root'],original_source_commit='a'*40,original_gpu_entry_sha256=h,
        failed_gpu01_job_id='35498218',before_observation_sha256=h,after_observation_sha256=h,custody_sha256=h,
        gpu_launch_packet_sha256=h,archive_receipt_sha256=h,failed_archive_receipt_sha256=h,
        operational_source_commit='b'*40,evidence_pins={'operational-source-review':h},postcustody_marker_sha256=marker_pin)
    data['gpu-recovery-acceptance.json']=raw(s)
    calls=[]
    def read(path,pin):
        need(str(path)==packet['paths']['gpu_output']+'/'+ns['HISTORICAL_MARKER'] and pin==marker_pin,'wrong marker')
        calls.append(str(path));return marker_raw
    ns['read']=read
    return ns,packet,gp,data,gpu,terminal,types.SimpleNamespace(check_preflight=lambda *a:scoped),s,calls

class RecoveryDelta(unittest.TestCase):
    def test_child_requires_independent_postcustody_and_exact_recovery_join(self):
        ns,p,gp,data,g,t,api,s,calls=child_fixture()
        ns['admit_recovery'](p,gp,data,g,t,api);self.assertEqual(len(calls),1)
        for key,value in [('historical_postcustody_verified',False),('acceptance',False),
                          ('required_findings',False),('gpu_acceptance_sha256','2'*64)]:
            bad=dict(s);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                ns['admit_recovery'](p,gp,{**data,'gpu-recovery-acceptance.json':raw(bad)},g,t,api)
        bad=copy.deepcopy(gp);bad['paths']['gpu_inputs']=ns['GPU_ORIGINAL_ROOT']+'/inputs'
        with self.assertRaises(ValueError):ns['admit_recovery'](p,bad,data,g,t,api)
        bad=dict(t);bad['historical_postcheck_performed_on_gpu']=True
        with self.assertRaises(ValueError):ns['admit_recovery'](p,gp,data,g,bad,api)

    def test_control_input_routing_and_original_directory_container(self):
        name='compact_inputs.py';ns=extract(name,{'need','sha','pin','canonical','relative','record','read_exact','observed_directories'})
        tree=ast.parse((HERE/'implementation'/name).read_text())
        method=copy.deepcopy(next(n for n in next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Inputs').body if isinstance(n,ast.FunctionDef) and n.name=='read'))
        method.decorator_list=[];exec(compile(ast.Module(body=[method],type_ignores=[]),name,'exec'),ns)
        self.assertEqual(ns['observed_directories'](['b','a']),['a','b'])
        for bad in ({'a':1},'a',['a','a'],['../a']):
            with self.subTest(dirs=bad),self.assertRaises(ValueError):ns['observed_directories'](bad)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);member='control/inputs/example.json';value=b'{}\n';info=dict(bytes=len(value),sha256=sha(value))
            (root/member).parent.mkdir(parents=True);(root/member).write_bytes(value)
            receipt={'fixture':True}
            obj=types.SimpleNamespace(full={'files':{member:info}},selected={'files':{member:info}},
                receipt=receipt,receipt_sha256=sha(ns['canonical'](receipt)),attempted=set(),used={},captured={},local_roots=(root,))
            self.assertEqual(ns['read'](obj,member,sha(value),256*1024**2),value)
            for bad in ('launch/inputs/example.json','control/logs/example.json','control/inputs/a/example.json'):
                with self.subTest(member=bad),self.assertRaises(ValueError):ns['read'](obj,bad,sha(value))
            with self.assertRaises(ValueError):ns['read'](obj,member,sha(value))

    def test_exact_new_accounting_extension_keeps_failed_job(self):
        ns=extract('pull_terminal.py',{'accounting_extension','normalize','cost','gpu_count','need'})
        ns.update(ROOT='/fixture/final',TERMINAL={'COMPLETED','FAILED'})
        rows=[[str(i),'ace_llm_fixture','COMPLETED','0:0','2','1','ucb736_asc2','cpu=2'] for i in range(1,48)]
        rows.append(['35498218','ace_llm_failed','FAILED','1:0','4','245','ucb736_asc2','cpu=4,gres/gpu=1'])
        jp={f'/fixture/{i}':'1'*64 for i in range(98)};before=ns['cost'](rows)
        base=dict(terminal_jobs=48,submission_journals=98,related_accounting=rows,journal_pins=jp,
                  closed_cpu_seconds=before[0],closed_gpu_slice_seconds=before[1])
        row=['99999999','ace_llm_final','COMPLETED','0:0','2','100','ucb736_asc2','cpu=2']
        obs=dict(related_accounting=rows+[row],parent_accounting=row,
                 journal_pins={**jp,'/fixture/final/submission.intent.json':'2'*64,'/fixture/final/submission.result.json':'3'*64})
        result=ns['accounting_extension'](obs,base,row[0],'2'*64,'3'*64)
        self.assertEqual((result['allocated_cpu_seconds'],result['gpu_slice_seconds'],result['closed_gpu_slice_seconds']),(200,0,245))
        bad=copy.deepcopy(obs);bad['related_accounting']=[r for r in bad['related_accounting'] if r[0]!='35498218']
        with self.assertRaises(ValueError):ns['accounting_extension'](bad,base,row[0],'2'*64,'3'*64)

    def test_static_reporting_lifecycle_resources_and_source_bindings_unchanged(self):
        old=ast.parse((HERE/'predecessor'/ENTRY).read_text());new=ast.parse((HERE/'implementation'/ENTRY).read_text())
        nodes=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        a,b=nodes(old),nodes(new)
        changed={k for k in a if a[k]!=b[k]}
        self.assertEqual(changed,{'packet_destination','admit_evidence','upstream_shape','authenticate'})
        self.assertEqual(set(b)-set(a),{'admit_recovery'})
        assigns=lambda tree:{ast.unparse(n.targets[0]):ast.dump(n.value,include_attributes=False) for n in tree.body if isinstance(n,ast.Assign)}
        for key in ('RESOURCES','(STARTUP, WORK, CLEANUP, HARD)','GPU_ACCEPTANCE_KEYS','GPU_SHA','CAMPAIGN_SHA','POLICIES','WORLD_IDS'):
            self.assertEqual(assigns(old)[key],assigns(new)[key],key)
        fn=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='admit_evidence')
        gate=next(n.lineno for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='admit_recovery')
        private=next(n.lineno for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='bundle' for t in n.targets))
        self.assertLess(gate,private)
        self.assertEqual(literal('build_packet.py','ENTRY_SHA'),sha((HERE/'implementation'/ENTRY).read_bytes()))
        self.assertEqual(literal('build_packet.py','COMPACT_ADAPTER_SHA'),sha((HERE/'implementation/compact_inputs.py').read_bytes()))
        self.assertEqual((HERE/'implementation/monitor_once.py').read_bytes(),(HERE/'predecessor/monitor_once.py').read_bytes())

if __name__=='__main__':unittest.main(verbosity=2)
