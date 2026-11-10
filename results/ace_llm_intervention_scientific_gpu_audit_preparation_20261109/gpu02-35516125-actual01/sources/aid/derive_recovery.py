"""Author a separate derivative from pinned text. Never execute the original aid."""
import ast
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent
old=(ROOT/'predecessor/preparation02/audit_saved_gpu72.py').read_bytes()
assert hashlib.sha256(old).hexdigest()=='e28392bb02adc2441c68be9279704cfec795f6af8d9d587a2bf2b41157318908'
text=old.decode()

def replace(a,b,count=1):
    global text
    assert text.count(a)==count,(a,text.count(a),count)
    text=text.replace(a,b)

helper=hashlib.sha256((ROOT/'dependencies/gpu_recovery_delta.py').read_bytes()).hexdigest()
replace("BACKEND_SHA=", "RECOVERY_SHA="+repr(helper)+"\nPREFLIGHT_SHA='b9d0bf56e6c3174ad4ca2c21a43e3c13372c4755575a945437137dc2a1e942c8'\nBACKEND_SHA=")
replace("supp=dependency('audit_supplement.py',SUPPLEMENT_SHA)",
        "supp=dependency('audit_supplement.py',SUPPLEMENT_SHA)\nrecovery=dependency('gpu_recovery_delta.py',RECOVERY_SHA)\npreflight=dependency('llm_intervention_cpu_preflight_receipt.py',PREFLIGHT_SHA)")
replace("and set(current['roots'])==set(mapping.values())|{'launch'}", "and set(current['roots'])==set(mapping.values())|{'launch','control'}")
replace("return name=='launch' or name.startswith('launch/') or any", "return name=='control' or name.startswith('control/') or name=='launch' or name.startswith('launch/') or any")
replace("  super().__init__(cfg);self.oracles,self.oracle_pins=load_oracles();self.completed=[];self.cells=[]",
        "  super().__init__(cfg);self.oracles,self.oracle_pins=load_oracles();self.completed=[];self.cells=[]\n"
        "  for filename,name in [('host-entry04.py',recovery.ENTRY),('history-adapter04.py',recovery.HISTORY),('preflight-checker.py',recovery.CHECKER)]:\n"
        "   aid.file_bytes(HERE/'recovery-references'/filename,recovery.OPERATIONAL[name],maximum=1024**2)")
replace("ace-prospective-gpu72-saved-audit-input-v1", "ace-prospective-gpu72-recovery-saved-audit-input-v2")
replace("  self.before,after=observations", "  self.before,after=observations;self.after=after")
replace("  need(inventories[0]==inventories[1],'original second directory inventory differs')", """  need(inventories[0]==inventories[1],'original second directory inventory differs')
  for key in ('root_present','failed_gpu_root_present','cpu_present','training_present','token_present',
              'failed_token_present','common_present','failed_present','namespace_present','original_namespace_present',
              'current_control_intent_base64'):
   need(self.before[key]==after[key],'second recovery observation differs: '+key)
  need(self.before['results']['details']==after['results']['details'],'second saved scheduler request differs')""")
# All operational inputs/journals are under control. Output remains launch/gpu.
replace("'launch/inputs/", "'control/inputs/",count=text.count("'launch/inputs/"))
replace("('outer','launch/packet.json')", "('outer','control/packet.json')")
replace("('result','launch/submission.result.json')", "('result','control/submission.result.json')")
replace("('remote_intent','launch/submission.intent.json')", "('remote_intent','control/submission.intent.json')")
replace("('baseline','launch/cpu-baseline.json')", "('baseline','control/gpu-failure-baseline.json')")
replace("('freeze','launch/freeze.json')", "('freeze','control/freeze.json')")
replace("('training_acceptance','launch/training-acceptance.json')", "('training_acceptance','control/training-acceptance.json')")
replace("self.h('launch/cpu-disposition.json'", "self.h('control/cpu-disposition.json'")
replace("self.h('launch/'+aid.name(n),h)", "self.h('control/'+aid.name(n),h)")
replace("p['schema']=='ace-scientific-paired-campaign-gpu-v1'", "p['schema']=='ace-scientific-paired-campaign-gpu-cpu-history-v2'")
replace("and p['resources']==resources and p['limits']==limits and self.outer['source_commit']==SOURCE,'frozen scientific GPU contract')", """and p['resources']==resources and p['limits']==limits and self.outer['source_commit']==SOURCE
       and self.outer['schema']=='ace-scientific-campaign-gpu-recovery-launch-v2'
       and self.outer['gpu_packet_sha256']==self.packet_pin
       and self.outer['baseline_sha256']==self.c['artifacts']['baseline']['sha256']
       and p['recovery_files']==recovery.RECOVERY_FILES,'explicit recovery with frozen scientific GPU contract')""")
a="""  need(p['source_names']==closure['source_names'],'exact frozen stage source closure differs')
  for name in p['source_names']:
   origins=[h for n,h in freeze['source_pins'].items() if PurePosixPath(n).name==name]
   need(len(origins)==1 and origins[0]==p['files'][name]==closure['input_source_files'][name],'captured source/freeze basename/closure differs')
   if name in self.oracle_pins:need(p['files'][name]==self.oracle_pins[name],'original source oracle differs')"""
b="""  authority=aid.decode(self.artifact('recovery_authority'));context=aid.decode(self.artifact('recovery_context'))['execution_context']
  manifest=aid.decode(self.artifact('operational_manifest'))
  recovery.validate_sources(p,closure,authority,context,manifest,freeze['source_pins'])
  need(self.outer['operational_source_commit']==manifest['source_commit'],'outer operational Git identity differs')
  for name in p['source_names']:
   if name in self.oracle_pins and name!=ENTRY:need(p['files'][name]==self.oracle_pins[name],'original scientific oracle differs')
  for role,key in [('authority','recovery_authority'),('context','recovery_context'),('receipt','preflight_receipt'),('read_set','readset')]:
   need(p['files'][p['recovery_files'][role]]==self.c['artifacts'][key]['sha256'],'independent recovery role differs')"""
replace(a,b)
replace("root=supp.constant(builder,'ROOT');cpu_root=supp.constant(builder,'CPU_ROOT')", "root=recovery.CONTROL;cpu_root=supp.constant(builder,'CPU_ROOT')")
replace("gpu_inputs=root+'/inputs',gpu_output=root+'/gpu')", "gpu_inputs=root+'/inputs',gpu_output=recovery.OUTPUT)")
replace("and self.d['remote_root']==root and self.d['output_root']==root+'/gpu'", "and self.d['remote_root']==self.outer['remote_root']==root and self.d['output_root']==recovery.OUTPUT")
replace("  self.facts['packet']=dict(source_commit=SOURCE", "  recovery.recovery(self,preflight,aid.decode,raw,aid.validate_inventory)\n  self.facts['packet']=dict(source_commit=SOURCE")
replace("baseline['terminal_jobs']==len(prior)==46 and len(rows)==47", "baseline['terminal_jobs']==len(prior)==47 and len(rows)==48")
replace("baseline['submission_journals']==len(baseline['journal_pins'])==94", "baseline['submission_journals']==len(baseline['journal_pins'])==96")
replace("'exact46+1 actual history differs'", "'exact47+1 actual history differs'")
replace("need(len(journals)==96 and obs['journal_pins']==journals,'full94+2 journals differ')", """need(len(journals)==98 and obs['journal_pins']==journals,'full96+2 journals differ')
  recovery.history_extension(baseline,rows,obs['journal_pins'],job,result['intent_sha256'],self.c['artifacts']['result']['sha256'],aid.normalize)""")
replace("jobs=47,journals=96", "jobs=48,journals=98")
replace("r[2:4]==['COMPLETED','0:0'] and r[4]==4", "r[2:4]==['COMPLETED','0:0'] and r[1]=='ace_llm_scientific_campaign_gpu02' and r[4]==4")
replace("  self.facts['terminal']=dict(", "  recovery.recovery_terminal(self)\n  self.facts['terminal']=dict(")
# Gather new, bounded control metadata/log omissions without changing the closed scientific selector.
replace(" def fetch_requirements(self):", """ def recovery_fetch_requirements(self):
  p=aid.decode(self.artifact('inner'))
  need(p['schema']=='ace-scientific-paired-campaign-gpu-cpu-history-v2','new packet schema required')
  wanted={'control/inputs/cpu-packet.json'}
  need(set(p['services'])==set(LLMS),'exact three service freezes required')
  wanted.update('control/inputs/'+aid.name(p['services'][policy]) for policy in LLMS)
  logs={'control/logs/'+self.c['job_id']+suffix for suffix in ('.out','.err')}
  wanted.update(logs)
  need({n for n in self.full if n.startswith('control/logs/')}==logs,'exact current-job stdout/stderr custody required')
  for n in sorted(wanted):
   need(n in self.full,'required control metadata/log member absent: '+n)
   need(self.full[n]['bytes']<=32*1024**2,'control member exceeds bounded exact materialization; separate reviewed path needed')
   if n not in self.local and n not in self.c.get('fetched',{}):self.missing[n]=self.full[n]
  need(sum(self.full[n]['bytes'] for n in wanted)<=512*1024**2,'control fetch aggregate exceeds bound')
  self.facts['control_selection']={'required_members':{n:self.full[n] for n in sorted(wanted)},
      'collector_selects_control':False,'batch03_selects_control':False,'control_log_content_interpreted':False}
  need(not self.missing,'new control metadata/log exact fetches required; batch03 is insufficient')
  for n in sorted(logs):self.b(n)  # Hash opaque log bytes; do not parse outcomes.
 def fetch_requirements(self):""")
replace("for name in ('custody','packet','accounting','runtime','terminals','fetch_requirements','services'):", "for name in ('custody','recovery_fetch_requirements','packet','accounting','runtime','terminals','fetch_requirements','services'):")
replace("schema='ace-gpu72-saved-audit-report-v1'", "schema='ace-gpu72-recovery-saved-audit-report-v2'")
replace("dependency_sha256={'aid':AID_SHA,'supplement':SUPPLEMENT_SHA}", "dependency_sha256={'aid':AID_SHA,'supplement':SUPPLEMENT_SHA,'recovery':RECOVERY_SHA,'preflight_metadata_checker':PREFLIGHT_SHA}")
replace("Only pinned pure audit dependencies execute.", "Only pinned pure audit/receipt metadata dependencies execute.")
ast.parse(text)
with (ROOT/'audit_saved_gpu72.py').open('x') as f:f.write(text)
