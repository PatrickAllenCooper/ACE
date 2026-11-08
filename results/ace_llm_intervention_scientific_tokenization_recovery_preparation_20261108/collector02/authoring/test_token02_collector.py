"""New TOKEN02 collector fixtures only; no remote/core main/scientific execution."""
import ast,copy,hashlib,importlib.util,json,os,shlex,shutil,types,unittest
from pathlib import Path
from unittest import mock
HERE=Path(__file__).resolve().parent
FIXTURES=Path(os.environ['TOKEN02_COLLECTOR_FIXTURES']).resolve()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
C=load('new_token02_collector',HERE/'pull_terminal.py');H=load('cleared_stream_helper',HERE/'stream_custody.py')
class Token02Collector(unittest.TestCase):
 def test_44rows90journals_retain_failed_token_charge_and_request(self):
  job='9000044';t='billing=3,cpu=3,node=1,mem=11520M'
  row=[job,'ace_llm_scientific_token02','COMPLETED','0:0','3','71','ucb736_asc2',t]
  rows=[[str(100+i),'ace_llm_fixture','COMPLETED','0:0','2','1','ucb736_asc2','cpu=2,node=1,mem=7680M'] for i in range(42)]+[[C.FAILED_TOKEN_JOB,'ace_llm_scientific_token01','FAILED','1:0','3','129','ucb736_asc2',t]]
  jp={f'/fixture/{i}':str(i).zfill(64) for i in range(88)}
  baseline=dict(related_accounting=copy.deepcopy(rows),journal_pins=jp)
  obs=dict(parent_accounting=row,related_accounting=rows+[row],journal_pins={**jp,C.ROOT+'/submission.intent.json':'a'*64,C.ROOT+'/submission.result.json':'b'*64},results=dict(details=dict(returncode=0,stdout=f'{job}|acpu|cpu-normal|10|{t}|paco0228|\n'),job=dict(returncode=0,stdout=f'JobId={job} Requeue=0 Restarts=0')))
  self.assertEqual(C.accounting_extension(obs,baseline,job,'a'*64,'b'*64),(True,True));self.assertEqual(C.cost(obs['related_accounting']),(84+387+213,0))
  self.assertTrue(C.resources(obs,job)['resource_contract_matched'])
  bad=copy.deepcopy(obs);bad['parent_accounting'][1]='ace_llm_scientific_token01';self.assertFalse(C.resources(bad,job)['resource_contract_matched'])
  bad=copy.deepcopy(obs);bad['related_accounting'][42][5]='0';self.assertFalse(C.accounting_extension(bad,baseline,job,'a'*64,'b'*64)[0])
  bad=copy.deepcopy(obs);bad['journal_pins']['/fixture/0']='c'*64;self.assertFalse(C.accounting_extension(bad,baseline,job,'a'*64,'b'*64)[1])
  root=FIXTURES/'allocation';root.mkdir();record=dict(returncode=0,stdout=f'JobId={job} Requeue=0 Restarts=0');(root/'allocation.json').write_bytes(C.raw(record));files={'launch/preparation/allocation.json':C.fingerprint(root/'allocation.json')}
  obs['results']['job']=dict(returncode=1,stdout='')
  self.assertTrue(C.no_requeue_record(obs,job,root,files)['verified'])
  record['stdout']=f'JobId={job} Requeue=1 Restarts=1';(root/'allocation.json').write_bytes(C.raw(record));files['launch/preparation/allocation.json']=C.fingerprint(root/'allocation.json')
  self.assertFalse(C.no_requeue_record(obs,job,root,files)['verified'])
 def test_failed_token_and_both_collections_are_immutable_including_empty_dirs(self):
  preserved={label:{'original.bin':dict(bytes=1,sha256=str(i).zfill(64))} for i,label in enumerate(('token01','common','failed01'))}
  dirs={label:['empty'] for label in preserved};actual=dict(files={label+'/'+n:v for label,values in preserved.items() for n,v in values.items()},directories=[label+'/empty' for label in preserved])
  self.assertEqual(C.preserved_roots_equal(actual,preserved,dirs),[])
  for label in preserved:
   bad=copy.deepcopy(actual);bad['files'][label+'/extra.bin']=dict(bytes=1,sha256='f'*64)
   self.assertTrue(C.preserved_roots_equal(bad,preserved,dirs))
   bad=copy.deepcopy(actual);bad['directories'].remove(label+'/empty');self.assertTrue(C.preserved_roots_equal(bad,preserved,dirs))
 def test_four_roots_full_archives_and_partial_capture_preserved(self):
  source_bytes=(HERE/'stream_custody.py').read_bytes()
  for scenario in ('complete','partial'):
   root=FIXTURES/scenario;root.mkdir();source=root/'source';source.mkdir();ext=root/'external';local=root/'local';ext.mkdir();local.mkdir()
   for label in ('launch','token01','common','failed01'):
    p=source/label;p.mkdir();(p/'empty').mkdir();(p/'original.bin').write_bytes(b'opaque\x00'+label.encode())
   for n in ('namespace.json','original-namespace.json'):(source/n).write_bytes(b'opaque original claim')
   before=dict(**C.inventory(source),root_present=True,failed_token_present=True,common_present=True,failed_present=True,namespace_present=True,original_namespace_present=True,original_hardlink_groups=[],parent_accounting=[],related_accounting=[],journal_pins={},upstream_pins={},upstream_errors={})
   calls=[]
   def capture(command,request,attempt,timeout):
    calls.append(command);self.assertEqual(command[:3],['bash',C.HELPER,'run']);self.assertEqual(shlex.split(command[3]),['python3','-B','-c',source_bytes.decode(),'emit','--request-sha256',C.sha(request)])
    attempt.mkdir();C.write(attempt/'request.json',request);C.write(attempt/'command.json',C.raw(command))
    if scenario=='partial':
     C.write(attempt/'archive.tar',b'partial original');C.write(attempt/'stderr.log',b'fabricated failure');raise RuntimeError('fabricated capture failure')
    with (attempt/'archive.tar').open('xb') as f:receipt=H.emit(C.decode(request),f)
    C.write(attempt/'stderr.log',(json.dumps(receipt)+'\n').encode());C.write(attempt/'capture.json',C.raw(dict(status='captured_pending_inventory_verification',returncode=0,request_sha256=C.sha(request),command_sha256=C.sha(C.raw(command)),**{n:C.fingerprint(attempt/n) for n in ('archive.tar','stderr.log')})))
   def save(n,v):
    for dest in (ext,local):C.write(dest/n,C.raw(v))
   helper=types.SimpleNamespace(capture=capture,request_check=H.request_check,archive_bound=H.archive_bound,materialize=H.materialize)
   changes=dict(ROOT=str(source/'launch'),FAILED_TOKEN_ROOT=str(source/'token01'),COMMON_ROOT=str(source/'common'),FAILED_ROOT=str(source/'failed01'),REMOTE_NAMESPACE=str(source/'namespace.json'),OLD_REMOTE_NAMESPACE=str(source/'original-namespace.json'))
   with mock.patch.multiple(C,**changes):
    call=lambda:C.stream_transfer(before,observe=lambda _:copy.deepcopy(before),save=save,helper=helper,helper_source=source_bytes,external=ext,local=local,copy_file=shutil.copyfile)
    if scenario=='partial':
     with self.assertRaisesRegex(RuntimeError,'fabricated capture'):call()
     self.assertFalse((ext/'remote').exists())
    else:
     call();self.assertEqual(C.inventory(ext),C.inventory(local));self.assertFalse(C.decode((ext/'stream-transport.json').read_bytes())['acceptance'])
   self.assertEqual(len(calls),1);self.assertEqual(C.inventory(ext/'stream-attempt'),C.inventory(local/'stream-attempt'))
   self.assertNotEqual((ext/'stream-attempt/archive.tar').stat().st_ino,(local/'stream-attempt/archive.tar').stat().st_ino)
 def test_exact_core_and_standalone_binding_order_and_no_task_summary_change(self):
  text=(HERE/'pull_terminal.py').read_text();tree=ast.parse(text);old=ast.parse((HERE/'predecessors/pull_terminal.py').read_text())
  funcs=lambda t:{n.name:ast.dump(n) for n in t.body if isinstance(n,ast.FunctionDef)}
  self.assertEqual(funcs(tree)['task_counts'],funcs(old)['task_counts'])
  self.assertEqual(hashlib.sha256((HERE/'predecessors/token02-build_packet.py').read_bytes()).hexdigest(),C.BUILDER_SHA)
  self.assertLess(text.index("'independent original packet review/commit required'"),text.index('ctx=b.admit(supplied)'))
  self.assertLess(text.index('ctx=b.admit(supplied)'),text.index("before=observe('inventory_transport.json')"))
  self.assertIn("pins['token-failure-baseline.json']",text);self.assertIn("('token01','launch')",text)
  self.assertIn("'collector-repository-path'",text);self.assertNotIn("pins.get('pull_terminal.py')==a.collector_sha256",text)
  self.assertIn("('remote','coordinator','upstream-token01-custody','stream-attempt')",text)
  remote=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='REMOTE_OBSERVATION' for x in n.targets))
  self.assertIn("('token01',Path(FAILED_TOKEN_ROOT),v['failed_token_present'])",remote)
  compile(remote,'remote-observer-syntax-only','exec')

if __name__=='__main__':unittest.main(verbosity=2)
