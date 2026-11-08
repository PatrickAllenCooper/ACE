"""New failed-TOKEN custody/history branches only; no remote/science/model calls."""
import copy,importlib.util,json,os,shutil,types,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;EVIDENCE=Path(os.environ['ACE_HISTORY_TEST_EVIDENCE'])
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
T=load('toy_gpu_core_fixture',HERE/'test_token02_history.py');B=T.B
C=load('gpu_collector_token02',HERE/'collector/pull_terminal.py')
H=load('accepted_stream_bytes',HERE/'collector/stream_custody.py')
SOURCE=(HERE/'collector/stream_custody.py').read_bytes()

class NewCollectorHistory(unittest.TestCase):
    def setUp(self):self.out=EVIDENCE/self._testMethodName;self.out.mkdir()
    def context(self):
        m,p=T.fixture();cpuresult=dict(job_id='99999999',intent_sha256=T.h('new intent'))
        old=m['token-failure-baseline'];new=T.row(cpuresult['job_id'],400)
        baseline=dict(terminal_jobs=44,submission_journals=90,related_accounting=old['related_accounting']+[new],
            journal_pins={**old['journal_pins'],C.CPU_ROOT+'/submission.intent.json':cpuresult['intent_sha256'],C.CPU_ROOT+'/submission.result.json':p['cpu-result']},
            closed_cpu_seconds=1713,closed_gpu_slice_seconds=0,outstanding_reservation=0,preparation_allowance_cpu_seconds=3600)
        names={n+'.json':v for n,v in p.items()};names['token-operational-adoption-review.md']=p['token-operational-adoption-review']
        expected=dict(amendment_sha256=p['token-operational-amendment'],adoption_review_sha256=p['token-operational-adoption-review'],
            failure_acceptance_sha256=p['token-failure-acceptance'],no_learnedwork_proof_sha256=p['token-no-learnedwork-proof'],
            failure_baseline_sha256=p['token-failure-baseline'],original_packet_sha256=p['failed-token-packet'],original_stage_packet_sha256=p['failed-token-stage'])
        packet=dict(token_recovery=expected,failed_token_root=C.FAILED_CPU_ROOT,failed_token_job_id=C.FAILED_CPU_JOB,source_commit=B.SOURCE_COMMIT)
        m['cpu-launch-packet']=dict(token_recovery=expected)
        m['token-operational-amendment']=dict(schema='ace-scientific-tokenization-operational-amendment-v1',status='adopted',stage='tokenize',recovery_id='token02',freeze_sha256=p['freeze'],source_commit=B.SOURCE_COMMIT,
            failed_root=C.FAILED_CPU_ROOT,recovery_root=C.CPU_ROOT,failure_acceptance_sha256=p['token-failure-acceptance'],no_learnedwork_proof_sha256=p['token-no-learnedwork-proof'],
            baseline_sha256=p['token-failure-baseline'],collection_baseline_sha256=p['previous-baseline'],original_deadlines=B.TOKEN_ORIGINAL_DEADLINES,deadlines=B.TOKEN_DEADLINES,
            scientific_inputs_unchanged=True,other_stages_unchanged=True,automatic_retry_allowed=False,replacement_worlds_allowed=False)
        original=json.loads(m['token-failure-inventory']['stdout'])
        files={'producer_failed_label/'+n[len('launch/'):]:v for n,v in original['files'].items() if n.startswith('launch/')}
        dirs=['producer_failed_label']+['producer_failed_label/'+n[len('launch/'):] for n in original['directories'] if n.startswith('launch/')]
        def read(path,pin):return C.raw(m[Path(path).stem])
        return m,names,packet,baseline,files,cpuresult,dirs,read
    def test_failed_history_context_and45_92_extension(self):
        m,p,packet,base,files,result,dirs,read=self.context()
        with patch.object(C,'read_metadata',side_effect=read):
            v=C.token_failure_context(p,packet,packet,base,files,result,dirs)
            self.assertEqual(v['producer_failed_label'],'producer_failed_label')
            broken=copy.deepcopy(files);broken.pop('producer_failed_label/preparation/failure.json')
            with self.assertRaisesRegex(ValueError,'raw bytes'):C.token_failure_context(p,packet,packet,base,broken,result,dirs)
            with self.assertRaisesRegex(ValueError,'empty directories'):C.token_failure_context(p,packet,packet,base,files,result,dirs[:-1])
            m['token-no-learnedwork-proof']['unknown_return_attempts']=None
            with self.assertRaisesRegex(ValueError,'unknown/nonzero'):C.token_failure_context(p,packet,packet,base,files,result,dirs)
        job='88888888';current=[job,'fixture train','COMPLETED','0:0','4','500','ucb736_asc2','cpu=4,gres/gpu=1,gres/gpu:a100_3g.20gb=1']
        obs=dict(related_accounting=base['related_accounting']+[current],journal_pins={**base['journal_pins'],C.ROOT+'/submission.intent.json':T.h('train intent'),C.ROOT+'/submission.result.json':T.h('train result')})
        self.assertEqual(C.accounting_extension(obs,base,job,T.h('train intent'),T.h('train result')),(True,True))
        bad=copy.deepcopy(obs);bad['related_accounting']=[r for r in bad['related_accounting'] if r[0]!=C.FAILED_CPU_JOB]
        self.assertFalse(C.accounting_extension(bad,base,job,T.h('train intent'),T.h('train result'))[0])
        (self.out/'fixture.json').write_bytes(C.raw(dict(artificial=True,packet=packet,baseline=base,observation=obs)))
    def test_whole_failed_capsule_is_independently_required(self):
        source=self.out/'failed-capsule';source.mkdir();(source/'remote').mkdir();(source/'remote/empty').mkdir()
        C.write(source/'remote/failure.bin',b'artificial failure and opaque raw bytes')
        C.write(source/'archive-receipt.json',b'opaque original receipt')
        expected=C.inventory(source/'remote');before=C.inventory(source)
        C.write(source/'custody-inventory.json',C.raw(dict(identical=True,copies={str(source):before,'/other/artificial/copy':before})))
        pin=C.sha((source/'custody-inventory.json').read_bytes())
        root,_=C.load_failed_token_capsule(source,pin,expected);self.assertEqual(root,source)
        C.write(source/'unexpected.txt',b'retained intrusion')
        with self.assertRaisesRegex(ValueError,'whole failed'):C.load_failed_token_capsule(source,pin,expected)
        self.assertTrue((source/'unexpected.txt').exists())
    def test_stream_materializes_new_fifth_failed_root(self):
        source=self.out/'fabricated';source.mkdir();external=self.out/'external';local=self.out/'local';external.mkdir();local.mkdir()
        for label in ('launch','cpu','common','failed01','failed-token01'):
            d=source/label;d.mkdir();(d/'empty').mkdir();C.write(d/'opaque.bin',b'artificial '+label.encode())
        for label in ('namespace.json','original-namespace.json'):C.write(source/label,b'artificial namespace')
        paths=dict(ROOT=str(source/'launch'),CPU_ROOT=str(source/'cpu'),COMMON_ROOT=str(source/'common'),FAILED_ROOT=str(source/'failed01'),FAILED_CPU_ROOT=str(source/'failed-token01'),REMOTE_NAMESPACE=str(source/'namespace.json'),OLD_REMOTE_NAMESPACE=str(source/'original-namespace.json'))
        before=dict(**C.inventory(source),root_present=True,cpu_present=True,common_present=True,failed_present=True,failed_token_present=True,
            namespace_present=True,original_namespace_present=True,original_hardlink_groups=[],parent_accounting=['fabricated'],journal_pins={},upstream_pins={},upstream_errors={},related_accounting=[])
        calls=[]
        def capture(command,request,attempt,timeout):
            calls.append(command);self.assertEqual(timeout,1800);attempt.mkdir()
            C.write(attempt/'request.json',request);C.write(attempt/'command.json',C.raw(command))
            with (attempt/'archive.tar').open('xb') as f:emitted=H.emit(C.decode(request),f)
            C.write(attempt/'stderr.log',(json.dumps(emitted,sort_keys=True)+'\n').encode())
            receipt=dict(status='captured_pending_inventory_verification',returncode=0,acceptance=False,request_sha256=C.sha(request),command_sha256=C.sha(C.raw(command)),
                **{name:C.fingerprint(attempt/name,sole_link=True) for name in ('archive.tar','stderr.log')})
            C.write(attempt/'capture.json',C.raw(receipt));return receipt
        def save(name,v):
            for d in (external,local):C.write(d/name,C.raw(v))
        helper=types.SimpleNamespace(request_check=H.request_check,archive_bound=H.archive_bound,capture=capture,materialize=H.materialize)
        with patch.multiple(C,**paths):
            actual,_=C.stream_transfer(before,observe=lambda _:copy.deepcopy(before),save=save,helper=helper,helper_source=SOURCE,external=external,local=local,copy_file=shutil.copyfile)
        self.assertEqual(len(calls),1);self.assertIn('failed-token01/opaque.bin',actual['files'])
        self.assertIn('failed-token01/empty',actual['directories'])
        self.assertEqual(C.inventory(external),C.inventory(local))
        self.assertNotEqual((external/'remote/failed-token01/opaque.bin').stat().st_ino,(local/'remote/failed-token01/opaque.bin').stat().st_ino)

if __name__=='__main__':unittest.main(verbosity=2)
