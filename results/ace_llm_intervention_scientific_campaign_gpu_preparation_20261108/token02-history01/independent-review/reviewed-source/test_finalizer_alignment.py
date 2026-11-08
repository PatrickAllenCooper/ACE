"""One new fabricated producer/consumer alias check; no finalizer main or old tests."""
import ast,copy,json,unittest
from pathlib import Path
import test_history_delta as T
G,C=T.G,T.C
class FinalizerAlignment(unittest.TestCase):
 def test_exact_finalizer_metadata_aliases_and_seventh_root(self):
  source=(T.HERE/'interfaces/finalizer-build_packet.py').read_bytes()
  self.assertEqual(G.sha(source),'63ec2ae617de0bf7ff0a6428a8c91e296c04aca4a9829a8ab15c4e00d2194c71')
  required=tuple(G.literal(source,'FIXED'))+G.literal(source,'TOKEN_METADATA')+('token-packet','namespace-claim','cpu-review','cpu-packet','cpu-phase-terminal')+tuple(prefix+'-'+kind for prefix,*_ in G.literal(source,'STAGES')[:-1] for kind in ('acceptance','launch-packet','result','disposition','inventory','baseline'))
  self.assertEqual(set(required),set(G.METADATA)-{'site-readiness'})
  self.assertEqual(G.HISTORY_ALIASES['cpu-packet'],'token-packet');self.assertNotIn('token-stage',G.METADATA)
  supplied,m,pins,obs=T.gpu_fixture();bound=G.token_recovery_history(supplied,m,pins,obs)
  for k in G.METADATA:pins.setdefault(k,T.h(k))
  m['gpu-launch-packet']=dict(token_recovery=bound['token_recovery'],failed_token_root=G.FAILED_TOKEN_ROOT,failed_token_job_id=G.FAILED_TOKEN_JOB,new_input_files={G.metadata_name(n):pins[n] for n in G.METADATA})
  # The isolated, unchanged consumer function checks actual metadata names; no finalizer imports/main.
  function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='recovery_joins')
  scope=dict(TOKEN_METADATA=G.TOKEN_METADATA,FAILED_TOKEN_ROOT=G.FAILED_TOKEN_ROOT,FAILED_TOKEN_JOB=G.FAILED_TOKEN_JOB,need=G.need,metadata_name=G.metadata_name)
  exec(compile(ast.Module(body=[function],type_ignores=[]),'captured-finalizer-recovery-joins','exec'),scope)
  scope['recovery_joins'](m,pins,bound['token_recovery'])
  bad=copy.deepcopy(m);bad['gpu-launch-packet']['new_input_files']['token-stage.json']=bad['gpu-launch-packet']['new_input_files'].pop('token-packet.json')
  with self.assertRaises((KeyError,ValueError)):scope['recovery_joins'](bad,pins,bound['token_recovery'])
  before=dict(files={},directories=[],**{n:True for n in ('root_present','cpu_present','training_present','token_present','failed_token_present','common_present','failed_present','namespace_present','original_namespace_present')})
  request=C.stream_request(before);expected=G.literal(source,'RAW_ROOTS');expected['failed-token01']=G.FAILED_TOKEN_ROOT
  self.assertEqual(request['roots'],expected);self.assertEqual(set(C.UPSTREAM_ROOTS)|{'launch'},set(expected))
  self.assertEqual(C.BUILDER_SHA,G.sha((T.HERE/'build_packet.py').read_bytes()))
  (T.EVIDENCE/'finalizer-alignment.json').write_bytes(G.raw(dict(finalizer_sha256=G.sha(source),required_captured_metadata=sorted(required),raw_roots=request['roots'],source_only=True)))
if __name__=='__main__':unittest.main(verbosity=2)
