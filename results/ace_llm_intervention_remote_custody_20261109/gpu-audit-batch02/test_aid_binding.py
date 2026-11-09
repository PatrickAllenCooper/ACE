"""One NEW changed-pin gate regression; no full batch or audit execution."""
import ast,base64,hashlib,json,os,re,types,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
FIXTURES=Path(os.environ['AID_PIN_FIXTURES']).resolve()
sha=lambda b:hashlib.sha256(b).hexdigest()
def need(ok,message):
 if not ok:raise ValueError(message)

class Binding(unittest.TestCase):
 def test_new_aid_binding_accepts_exact_bytes_rejects_old_and_altered(self):
  tree=ast.parse((HERE/'batch_gpu_audit.py').read_bytes())
  pin=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
   and any(isinstance(t,ast.Name) and t.id=='GPU_AUDIT_SHA256' for t in n.targets))
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_selection')
  index=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
   and any(isinstance(t,ast.Name) and t.id=='source' for t in n.targets))
  # Execute exactly the unchanged source/review authentication block, with the
  # successor constant, before any selection/backend/transport callable.
  gate=ast.Module(body=fn.body[index:index+4],type_ignores=[])
  self.assertEqual([type(n).__name__ for n in gate.body],['Assign','Assign','Expr','Expr'])
  code=compile(ast.fix_missing_locations(gate),'<captured-source-binding-block>','exec')
  new=(HERE/'references/audit_saved_gpu72.py').read_bytes()
  old=(HERE/'old-audit.py').read_bytes();review=b'FABRICATED REVIEW ONLY\n'
  inputs=[('new',new,sha(new),True),('old',old,sha(old),False),
   ('altered_retained_pin',new+b'\n',sha(new),False),('altered_self_pin',new+b'\n',sha(new+b'\n'),False)]
  for label,data,expected,success in inputs:
   request=dict(gpu_audit_source=base64.b64encode(data).decode(),gpu_audit_sha256=expected,
    gpu_audit_review=base64.b64encode(review).decode(),gpu_audit_review_sha256=sha(review))
   (FIXTURES/(label+'.json')).write_text(json.dumps(request,sort_keys=True,indent=2)+'\n')
   namespace=dict(base64=base64,GPU_AUDIT_SHA256=pin,request=request,
    b=types.SimpleNamespace(need=need,sha=sha,pin=lambda s:type(s) is str and re.fullmatch('[0-9a-f]{64}',s) is not None))
   with self.subTest(case=label):
    if success:exec(code,namespace)
    else:
     with self.assertRaises(ValueError):exec(code,namespace)

if __name__=='__main__':
 FIXTURES.mkdir(mode=0o700)
 unittest.main(verbosity=2)
