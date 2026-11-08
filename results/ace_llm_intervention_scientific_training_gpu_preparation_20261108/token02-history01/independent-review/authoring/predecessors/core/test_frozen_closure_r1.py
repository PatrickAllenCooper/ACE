"""One metadata-only regression: no coordinator/production import or build."""
import ast,hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).parent
class FrozenClosure(unittest.TestCase):
 def test_actual_frozen_token_closure_lookup(self):
  data=Path('/Users/pat/code/ACE/results/ace_llm_intervention_scientific_contract_20261108/candidate03/scientific-freeze.json').read_bytes()
  self.assertEqual(hashlib.sha256(data).hexdigest(),'15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d')
  f=json.loads(data)
  def expression(path):
   fn=next(n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='make_packet')
   return next(n.value for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prep' for t in n.targets))
  def lookup(n):
   if isinstance(n,ast.Name) and n.id=='f':return f
   if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) and isinstance(n.slice.value,str):return lookup(n.value)[n.slice.value]
   self.fail('non-static metadata lookup')
  with self.assertRaisesRegex(KeyError,'tokenize'):lookup(expression(ROOT/'predecessor/build_packet.py'))
  closure=lookup(expression(ROOT/'build_packet.py'))
  self.assertEqual(closure,f['stage_closure']['stages']['token'])
  self.assertEqual(len(closure['source_names']),28)
  self.assertEqual(len(closure['input_source_files']),32)
  self.assertTrue(set(closure['source_names'])<=set(closure['input_source_files']))
  for name,pin in closure['input_source_files'].items():
   origins=[h for path,h in f['source_pins'].items() if path.endswith('/'+name)]
   self.assertEqual(origins,[pin])
  self.assertEqual(next(s['remote_root'] for s in f['stages'] if s['stage']=='tokenize'),'/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01')
if __name__=='__main__':unittest.main(verbosity=2)
