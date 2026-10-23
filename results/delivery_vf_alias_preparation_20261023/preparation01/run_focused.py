import sys, unittest, importlib.util
sys.path.insert(0,'/Users/pat/code/ACE/scripts/research')
import test_delivery_vf_proxy as t
p='/Users/pat/code/ACE/results/delivery_vf_replay_20261023/launch07/qualify_runtime.py'
spec=importlib.util.spec_from_file_location('failed07',p);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
c=t.VFProxy();c.setUp()
try:
 c.alias_fixture()
 with t.patch.dict(sys.modules,c.modules):
  old.capture_vf_import_binding()
  try:old.torch_vf_proxy(c.proxy,c.prefix,c.native,c.synthetic)
  except ValueError as exc:
   assert str(exc)=='exact VF source pin required';print('frozen07 alias failure reproduced with fabricated modules')
  else:raise AssertionError('frozen07 should reject alias')
finally:c.doCleanups()
names=['test_site_alias_accepts_same_source_native_and_live_filename','test_site_alias_duplicate_wrong_digest_and_external_rejected','test_site_alias_rejects_leaf_symlink_and_false_live_filename','test_site_alias_early_pin_check_needs_no_dependency_import','test_same_filename_changed_code_rejected']
r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(t.VFProxy(n) for n in names))
sys.exit(not r.wasSuccessful())
