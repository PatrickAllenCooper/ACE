import json, os, pathlib, runpy, unittest
root=pathlib.Path(__file__).parent
ns=runpy.run_path(str(root/'source/test_qualify_llm_intervention_screen_training_gpu.py'))
case=ns['ScreenQualifierTests']
suite=unittest.TestSuite([case('test_prepared_source_inventory_cannot_be_emptied_or_extended_to_unrelated_helpers')])
def explicit_valid_exact_preparation_inventory():
    toy=ns['ToyPacket'](pathlib.Path(os.environ['ACE_SCREEN_QUALIFIER_TEST_OUTPUT'])/'explicit_valid_positive',ns['SOURCE'].read_bytes())
    assert len(toy.prepared['source_sha256'])==18
    assert set(toy.prepared['source_sha256'])==set(ns['PREPARATION_SOURCES'])==toy.g.PREPARATION_SOURCES
    context=toy.auth()
    assert context[4]==toy.prepared
    (toy.root/'positive-result.json').write_text(json.dumps({'status':'passed','exact_preparation_sources':18,'scope':'fabricated in-process control only','actual_acceptance_qualified':False},indent=2)+'\n')
suite.addTest(unittest.FunctionTestCase(explicit_valid_exact_preparation_inventory))
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
