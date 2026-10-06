import re
import tempfile
from pathlib import Path
import unittest

from audit_delivery_prospective_results import WORLD_IDS,HISTORIES,ARMS,primary_statistics,secondary_summary
from export_delivery_prospective_claims import export
from runner_delivery_confirmation import read,write,sha


def fixture(out):
    """Analytic receipt fixture only: no fit, simulator, or selected outcome."""
    out.mkdir()
    reg={'source_revision':'fixture-not-an-experiment','scope':'analytic placeholder; not prospective results'}
    write(out/'registration.json',reg);write(out/'fit_seal.json',{'fixture':True})
    cells={};primary=[]
    for case in WORLD_IDS:
        for history in HISTORIES:
            for arm,init in ARMS:
                c={'case':case,'graph_size':int(case.split(':')[0]),'history':history,'arm':arm,'init':init}
                value={'delivery':2.,'online':4.,'simpler':3.,'ablation':8.}[arm]/(init+1)
                cells[str(len(cells))]={'cell':c,'metric':{'nmse':value}}
                if init==0 and arm in ('delivery','online','simpler'):
                    primary.append({'graph_size':c['graph_size'],'system_id':case,'history':history,
                        'arm':arm,'init':0,'status':'complete','nmse':value})
    stats=primary_statistics(primary)
    write(out/'scores.json',{'cells':cells,'primary_rows':primary,'primary_analysis':stats})
    write(out/'complete.json',{'n_fits':640,'n_primary_cells':240,'charged_responses':48000})
    acceptance={'full_acceptance':True,'fits_checked':640,'checkpoints_replayed':640,'primary_cells_checked':240,
        'charged_cached_responses':48000,'new_simulator_responses':0,'source_revision':reg['source_revision'],
        'study_registration_sha256':sha(out/'registration.json'),'scores_sha256':sha(out/'scores.json'),
        'complete_sha256':sha(out/'complete.json'),'fit_seal_sha256':sha(out/'fit_seal.json'),
        'primary_analysis':stats,'secondary_descriptive_analysis':secondary_summary([(v['cell'],v['metric']['nmse']) for v in cells.values()]),
        'auditor_sha256':'analytic fixture','scope':reg['scope']}
    write(out/'acceptance.json',acceptance)
    write(out/'audit_execution.json',{'status':'complete','exit_code':0,
        'registration_sha256':sha(out/'registration.json'),'acceptance_sha256':sha(out/'acceptance.json')})


class ProspectiveClaimTests(unittest.TestCase):
    def test_macros_tables_and_every_numeric_macro_have_receipt_selectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);results=root/'fixture';fixture(results)
            out=root/'export';macros=export(results,out)
            self.assertAlmostEqual(float(macros['BFiveOnlineRatio']),.5)
            self.assertAlmostEqual(float(macros['BThirtyFlatRatio']),2/3,places=6)
            index=read(out/'claim_index.json')
            self.assertEqual(set(macros),set(index['macro_provenance']))
            self.assertEqual(index['deployment_initialization'],0)
            self.assertFalse(index['supervision_and_compute_matched'])
            self.assertTrue(all(re.fullmatch('[A-Za-z]+',n) for n in macros))
            self.assertEqual(set(index['output_hashes']),{'delivery_prospective_primary_table.tex',
                'delivery_prospective_sensitivity_table.tex','delivery_prospective_claims.tex'})
            with self.assertRaises(FileExistsError):export(results,out)

    def test_missing_audit_or_failed_runtime_cannot_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);results=root/'fixture';fixture(results)
            execution=read(results/'audit_execution.json');execution['status']='time_limit'
            write(results/'audit_execution.json',execution)
            with self.assertRaises(ValueError):export(results,root/'export')
            self.assertFalse((root/'export').exists())

    def test_tampered_scores_and_incomplete_matrix_cannot_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);results=root/'fixture';fixture(results)
            scores=read(results/'scores.json');scores['cells']['0']['metric']['nmse']=100.
            write(results/'scores.json',scores)
            with self.assertRaises(ValueError):export(results,root/'export')
            # Updating only the score hash still cannot conceal mismatched audit.
            a=read(results/'acceptance.json');a['scores_sha256']=sha(results/'scores.json')
            write(results/'acceptance.json',a)
            e=read(results/'audit_execution.json');e['acceptance_sha256']=sha(results/'acceptance.json')
            write(results/'audit_execution.json',e)
            with self.assertRaises(ValueError):export(results,root/'export')
            self.assertFalse((root/'export').exists())


if __name__=='__main__':unittest.main()
