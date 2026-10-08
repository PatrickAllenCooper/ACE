"""Fabricated reporting data only; no selected world, fit, model or response."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_delivery_prospective_supplement as reporting
from audit_delivery_prospective_results import primary_statistics, secondary_summary


def fixture():
    cells={}; primary=[]
    for case in reporting.WORLDS:
        world=int(case.split(':')[1]);size=int(case.split(':')[0])
        for hi,history in enumerate(reporting.HISTORIES):
            variance=(world+1)*(hi+1)/7
            for arm,init in reporting.ARMS:
                nmse=(world+1)*(hi+1)*{'delivery':.02,'online':.05,'simpler':.04,'ablation':.12}[arm]*(1+init/10)
                # Unequal histories, not a ratio-of-average-errors fixture.
                if arm=='online': nmse *= (hi+1)*2
                if world==0 and arm=='delivery':nmse=0.
                cell={'index':len(cells),'case':case,'graph_size':size,'history':history,'arm':arm,'init':init}
                cells[str(len(cells))]={'cell':cell,'metric':{'nmse':nmse,'mse':nmse*variance,'training_target_variance':variance,'evaluation_rows':400}}
                if init==0 and arm in ('delivery','online','simpler'):
                    primary.append({'graph_size':size,'system_id':case,'history':history,'arm':arm,'init':0,'status':'complete','nmse':nmse})
    stats=primary_statistics(primary)
    secondary=secondary_summary([(v['cell'],v['metric']['nmse']) for v in cells.values()])
    return {'cells':cells,'primary_analysis':stats},stats,secondary


class SupplementTests(unittest.TestCase):
    def test_every_identity_absolute_value_and_floor_retained(self):
        score,primary,secondary=fixture();r=reporting.describe(score,primary,secondary)
        self.assertEqual(len(r['cells']),640);self.assertEqual(len(r['fixed_init0']),320)
        self.assertEqual(len({(v['system'],v['history']) for v in r['fixed_init0']}),80)
        self.assertEqual(len(r['system_log_ratios']),80)
        self.assertEqual(len(r['history_specific_ratios']),8)
        self.assertEqual(len(r['short_fit_ablation']),2)
        self.assertEqual([v['system'] for v in r['system_log_ratios'][:20]],reporting.WORLDS[:20])
        for v in r['cells']:
            index=int(v['selector'].split('.')[1]);old=score['cells'][str(index)]
            self.assertEqual((v['system'],v['history'],v['arm'],v['init']),tuple(old['cell'][k] for k in ('case','history','arm','init')))
            self.assertEqual(v['mse'],old['metric']['mse']);self.assertEqual(v['nmse'],old['metric']['nmse'])
        self.assertEqual(r['cells'][0]['nmse'],0.)
        self.assertTrue(r['cells'][0]['floor_active'])
        self.assertEqual(r['primary_floor_counts'][0]['delivery'],2)
        ratio=r['system_log_ratios'][1]['ratio']
        d=[v['nmse'] for v in r['fixed_init0'] if v['system']=='5:01' and v['arm']=='delivery']
        c=[v['nmse'] for v in r['fixed_init0'] if v['system']=='5:01' and v['arm']=='online']
        self.assertNotAlmostEqual(ratio,sum(d)/sum(c))
        self.assertEqual(r['additional_significance_tests'],0)

    def test_incomplete_duplicate_reordered_boolean_nonfinite_and_bad_normalization(self):
        score,primary,secondary=fixture()
        for change in ['missing','duplicate','init','graph','nan','negative','normalizer','variance','rows']:
            altered=copy.deepcopy(score)
            if change=='missing':del altered['cells']['639']
            elif change=='duplicate':altered['cells']['1']['cell']=copy.deepcopy(altered['cells']['0']['cell'])
            elif change=='init':altered['cells']['0']['cell']['init']=False
            elif change=='graph':altered['cells']['0']['cell']['graph_size']=True
            elif change=='nan':altered['cells']['0']['metric']['nmse']=float('nan')
            elif change=='negative':altered['cells']['0']['metric']['mse']=-1
            elif change=='normalizer':altered['cells']['1']['metric']['nmse']*=2
            elif change=='variance':altered['cells']['0']['metric']['training_target_variance']=0
            elif change=='rows':altered['cells']['0']['metric']['evaluation_rows']=True
            with self.subTest(change=change),self.assertRaises(ValueError):reporting.describe(altered,primary,secondary)

    def test_changed_registered_logs_floor_aggregate_or_secondary_reject(self):
        score,primary,secondary=fixture()
        for change in ['logs','floor','order','ablation','history','count']:
            p,s=copy.deepcopy(primary),copy.deepcopy(secondary)
            if change=='logs':p['contrasts'][0]['system_log_ratios']['5:00']+=.001
            elif change=='floor':p['contrasts'][0]['floor_activations']['delivery']=0
            elif change=='order':p['contrasts'].reverse()
            elif change=='ablation':s['strata']['5']['delivery_vs_short_fit']['ratio']*=2
            elif change=='history':s['strata']['5']['history_specific_primary_ratios']['matched_random']['online']*=2
            elif change=='count':p['contrasts'][0]['n_systems']=19
            with self.subTest(change=change),self.assertRaises(ValueError):reporting.describe(score,p,s)

    def test_snapshot_digest_duplicates_nonfinite_and_symlinks(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder).resolve()/'data.json'
            for raw in [b'{"x":1,"x":2}',b'{"x":NaN}']:
                p.write_bytes(raw)
                with self.assertRaises(ValueError):reporting.snapshot(p,hashlib.sha256(raw).hexdigest())
            p.write_bytes(b'{}')
            with self.assertRaises(ValueError):reporting.snapshot(p,'0'*64)
            link=Path(folder).resolve()/'link';link.symlink_to(p)
            with self.assertRaises(ValueError):reporting.snapshot(link,hashlib.sha256(p.read_bytes()).hexdigest())

    def test_positive_export_preserves_all_rows_hashes_and_refuses_overwrite(self):
        # Only original metadata gate is substituted for fabricated contract.
        # Full replay/runtime here are fixture claims, not actual qualification.
        score,primary,secondary=fixture()
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder).resolve();root=base/'root';(root/'B').mkdir(parents=True)
            def put(path,data):
                raw=json.dumps(data).encode();path.write_bytes(raw);return hashlib.sha256(raw).hexdigest()
            files=[]
            for name,data in [('B/replay_contract.json',{'registration_sha256':reporting.gate.__globals__['REGISTRATION']}),
                              ('B/scores.json',score),('B/acceptance.json',{'primary_analysis':primary,'secondary_descriptive_analysis':secondary}),
                              ('B/retained_attempt_sentinel.json',{'fixture_only':True,'attempts':[{'status':'interrupted','returned_responses':'unknown'},{'status':'failed','preserve':True}]})]:
                h=put(root/name,data);files.append({'path':name,'bytes':(root/name).stat().st_size,'sha256':h,'original_sha256':h,'transform':{'kind':'identity'}})
            mh=put(root/'manifest.json',{'schema':'delivery-anonymous-derived-v1','files':files})
            (root/'manifest.sha256').write_text(mh)
            receipt=base/'replay.json';rh=put(receipt,{'full_supplemental_replay':True,'runtime':reporting.DEPENDENCIES,
                'checkpoints_replayed':640,'primary_cells_recomputed':240,'cached_responses_checked':48000,
                'new_optimizer_updates':0,'new_responses':0,'integrity':{'manifest_sha256':mh,'expected_digest_supplied':True},
                'primary_analysis':primary,'secondary_descriptive_analysis':secondary})
            def inventory():return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
            before=inventory();out=base/'output'
            alias=base/'alias';alias.symlink_to(root,target_is_directory=True)
            for destination in [alias/'new-output',base/'outside'/'..'/'root'/'new-output',root/'new-output']:
                with self.subTest(destination=str(destination)),self.assertRaisesRegex(ValueError,'exclusive output'):
                    reporting.export(root,mh,receipt,rh,destination)
                self.assertEqual(inventory(),before)
            with patch.object(reporting,'gate'):
                result=reporting.export(root,mh,receipt,rh,out)
                with self.assertRaises(FileExistsError):reporting.export(root,mh,receipt,rh,out)
            self.assertEqual(inventory(),before)
            self.assertEqual(json.loads((root/'B/retained_attempt_sentinel.json').read_text())['attempts'][0]['status'],'interrupted')
            self.assertEqual(len(result['fixed_init0']),320)
            import csv
            with (out/'fixed_init0.csv').open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),320)
            self.assertEqual({(v['system'],v['history'],v['arm']) for v in rows},
                {(c,h,a) for c in reporting.WORLDS for h in reporting.HISTORIES for a in ('delivery','online','simpler','ablation')})
            for name,h in json.loads((out/'output_index.json').read_text())['output_hashes'].items():
                self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(),h)
            tex=(out/'delivery_prospective_absolute_table.tex').read_text()
            for case in reporting.WORLDS:self.assertEqual(tex.count(case+' & '),4)
            self.assertEqual(tex.count('\\begin{table}'),8)
            descriptive=(out/'delivery_prospective_descriptive_table.tex').read_text()
            self.assertEqual(descriptive.count('\\begin{table}'),7)
            for case in reporting.WORLDS:self.assertEqual(descriptive.count(case+' & '),2)
            # A changed receipt cannot decode scores or create another output.
            changed=base/'changed.json';bad=json.loads(receipt.read_text());bad['integrity']['manifest_sha256']='0'*64
            bh=put(changed,bad)
            with patch.object(reporting,'gate'),self.assertRaisesRegex(ValueError,'supplemental replay'):
                reporting.export(root,mh,changed,bh,base/'bad')
            self.assertFalse((base/'bad').exists())

    def test_invalid_upstream_and_replay_gate_precede_score_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()/'root';root.mkdir();(root/'B').mkdir();out=Path(folder).resolve()/'out'
            def put(path,data):
                raw=json.dumps(data).encode();path.write_bytes(raw);return hashlib.sha256(raw).hexdigest()
            ch=put(root/'B/replay_contract.json',{})
            manifest={'files':[{'path':'B/replay_contract.json','sha256':ch}]}
            mh=put(root/'manifest.json',manifest)
            replay=Path(folder).resolve()/'replay.json';rh=put(replay,{'full_supplemental_replay':False})
            with patch.object(reporting,'gate',side_effect=ValueError('upstream gate')):
                with self.assertRaisesRegex(ValueError,'upstream'):reporting.export(root,mh,replay,rh,out)
            with patch.object(reporting,'gate'):
                with self.assertRaisesRegex(ValueError,'supplemental replay'):reporting.export(root,mh,replay,rh,out)
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
