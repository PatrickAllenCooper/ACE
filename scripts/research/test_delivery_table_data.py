"""Fabricated table mutations check omissions, routing and estimand confusion."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from verify_delivery_table_data import audit, check_data, digest, tables, HISTORIES, CONDITIONS, TABLES, HEADERS


def tabular(rows, header):
    return ('\\begin{tabular}{lrrrrr}\n\\toprule\n'+' & '.join(header)+'\\\\\n\\midrule\n'+
            '\n'.join(' & '.join(row)+'\\\\' for row in rows)+
            '\n\\bottomrule\n\\end{tabular}\n').encode()


def fixture():
    configs, config_rows = {}, []
    for dataset, display in [('final_buffer','Final50'), ('online_admitted','Admitted'), ('all_paid','All paid')]:
        for arm in ('scm','flat'):
            for budget in (100,30000):
                for init in (0,1,2):
                    configs[f'{dataset}-{arm}-{budget}-i{init}'] = {
                        'n_histories':12,'geomean_nmse_ratio_online':0.125+init,
                        'geomean_snapped_ratio_online':0.25+init,
                        'fit_cpu_core_hours':0.0002 if budget == 100 else 0.5}
                config_rows.append([display,'SCM' if arm=='scm' else arm,f'{budget:,}',
                                    '0.125','0.250','$<0.001$' if budget==100 else '0.500'])
    for arm in ('ridge','polynomial','tree'):
        configs[f'all_paid-{arm}-100-i0'] = {'n_histories':12,'geomean_nmse_ratio_online':0.125,
            'geomean_snapped_ratio_online':0.25,'fit_cpu_core_hours':0.0002}
        config_rows.append(['All paid',arm,'--','0.125','0.250','$<0.001$'])
    configs['all_paid-flat-matched-cpu-i0'] = {'n_histories':12,'geomean_nmse_ratio_online':0.125,
        'geomean_snapped_ratio_online':0.25,'fit_cpu_core_hours':0.5}
    config_rows.append(['All paid','flat','CPU','0.125','0.250','0.500'])
    history_rows, histories = [], []
    for seed in sorted(HISTORIES):
        histories.append({'seed':seed,'online':{'nmse':0.01},'delivery':{'nmse':0.005,'exact':0.8},
                          'simpler':{'nmse':0.02},'unused_observations_ablation':{'nmse':0.025}})
        history_rows.append([str(seed),'0.010000','0.005000','0.020000','0.200','0.200'])
    physical, absolute, conditional = {}, [], []
    for condition in sorted(CONDITIONS):
        physical[condition] = {'nmse':{'delivery':0.004,'rolling_buffer':0.002,
            'physics':0.000943,'fourier':0.0001}, 'conditional_uncertainty':{}}
        # Deliberately different from ratios of row-weighted NMSE.
        for arm in ('rolling_buffer','physics','fourier'):
            physical[condition]['conditional_uncertainty'][arm] = {
                'n_action_blocks':8,'block_weighted_ratio':1.25,'conditional_bootstrap_ci95':[0.5,2.5]}
        label=condition.replace('_',r'\_')
        absolute.append([label,'0.004','0.002',r'$9.43\times10^{-4}$',r'$1.00\times10^{-4}$'])
        conditional.append([label,*['1.25 [0.50, 2.50]']*3])
    raw={TABLES[0]:tabular(config_rows, HEADERS[TABLES[0]][0]),TABLES[1]:tabular(history_rows, HEADERS[TABLES[1]][0]),
         TABLES[2]:tabular(absolute, HEADERS[TABLES[2]][0])+tabular(conditional, HEADERS[TABLES[2]][1])}
    return raw,{'configurations':configs},{'histories':histories},{'conditions':physical}


class TableDataTests(unittest.TestCase):
    def test_all_rows_and_repeated_interval_fields(self):
        result=check_data(*fixture())
        self.assertEqual(result['tables_checked'],4)
        self.assertEqual(result['metric_scalar_fields_checked'],251)
        self.assertEqual(result['numeric_epoch_fields_checked'],12)
        self.assertEqual(result['numeric_display_fields_checked'],263)

    def test_missing_duplicate_or_reordered_history_rejects(self):
        for action in ('missing','duplicate','reorder'):
            raw,s,g,p=fixture();rows=tables(raw[TABLES[1]])[0]
            if action=='missing':rows.pop()
            elif action=='duplicate':rows[-1]=rows[0]
            else:rows[0],rows[1]=rows[1],rows[0]
            raw[TABLES[1]]=tabular(rows, HEADERS[TABLES[1]][0])
            with self.subTest(action=action),self.assertRaises(ValueError):check_data(raw,s,g,p)

    def test_wrong_initialization_or_cpu_budget_rejects(self):
        raw,s,g,p=fixture();rows=tables(raw[TABLES[0]])[0]
        for column,value in ((3,'1.125'),(2,'30,000'),(5,'1800')):
            changed=copy.deepcopy(rows);changed[-1][column]=value
            mutated=dict(raw);mutated[TABLES[0]]=tabular(changed, HEADERS[TABLES[0]][0])
            with self.subTest(column=column),self.assertRaises(ValueError):check_data(mutated,s,g,p)

    def test_condition_omission_or_development_inclusion_rejects(self):
        raw,s,g,p=fixture()
        for action in ('missing','development'):
            changed=copy.deepcopy(p)
            if action=='missing':changed['conditions'].pop('red_64')
            else:changed['conditions']['white_64']=changed['conditions'].pop('white_128')
            with self.subTest(action=action),self.assertRaises(ValueError):check_data(raw,s,g,changed)

    def test_row_weighted_ratio_cannot_replace_block_summary(self):
        raw,s,g,p=fixture();absolute,conditional=tables(raw[TABLES[2]])
        conditional[0][1]='2.00 [0.50, 2.50]'
        raw[TABLES[2]]=tabular(absolute, HEADERS[TABLES[2]][0])+tabular(conditional, HEADERS[TABLES[2]][1])
        with self.assertRaises(ValueError):check_data(raw,s,g,p)

    def test_wrong_interval_quantizer_or_nonfinite_saved_value_rejects(self):
        raw,s,g,p=fixture();absolute,conditional=tables(raw[TABLES[2]])
        conditional[0][1]='1.25 [0.50, 2.51]'
        changed=dict(raw);changed[TABLES[2]]=tabular(absolute, HEADERS[TABLES[2]][0])+tabular(conditional, HEADERS[TABLES[2]][1])
        with self.assertRaises(ValueError):check_data(changed,s,g,p)
        rows=tables(raw[TABLES[1]])[0];rows[0][-1]='0.800'
        changed=dict(raw);changed[TABLES[1]]=tabular(rows, HEADERS[TABLES[1]][0])
        with self.assertRaises(ValueError):check_data(changed,s,g,p)
        s['configurations']['all_paid-scm-30000-i0']['geomean_nmse_ratio_online']=float('nan')
        with self.assertRaises(ValueError):check_data(raw,s,g,p)

    def test_hidden_rows_and_unsupported_environments_reject(self):
        raw,s,g,p=fixture()
        for corrupt in (
            raw[TABLES[0]].replace(b'\\bottomrule', b'\\bottomrule\nextra & 99\\\\'),
            raw[TABLES[0]].replace(b'{tabular}',b'{longtable}'),
            raw[TABLES[0]]+b'\\begin{tabularx}{l}\nextra & 99\\\\\n\\end{tabularx}'):
            changed=dict(raw);changed[TABLES[0]]=corrupt
            with self.subTest(corrupt=corrupt[-30:]),self.assertRaises(ValueError):
                check_data(changed,s,g,p)

    def test_header_units_and_estimand_labels_reject(self):
        raw,s,g,p=fixture()
        for name,old,new in ((TABLES[0],b'CPU h',b'CPU s'),
                             (TABLES[1],b'SCM snap',b'SCM accuracy'),
                             (TABLES[2],b'Delivery/Physics',b'Physics/Delivery')):
            changed=dict(raw);changed[name]=changed[name].replace(old,new)
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'headers or units'):
                check_data(changed,s,g,p)

    def test_pinned_snapshot_gate_and_corruption(self):
        raw,s,g,p=fixture()
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();root=base/'package';table_dir=base/'paper'
            root.mkdir();table_dir.mkdir()
            g['custody_audit']={'full_acceptance':True}
            s['custody']={'full_acceptance':True}
            artifacts={'A/complete.json':{'n_fits':480,'n_histories':12}, 'C/scores.json':p}
            blobs={name:json.dumps(obj).encode() for name,obj in artifacts.items()}
            g['complete_receipt_sha256']=digest(blobs['A/complete.json'])
            blobs['A/gate.json']=json.dumps(g).encode()
            s['complete_sha256']=digest(blobs['A/complete.json'])
            s['gate_sha256']=digest(blobs['A/gate.json'])
            blobs['A/summary.json']=json.dumps(s).encode()
            accepted={'full_acceptance':True,'conditions':sorted(CONDITIONS),
                'selected_gate_sha256':digest(blobs['A/gate.json']),
                'receipt_hashes':{'scores.json':digest(blobs['C/scores.json'])}}
            blobs['C/acceptance.json']=json.dumps(accepted).encode()
            entries=[]
            for name,data in blobs.items():
                path=root/name;path.parent.mkdir(exist_ok=True);path.write_bytes(data)
                entries.append({'path':name,'sha256':digest(data),
                    'original_sha256':digest(data),'transform':{'kind':'identity'}})
            manifest=json.dumps({'files':entries}).encode();(root/'manifest.json').write_bytes(manifest)
            for name,data in raw.items():(table_dir/name).write_bytes(data)
            pin=digest(manifest)
            self.assertEqual(audit(root,pin,table_dir)['tables_checked'],4)
            c_entry=next(e for e in entries if e['path']=='C/acceptance.json')
            c_entry['original_sha256']='1'*64
            wrong_identity=json.dumps({'files':entries}).encode()
            (root/'manifest.json').write_bytes(wrong_identity)
            with self.assertRaisesRegex(ValueError,'identity acceptance digests'):
                audit(root,digest(wrong_identity),table_dir)
            c_entry['transform']={'kind':'project-json'}
            projected_manifest=json.dumps({'files':entries}).encode()
            (root/'manifest.json').write_bytes(projected_manifest);pin=digest(projected_manifest)
            self.assertEqual(audit(root,pin,table_dir)['numeric_display_fields_checked'],263)
            # Repinning a modified manifest cannot repair inconsistent original acceptance bindings.
            scores_entry=next(e for e in entries if e['path']=='C/scores.json')
            original=scores_entry['original_sha256'];scores_entry['original_sha256']='2'*64
            wrong=json.dumps({'files':entries}).encode();(root/'manifest.json').write_bytes(wrong)
            with self.assertRaisesRegex(ValueError,'identity A/C'):
                audit(root,digest(wrong),table_dir)
            scores_entry['original_sha256']=original
            # Authenticated, individually valid receipts with inconsistent digest relationships fail.
            bad_accepted=dict(accepted);bad_accepted['selected_gate_sha256']='3'*64
            bad_bytes=json.dumps(bad_accepted).encode();(root/'C/acceptance.json').write_bytes(bad_bytes)
            c_entry['sha256']=digest(bad_bytes)
            wrong=json.dumps({'files':entries}).encode();(root/'manifest.json').write_bytes(wrong)
            with self.assertRaisesRegex(ValueError,'acceptance-to-input'):
                audit(root,digest(wrong),table_dir)
            (root/'C/acceptance.json').write_bytes(blobs['C/acceptance.json'])
            c_entry['sha256']=digest(blobs['C/acceptance.json'])
            (root/'manifest.json').write_bytes(projected_manifest)
            with patch('verify_delivery_table_data.decode',side_effect=RuntimeError('must not decode')):
                with self.assertRaisesRegex(ValueError,'manifest digest'):audit(root,'0'*64,table_dir)
            (root/'C/scores.json').write_text('malformed data that must fail authentication')
            with self.assertRaisesRegex(ValueError,'receipt snapshot digest'):audit(root,pin,table_dir)
            (root/'C/scores.json').unlink()
            (root/'C/scores.json').symlink_to(table_dir/TABLES[0])
            with self.assertRaisesRegex(ValueError,'symlink'):audit(root,pin,table_dir)


if __name__=='__main__':unittest.main()
