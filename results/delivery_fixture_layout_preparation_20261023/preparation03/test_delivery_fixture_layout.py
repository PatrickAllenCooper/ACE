"""New layout/import-preflight fixtures; sentinel sources are never executed."""
import copy
import importlib.machinery
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_delivery_fixture_layout as layout


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.closure=self.root/'closure';self.closure.mkdir()
        model=('import os\nfrom pathlib import Path\nPROJECT = '+layout.ORIGINAL['PROJECT']+
               '\nSOURCE = '+layout.ORIGINAL['SOURCE']+'\nraise RuntimeError("SOURCE EXECUTION FORBIDDEN")\n').encode()
        sentinel=b'raise RuntimeError("SOURCE EXECUTION FORBIDDEN")\n'
        project={layout.MODEL:model,layout.IO:sentinel,'baselines.py':sentinel,
                 'experiments/large_scale_scm.py':sentinel}
        for name in ['delivery_prospective_models','delivery_prospective_design',
                     'delivery_prospective_io','runner_delivery_confirmation']:
            project['scripts/research/'+name+'.py']=sentinel
        project.update({f'scripts/research/worker{i}.py':sentinel for i in range(16)})
        learners={'ace/__init__.py':sentinel,'ace/oracle.py':sentinel}
        learners.update({f'ace/worker{i}.py':sentinel for i in range(17)})
        self.pins={n:layout.digest(project[n]) for n in (layout.MODEL,layout.IO)}
        record={'source_revision':layout.REVISION,'registration_sha256':layout.REGISTRATION,
                'outcomes_accessed':False,'project_files':{},'learner_files':{}}
        for folder,mapping,key in [('project',project,'project_files'),('source',learners,'learner_files')]:
            for n,raw in mapping.items():
                p=self.closure/folder/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
                record[key][n]=layout.digest(raw)
        self.closure_file=self.closure/'source-closure.json';self.closure_file.write_text(json.dumps(record))
        self.closure_pin=layout.digest(self.closure_file.read_bytes())
        self.locations=self.root/'locations';self.locations.mkdir()
        files={'originals/'+Path(layout.MODEL).name:model,'originals/'+Path(layout.IO).name:sentinel,
               'fixtures/'+Path(layout.MODEL).name:layout.derive_model(model),'fixtures/'+Path(layout.IO).name:sentinel}
        contract={'schema':'delivery-B-fixture-locations-v1','original_revision':layout.REVISION,
                  'registration_sha256':layout.REGISTRATION,'original_closure_sha256':self.closure_pin,
                  'location_edits':{n:{'original_expression':layout.ORIGINAL[n],
                  'derived_expression':layout.REPLACEMENTS[n]} for n in layout.ORIGINAL},
                  **{n:False for n in ['fixtures_executed','original_training_reproduction_qualified',
                                      'anonymous_execution_qualified','public_release_approved']},'files':[]}
        for n,raw in files.items():
            p=self.locations/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
            contract['files'].append({'path':n,'sha256':layout.digest(raw),'bytes':len(raw)})
        self.location_file=self.locations/'contract.json';self.location_file.write_text(json.dumps(contract))
        self.location_pin=layout.digest(self.location_file.read_bytes())
        self.output=self.root/'output'
        self.addCleanup(self.unseal)
        self.patch=patch.dict(layout.PINS,self.pins);self.patch.start();self.addCleanup(self.patch.stop)

    def unseal(self):
        if self.output.exists():
            for p in [self.output,*self.output.rglob('*')]:
                if not p.is_symlink():p.chmod(0o700 if p.is_dir() else 0o600)

    def build(self):
        self.receipt=layout.prepare(self.closure_file,self.closure_pin,
                                   self.location_file,self.location_pin,self.output)
        return self.receipt['layout_contract_sha256']

    def test_complete_source_layout_and_resolution_without_execution(self):
        pin=self.build();before=list(sys.path)
        with (patch.object(importlib.machinery.SourceFileLoader,'exec_module',side_effect=AssertionError('execute forbidden')),
              patch.object(sys,'path_hooks',[lambda path: (_ for _ in ()).throw(AssertionError('path hook forbidden'))])):
            result=layout.preflight(self.output,pin,modules={})
        self.assertEqual(sys.path,before)
        self.assertEqual(self.receipt['files'],50)
        self.assertEqual(result['files_verified'],49)
        self.assertEqual(result['resolved']['test_delivery_prospective_models']['origin'],
                         str(self.output/'fixtures/test_delivery_prospective_models.py'))
        self.assertEqual(result['resolved']['ace.oracle']['origin'],str(self.output/'source/ace/oracle.py'))
        self.assertIsNone(result['resolved']['experiments']['origin'])
        self.assertFalse(result['runtime_qualified']);self.assertFalse(result['fixtures_executed'])
        self.assertEqual((self.output/'project'/layout.MODEL).read_bytes(),
                         (self.closure/'project'/layout.MODEL).read_bytes())
        with self.assertRaises(ValueError):layout.prepare(self.closure_file,self.closure_pin,
            self.location_file,self.location_pin,self.output)

    def test_all_reserved_cached_names_reject_before_lookup(self):
        pin=self.build()
        for name in ['ace','ace.oracle','ace.unknown','experiments','experiments.large_scale_scm',
                     'delivery_audited_large_scm','test_delivery_prospective_models',
                     'delivery_prospective_models','runner_delivery_confirmation']:
            with self.subTest(name=name),patch.object(importlib.machinery.FileFinder,'find_spec',
                    side_effect=AssertionError('lookup before cache rejection')):
                with self.assertRaisesRegex(ValueError,'cached source module'):
                    layout.preflight(self.output,pin,modules={name:None})

    def test_corruption_extra_bytecode_and_directory_shadow_reject(self):
        pin=self.build();self.unseal()
        target=self.output/'source/ace/oracle.py';raw=target.read_bytes();target.write_bytes(raw+b'#changed')
        with self.assertRaisesRegex(ValueError,'digest'):layout.preflight(self.output,pin,modules={})
        target.write_bytes(raw)
        extra=self.output/'fixtures/unknown.pyc';extra.write_bytes(b'not executable')
        with self.assertRaisesRegex(ValueError,'membership'):layout.preflight(self.output,pin,modules={})
        extra.unlink();shadow=self.output/'fixtures/ace';shadow.mkdir()
        with self.assertRaisesRegex(ValueError,'unexpected source'):layout.preflight(self.output,pin,modules={})

    def test_relative_root_symlink_and_bad_pin_reject(self):
        pin=self.build()
        for root,p in [('relative',pin),(self.output,'0'*64),(self.output,'HEAD')]:
            with self.subTest(root=root,p=p),self.assertRaises(ValueError):layout.preflight(root,p,modules={})
        alias=self.root/'alias';alias.symlink_to(self.output,target_is_directory=True)
        with self.assertRaises(ValueError):layout.preflight(alias,pin,modules={})

    def test_predecessor_disagreement_rejects_before_output(self):
        value=json.loads(self.location_file.read_text());value['original_closure_sha256']='0'*64
        self.location_file.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'lineage'):
            layout.prepare(self.closure_file,self.closure_pin,self.location_file,
                           layout.digest(self.location_file.read_bytes()),self.output)
        self.assertFalse(self.output.exists())

    def test_rehashed_contract_cannot_omit_original_or_change_import_plan(self):
        self.build();self.unseal();path=self.output/'layout-contract.json';original=json.loads(path.read_text())
        variants=[]
        v=copy.deepcopy(original);v['modules']['ace.oracle']='source/ace/worker0.py';variants.append(v)
        v=copy.deepcopy(original);v['required_python']='3.12';variants.append(v)
        v=copy.deepcopy(original);v['files'].pop('originals/'+Path(layout.IO).name);variants.append(v)
        for value in variants:
            path.write_text(json.dumps(value))
            with self.subTest(value=value),self.assertRaises(ValueError):
                layout.preflight(self.output,layout.digest(path.read_bytes()),modules={})


if __name__=='__main__':unittest.main(verbosity=2)
