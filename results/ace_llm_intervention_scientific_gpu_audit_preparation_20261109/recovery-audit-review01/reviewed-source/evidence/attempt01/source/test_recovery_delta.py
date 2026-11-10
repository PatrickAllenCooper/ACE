"""NEW fabricated metadata cases only; never import the aid, producers or old tests."""
import copy
import importlib.util
import json
from pathlib import Path
import types
import unittest

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('new_recovery_delta',ROOT/'dependencies/gpu_recovery_delta.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
H='a'*64;K='b'*64


class RecoveryDelta(unittest.TestCase):
    def setUp(self):
        self.folder=ROOT/'evidence/attempt01/fixtures'/self._testMethodName
        self.folder.mkdir(parents=True,exist_ok=False)

    def capture(self,name,value):
        (self.folder/(name+'.json')).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')

    def rejected(self,name,value,call):
        self.capture(name,value)
        with self.assertRaises((ValueError,KeyError,TypeError)):
            call()

    def test_only_three_operational_exceptions_preserve_science(self):
        old={r.ENTRY:r.ORIGINAL_ENTRY,'scientific.py':H}
        closure={'source_names':sorted(old),'input_source_files':old}
        packet={'source_names':sorted(set(old)|{r.CHECKER,r.HISTORY}),'files':{**old,**r.OPERATIONAL}}
        authority={'operational_source_files':packet['files']}
        context={'source_files_sha256':r.digest(packet['files'])}
        manifest={'schema':'ace-gpu02-operational-sources-v1','source_commit':'1'*40,
                  'files':{n:{'repository_path':'sources/'+n,'sha256':h} for n,h in r.OPERATIONAL.items()}}
        value={'packet':packet,'closure':closure,'authority':authority,'context':context,'manifest':manifest,'freeze_pins':old}
        self.capture('positive',value)
        self.assertEqual(r.validate_sources(**value),packet['files'])
        for name,change in [
            ('old_entry',lambda v:v['packet']['files'].__setitem__(r.ENTRY,r.ORIGINAL_ENTRY)),
            ('science_changed',lambda v:v['packet']['files'].__setitem__('scientific.py',K)),
            ('helper_changed',lambda v:v['packet']['files'].__setitem__(r.HISTORY,K)),
            ('authority_drift',lambda v:v['authority'].__setitem__('operational_source_files',{})),
            ('context_drift',lambda v:v['context'].__setitem__('source_files_sha256',K)),
            ('uncommitted_manifest',lambda v:v['manifest'].__setitem__('source_commit','HEAD')),
            ('manifest_path_escape',lambda v:v['manifest']['files'][r.ENTRY].__setitem__('repository_path','../entry.py')),
        ]:
            v=copy.deepcopy(value);change(v);self.rejected(name,v,lambda:r.validate_sources(**v))

    def test_full_old_root_and_directories_only_bounded_appends(self):
        old={'roots':{'launch':r.OLD,'cpu':'/scratch/alpine/paco0228/ACE/results/cpu',
                      'namespace.json':'/scratch/alpine/paco0228/ACE/ns.json'},
             'full_inventory':{'files':{'launch/inputs/original.py':{'bytes':10,'sha256':H},
                  r.CAMPAIGN+'policy-00/receipt.json':{'bytes':20,'sha256':K},'namespace.json':{'bytes':1,'sha256':H}},
                  'directories':['launch','launch/inputs','launch/empty','cpu','cpu/cpu','cpu/cpu/worker',
                      'cpu/cpu/worker/artifacts',r.CAMPAIGN.rstrip('/'),r.CAMPAIGN+'policy-00']}}
        current=copy.deepcopy(old);current['roots']['control']=r.CONTROL
        current['full_inventory']['files'].update({'control/packet.json':{'bytes':4,'sha256':K},
            'launch/gpu/terminal.json':{'bytes':3,'sha256':H},r.CAMPAIGN+'policy-05/attempt.json':{'bytes':2,'sha256':K}})
        current['full_inventory']['directories']+=['control','launch/gpu',r.CAMPAIGN+'policy-05']
        value={'previous':old,'current':current};self.capture('positive',value);r.inventory_extension(**value)
        for name,change in [
            ('changed_old_file',lambda v:v['current']['full_inventory']['files']['launch/inputs/original.py'].__setitem__('bytes',11)),
            ('missing_old_file',lambda v:v['current']['full_inventory']['files'].pop('namespace.json')),
            ('missing_empty_directory',lambda v:v['current']['full_inventory']['directories'].remove('launch/empty')),
            ('old_input_append',lambda v:v['current']['full_inventory']['files'].__setitem__('launch/inputs/new.py',{'bytes':1,'sha256':H})),
            ('cpu_append',lambda v:v['current']['full_inventory']['files'].__setitem__(r.CAMPAIGN+'policy-00/new.json',{'bytes':1,'sha256':H})),
            ('control_alias',lambda v:v['current']['roots'].__setitem__('control',r.OLD)),
            ('old_root_swap',lambda v:v['current']['roots'].__setitem__('launch',r.CONTROL)),
            ('preexisting_gpu',lambda v:v['previous']['full_inventory']['directories'].append('launch/gpu')),
        ]:
            v=copy.deepcopy(value);change(v);self.rejected(name,v,lambda:r.inventory_extension(**v))

    def test_complete_cpu_post_maps_reject_loss_drift_and_service(self):
        expected={'cpu_output':{'files':{'worker/terminal.json':{'bytes':3,'sha256':H}},'directories':['empty','worker','worker/artifacts']},
                  'campaign':{'files':{'campaign.json':{'bytes':2,'sha256':K},'policy-00/receipt.json':{'bytes':1,'sha256':H}},
                              'directories':['policy-00','policy-00/empty']}}
        full={'files':{},'directories':[]}
        for role,prefix in [('cpu_output','cpu/cpu/'),('campaign',r.CAMPAIGN)]:
            full['files'].update({prefix+n:v for n,v in expected[role]['files'].items()})
            full['directories'] += [prefix+n for n in expected[role]['directories']]
        full['directories'] += [r.CAMPAIGN.rstrip('/'),r.CAMPAIGN+'policy-05']
        full['files'][r.CAMPAIGN+'policy-05/new.json']={'bytes':5,'sha256':H}
        self.capture('positive',{'expected':expected,'full':full});self.assertEqual(r.cpu_post_maps(expected,full),expected)
        for name,change in [
            ('missing_file',lambda v:v['files'].pop('cpu/cpu/worker/terminal.json')),
            ('size_drift',lambda v:v['files']['cpu/cpu/worker/terminal.json'].__setitem__('bytes',4)),
            ('lost_empty_directory',lambda v:v['directories'].remove('cpu/cpu/empty')),
            ('cpu_service',lambda v:v['directories'].append(r.CAMPAIGN+'policy-00/service')),
            ('extra_cpu_file',lambda v:v['files'].__setitem__('cpu/cpu/new.json',{'bytes':2,'sha256':K})),
        ]:
            v=copy.deepcopy(full);change(v);self.rejected(name,{'expected':expected,'full':v},lambda:r.cpu_post_maps(expected,v))

    def test_actual_48_98_extension_preserves_failed47(self):
        rows=[[str(10000000+i),'prior','COMPLETED'] for i in range(46)]+[['35498218','gpu01','FAILED']]
        journals={f'/prior/journal-{i:03d}.json':H for i in range(94)}
        journals.update({r.OLD+'/submission.intent.json':H,r.OLD+'/submission.result.json':K})
        baseline={'terminal_jobs':47,'submission_journals':96,'related_accounting':rows,'journal_pins':journals}
        job='99999998';current=rows+[[job,'gpu02','COMPLETED']]
        alljournals={**journals,r.CONTROL+'/submission.intent.json':K,r.CONTROL+'/submission.result.json':H}
        value={'baseline':baseline,'rows':current,'journals':alljournals,'job':job,'intent_pin':K,'result_pin':H}
        normalize=lambda value:sorted(value)
        self.capture('positive',value);r.history_extension(**value,normalize=normalize)
        for name,change in [
            ('old_baseline_count',lambda v:v['baseline'].__setitem__('terminal_jobs',46)),
            ('old_journal_count',lambda v:v['baseline'].__setitem__('submission_journals',94)),
            ('lost_failed_job',lambda v:v['rows'].pop(46)),
            ('duplicate_current',lambda v:v['rows'].append(v['rows'][-1])),
            ('changed_prior_journal',lambda v:v['journals'].__setitem__(r.OLD+'/submission.result.json',H)),
            ('old_root_new_journal',lambda v:v['journals'].__setitem__(r.OLD+'/submission.intent.json',K)),
        ]:
            v=copy.deepcopy(value);change(v);self.rejected(name,v,lambda:r.history_extension(**v,normalize=normalize))

    def test_v2_terminal_requires_separate_postcustody_marker(self):
        files={r.RECOVERY_FILES['authority']:H,r.RECOVERY_FILES['receipt']:K}
        scoped={'scope':'independently_pinned_prelaunch_metadata_consistency_only','acceptance':False}
        terminal={'operational_recovery_authority_sha256':H,'historical_preflight_receipt_sha256':K,
                  'historical_postcheck_performed_on_gpu':False,'full_external_postcustody_required':True}
        marker={'preflight':scoped,'authority_sha256':H,'historical_postcheck_performed_on_gpu':False,
                'full_external_postcustody_required':True,'acceptance':False}
        def check(value):
            view=types.SimpleNamespace(t=value['terminal'],p={'files':files},preflight=scoped,j=lambda n:value['marker'])
            r.recovery_terminal(view)
        value={'terminal':terminal,'marker':marker};self.capture('positive',value);check(value)
        for name,change in [
            ('gpu_claimed_postcheck',lambda v:v['terminal'].__setitem__('historical_postcheck_performed_on_gpu',True)),
            ('postcheck_optional',lambda v:v['terminal'].__setitem__('full_external_postcustody_required',False)),
            ('authority_drift',lambda v:v['terminal'].__setitem__('operational_recovery_authority_sha256',K)),
            ('preflight_drift',lambda v:v['terminal'].__setitem__('historical_preflight_receipt_sha256',H)),
            ('marker_acceptance',lambda v:v['marker'].__setitem__('acceptance',True)),
        ]:
            v=copy.deepcopy(value);change(v);self.rejected(name,v,lambda:check(v))


if __name__=='__main__':
    unittest.main(verbosity=2)
