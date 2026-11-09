"""New GPU02 root/history metadata only; no real inputs, transports or collector main."""
import ast,copy,importlib.util,json,pathlib,unittest
HERE=pathlib.Path(__file__).resolve().parent

def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=load('collector02',HERE/'authoring/pull_terminal.py')
f=load('recovery_fixtures',HERE/'test_recovery_coordinator.py')

class CollectorDelta(unittest.TestCase):
    def test_preserve_old_failure_except_gpu_append(self):
        preserved={'launch':{'packet.json':f.info('old packet'),'logs/failure.err':f.info('old failure')},
                   'cpu':{'cpu/original':f.info('accepted CPU')}}
        dirs={'launch':['logs'],'cpu':['cpu']}
        actual={'files':{'launch/packet.json':f.info('old packet'),'launch/logs/failure.err':f.info('old failure'),
                         'launch/gpu/terminal.json':f.info('new GPU terminal'), 'control/packet.json':f.info('new packet'),
                         'cpu/cpu/original':f.info('accepted CPU'),
                         'cpu/cpu/worker/artifacts/campaign/policy-05/new.json':f.info('new attempt')},
                'directories':['launch','launch/logs','launch/gpu','control','cpu','cpu/cpu',
                               'cpu/cpu/worker/artifacts/campaign/policy-05']}
        self.assertEqual(c.upstream_disagreements(actual,preserved,dirs),[])
        for action in ('change_old','remove_old_dir','wrong_append','false_relabel'):
            x=copy.deepcopy(actual)
            if action=='change_old':x['files']['launch/packet.json']=f.info('new packet')
            if action=='remove_old_dir':x['directories'].remove('launch/logs')
            if action=='wrong_append':x['files']['launch/replacement.json']=f.info('forbidden')
            if action=='false_relabel':x['files']['control/logs/failure.err']=x['files'].pop('launch/logs/failure.err')
            with self.subTest(action=action):self.assertTrue(c.upstream_disagreements(x,preserved,dirs))

    def test_47_96_to_48_98_and_no_cost_relabel(self):
        m,_=f.fixture();base=m['gpu-failure-baseline'];job='99999999'
        row=[job,'ace_llm_scientific_campaign_gpu02','COMPLETED','0:0','4','3','ucb736_asc2','cpu=4,mem=15G,gres/gpu:a100_3g.20gb=1','']
        obs={'related_accounting':base['related_accounting']+[row],'parent_accounting':row,
             'journal_pins':{**base['journal_pins'],c.ROOT+'/submission.intent.json':f.H('new intent'),c.ROOT+'/submission.result.json':f.H('new result')}}
        self.assertEqual(c.accounting_extension(obs,base,job,f.H('new intent'),f.H('new result')),(True,True))
        self.assertEqual(c.cost(obs['related_accounting']), (base['closed_cpu_seconds']+12,base['closed_gpu_slice_seconds']+3))
        x=copy.deepcopy(obs);x['related_accounting']=[r for r in x['related_accounting'] if r[0]!=f.b.FAILED_GPU_JOB]
        self.assertFalse(c.accounting_extension(x,base,job,f.H('new intent'),f.H('new result'))[0])
        x=copy.deepcopy(obs);x['journal_pins'].pop(f.b.OLDROOT+'/submission.intent.json')
        self.assertFalse(c.accounting_extension(x,base,job,f.H('new intent'),f.H('new result'))[1])
        remote=ast.parse(c.REMOTE_OBSERVATION)
        loops=[n for n in ast.walk(remote) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [t.id for t in n.target.elts if isinstance(t,ast.Name)]==['label','tree','present']]
        self.assertEqual(len(loops),1)
        labels=[n.elts[0].value for n in loops[0].iter.elts]
        self.assertEqual(labels.count('launch'),1);self.assertEqual(labels.count('control'),1)
        self.assertIn("('launch',Path(OLDROOT)",c.REMOTE_OBSERVATION)
        self.assertIn("('control',root",c.REMOTE_OBSERVATION)

    def test_reconciliation_types_and_original_alias_metadata(self):
        m,p=f.fixture();self.assertTrue(f.b.failed_gpu_history(m,p))
        x=copy.deepcopy(m);x['gpu-failure-acceptance']=dict(x['gpu-failure-acceptance'],required_findings=False)
        with self.assertRaises(ValueError):f.b.failed_gpu_history(x,p)
        x=copy.deepcopy(m);o=json.loads(x['gpu-failure-inventory-after']['stdout']);o['original_hardlink_groups']=[{'paths':['launch/one','launch/two'],'link_count':2}];x['gpu-failure-inventory-after']['stdout']=json.dumps(o)
        with self.assertRaises(ValueError):f.b.failed_gpu_history(x,p)
        self.assertEqual(c.BUILDER_SHA,c.sha((HERE/'authoring/build_packet.py').read_bytes()))

if __name__=='__main__':unittest.main(verbosity=2)
