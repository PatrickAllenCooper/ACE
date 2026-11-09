"""NEW artificial traversal cases only. No transport, scientific input, or remote calls."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
FIXTURES=Path(os.environ['TRAVERSAL_FIXTURES']).resolve()
FIXTURES.mkdir(exist_ok=False)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
old=load('predecessor527d',HERE/'predecessor527d/remote_custody.py')
new=load('traversal02',HERE/'remote_custody.py')
MEASUREMENTS={}

class OpenCounter:
    def __init__(self):self.calls=0;self.live=set();self.peak=0;self.open=os.open;self.close=os.close
    def opened(self,*args,**kwargs):
        fd=self.open(*args,**kwargs);self.calls+=1;self.live.add(fd);self.peak=max(self.peak,len(self.live));return fd
    def closed(self,fd):
        result=self.close(fd);self.live.discard(fd);return result
    def __enter__(self):
        self.stack=contextlib.ExitStack();self.stack.enter_context(mock.patch.object(os,'open',self.opened));self.stack.enter_context(mock.patch.object(os,'close',self.closed));return self
    def __exit__(self,*exc):return self.stack.__exit__(*exc)

class TraversalCases(unittest.TestCase):
    def root(self,name):
        p=FIXTURES/name;p.mkdir(exist_ok=False);return p

    def test_archive_inventory_hardlinks_and_open_reduction(self):
        case=self.root('equivalence');source=case/'original'/'nested'/'deep';source.mkdir(parents=True)
        for directory in ('a','a-','a.b','a0','a/deep','b','c','empty'):(source/directory).mkdir(parents=True,exist_ok=True)
        for i in range(360):
            folder=('a','a-','a.b','a0','a/deep','b','c')[i%7]
            (source/folder/f'f{i:04}.utf8').write_bytes((f'artificial-only-{i}\n'.encode())*(i%4))
        os.link(source/'a/f0000.utf8',source/'hardlink-copy.utf8')
        namespace=case/'namespace.json';namespace.write_bytes(b'{"artificial":true}\n')
        roots={'launch':str(source),'namespace.json':str(namespace)}
        facts=[]
        for label,module in (('original',old),('successor',new)):
            started=time.monotonic();cpu=time.process_time()
            with OpenCounter() as counter:
                full,ids,groups=module.scan(roots)
                archive=case/(label+'.tar');module.make_tar(archive,full,roots,ids)
            self.assertEqual(counter.live,set())
            facts.append((full,ids,groups,archive))
            MEASUREMENTS[label]={'os_open_calls_scan_plus_make_tar':counter.calls,'peak_tracked_open_fds':counter.peak,'wall_seconds':time.monotonic()-started,'process_seconds':time.process_time()-cpu}
        self.assertEqual(facts[0][:3],facts[1][:3]);self.assertEqual(facts[0][3].read_bytes(),facts[1][3].read_bytes())
        full,ids,groups,archive=facts[1]
        self.assertEqual(groups,[{'paths':['launch/a/f0000.utf8','launch/hardlink-copy.utf8'],'link_count':2}])
        self.assertEqual(old.check_tar(archive,full),new.check_tar(archive,full))
        self.assertLess(MEASUREMENTS['successor']['os_open_calls_scan_plus_make_tar'],MEASUREMENTS['original']['os_open_calls_scan_plus_make_tar']/3)
        self.assertLessEqual(MEASUREMENTS['successor']['peak_tracked_open_fds'],new.MAX_SOURCE_DEPTH+1+2)
        # Selected bytes and full independent project-style archive readback remain exact.
        copy=case/'independent-copy.tar';meta=new.check_tar(archive,full);new.copy_archive(archive,copy,meta)
        self.assertNotEqual(archive.stat().st_ino,copy.stat().st_ino)
        self.assertEqual(new.check_tar(copy,full),meta)
        captured=io.BytesIO();wanted=['launch/a-/f0001.utf8','namespace.json']
        self.assertEqual(new.check_tar(copy,full,emit=captured.write,wanted=wanted),meta)
        self.assertEqual(captured.getvalue(),b''.join((source/'a-/f0001.utf8').read_bytes() if n.startswith('launch/') else namespace.read_bytes() for n in sorted(wanted)))
        self.assertEqual(new.scan(roots),facts[1][:3])
        MEASUREMENTS['equivalence']={'file_count':len(full['files']),'directory_count':len(full['directories']),'archive_sha256':meta['sha256'],'archive_bytes':meta['bytes'],'hardlink_groups':groups,'full_inventory':full,'all_bytes_artificial':True}

    def test_symlinks_open_races_and_detached_ancestors(self):
        for mode in ('symlink_leaf','symlink_ancestor','replace_during_open','symlink_during_open','detached_ancestor','directory_membership'):
            with self.subTest(mode=mode):
                case=self.root(mode);container=case/'container';source=container/'source';source.mkdir(parents=True)
                victim=source/'victim';victim.write_bytes(b'artificial')
                original_open=os.open;original_read=os.read;original_listdir=os.listdir;triggered=False
                if mode=='symlink_leaf':(source/'alias').symlink_to(victim)
                if mode=='symlink_ancestor':
                    (case/'alias').symlink_to(container,target_is_directory=True);source=case/'alias'/'source'
                def opened(path,*args,**kwargs):
                    nonlocal triggered
                    if mode in ('replace_during_open','symlink_during_open') and path=='victim' and kwargs.get('dir_fd') is not None and not triggered:
                        triggered=True;victim.rename(source/'held-original')
                        if mode=='replace_during_open':victim.write_bytes(b'artificial')
                        else:victim.symlink_to(source/'held-original')
                    return original_open(path,*args,**kwargs)
                def read(fd,count):
                    nonlocal triggered
                    result=original_read(fd,count)
                    if mode=='detached_ancestor' and result and not triggered:
                        triggered=True;container.rename(case/'detached');(container/'source').mkdir(parents=True);(container/'source'/'victim').write_bytes(b'artificial')
                    return result
                def listdir(fd):
                    nonlocal triggered
                    result=original_listdir(fd)
                    if mode=='directory_membership' and not triggered:
                        triggered=True;(source/'new-member').write_bytes(b'artificial')
                    return result
                with OpenCounter() as counter:
                    tracked_open=os.open
                    def race_open(path,*args,**kwargs):
                        # Counter sees the actual open, even when an alias/race rejects it.
                        nonlocal original_open
                        original_open=tracked_open
                        return opened(path,*args,**kwargs)
                    with mock.patch.object(os,'open',race_open),mock.patch.object(os,'read',read),mock.patch.object(os,'listdir',listdir):
                        with self.assertRaises((ValueError,OSError)):new.scan({'launch':str(source)})
                self.assertEqual(counter.live,set())
                if mode not in ('symlink_leaf','symlink_ancestor'):self.assertTrue(triggered)

    def test_cached_archive_parent_rename_and_primary_failure_cleanup(self):
        case=self.root('archive_parent_rename');source=case/'source';branch=source/'branch';branch.mkdir(parents=True)
        (branch/'a').write_bytes(b'a');(branch/'b').write_bytes(b'b')
        roots={'launch':str(source)};full,ids,groups=new.scan(roots);original_read=os.read;triggered=False
        def read(fd,count):
            nonlocal triggered
            data=original_read(fd,count)
            if data==b'b' and not triggered:
                triggered=True;branch.rename(source/'detached');branch.mkdir();(branch/'a').write_bytes(b'a');(branch/'b').write_bytes(b'b')
            return data
        with OpenCounter() as counter, mock.patch.object(os,'read',read):
            with self.assertRaises(ValueError):new.make_tar(case/'partial.tar',full,roots,ids)
        self.assertTrue(triggered);self.assertEqual(counter.live,set());self.assertTrue((case/'partial.tar').exists())
        # An owned FD close error must not replace an existing payload-read error.
        case=self.root('read_and_close_failure');source=case/'source';source.mkdir();(source/'x').write_bytes(b'x')
        primary=RuntimeError('artificial-primary-read');secondary=False
        with OpenCounter() as counter:
            tracked_close=os.close
            def close(fd):
                nonlocal secondary
                tracked_close(fd)
                if not secondary:secondary=True;raise OSError('artificial-secondary-close')
            with mock.patch.object(os,'read',side_effect=primary),mock.patch.object(os,'close',close):
                with self.assertRaises(RuntimeError) as raised:new.scan({'launch':str(source)})
        self.assertIs(raised.exception,primary);self.assertTrue(secondary);self.assertEqual(counter.live,set())

    def test_depth_boundary_and_constructor_failure_close_all_fds(self):
        case=self.root('depth');deep=case
        for i in range(new.MAX_SOURCE_DEPTH-len(case.parts[1:])):deep=deep/'a';deep.mkdir()
        self.assertEqual(len(deep.parts[1:]),new.MAX_SOURCE_DEPTH)
        with OpenCounter() as counter:
            with new.SourceCursor(str(deep)) as cursor:
                self.assertTrue(cursor.frames)
                with self.assertRaisesRegex(ValueError,'depth bound'):cursor.push('unopened')
        self.assertEqual(counter.peak,new.MAX_SOURCE_DEPTH+1);self.assertEqual(counter.live,set())
        too_deep=deep/'a';too_deep.mkdir()
        with OpenCounter() as counter:
            with self.assertRaisesRegex(ValueError,'depth bound'):new.SourceCursor(str(too_deep))
        self.assertEqual(counter.calls,0);self.assertEqual(counter.live,set())
        case=self.root('constructor_failure');(case/'alias').symlink_to(deep,target_is_directory=True)
        with OpenCounter() as counter:
            with self.assertRaises((ValueError,OSError)):new.SourceCursor(str(case/'alias'/'absent'))
        self.assertEqual(counter.live,set())
        with OpenCounter() as counter,mock.patch.object(os,'fstat',side_effect=OSError('artificial initial fstat failure')):
            with self.assertRaises(OSError):new.SourceCursor(str(case))
        self.assertEqual(counter.calls,1);self.assertEqual(counter.live,set())

    def test_sparse_progress_and_protected_pipeline_unchanged(self):
        case=self.root('progress');source=case/'source';source.mkdir()
        for i in range(1002):(source/f'f{i:04}').write_bytes(b'')
        stderr=io.StringIO()
        with contextlib.redirect_stderr(stderr):full,ids,groups=new.scan({'launch':str(source)})
        values=[json.loads(x) for x in stderr.getvalue().splitlines()]
        self.assertEqual([v['state'] for v in values],['start','progress','complete'])
        self.assertEqual([v['members'] for v in values],[0,1000,1003])
        for v in values:
            self.assertEqual(set(v),{'phase','state','monotonic','elapsed_seconds','process_seconds','members','files','bytes'})
            self.assertGreaterEqual(v['elapsed_seconds'],0);self.assertGreaterEqual(v['process_seconds'],0)
        # Static oracle: no change to transport/authentication/capacity/receipt gates.
        def functions(path):return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        a=functions(HERE/'predecessor527d/remote_custody.py');b=functions(HERE/'remote_custody.py')
        changed={'scan','make_tar','check_tar','copy_archive','remote_collect'}
        for name,node in a.items():
            if name not in changed:self.assertEqual(ast.dump(node,include_attributes=False),ast.dump(b[name],include_attributes=False),name)
        class RemoveProgress(ast.NodeTransformer):
            def visit_With(self,node):
                node=self.generic_visit(node)
                node.items=[x for x in node.items if not (isinstance(x.context_expr,ast.Call) and isinstance(x.context_expr.func,ast.Name) and x.context_expr.func.id=='Progress')]
                return node if node.items else node.body
        self.assertEqual(ast.dump(a['remote_collect'],include_attributes=False),ast.dump(RemoveProgress().visit(b['remote_collect']),include_attributes=False))
        self.assertEqual(new.TIMEOUT,old.TIMEOUT)
        MEASUREMENTS['progress']={'events':values,'protected_original_functions_compared':sorted(set(a)-changed),'remote_collect_only_progress_wrappers':True,'transport_seconds':new.TIMEOUT,'remote_seconds':new.TIMEOUT-30}

if __name__=='__main__':
    try:unittest.main(verbosity=2)
    finally:(FIXTURES.parent/'measurements.json').write_text(json.dumps(MEASUREMENTS,sort_keys=True,indent=2)+'\n')
