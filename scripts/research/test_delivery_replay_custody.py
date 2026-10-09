"""Focused fabricated cleanup and real temporary-directory descriptor checks."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


s = module('delivery_replay_supervision')
g = module('delivery_replay_checkpoint_guard')


class Custody(unittest.TestCase):
    def test_exited_leader_still_kills_group_and_reaps_detached_child(self):
        child = Mock(pid=1234)
        child.poll.return_value = 0
        with patch.object(s.os, 'killpg') as group, patch.object(s.os, 'kill') as kill, \
             patch.object(s, 'direct_children', side_effect=[{2345}, set()]), \
             patch.object(s.os, 'waitpid', side_effect=[(2345, 9), ChildProcessError()]):
            receipt = s.cleanup_owned(child, s.time.monotonic()+5)
        group.assert_called_once_with(1234, s.signal.SIGKILL)
        kill.assert_called_once_with(2345, s.signal.SIGKILL)
        self.assertTrue(receipt['complete'])
        self.assertEqual(receipt['descendants_reaped'], [{'pid':2345,'status':9}])

    def test_missing_group_is_not_failure(self):
        child = Mock(pid=1234)
        with patch.object(s.os,'killpg',side_effect=ProcessLookupError()), \
             patch.object(s,'direct_children',return_value=set()), \
             patch.object(s.os,'waitpid',side_effect=ChildProcessError()):
            self.assertTrue(s.cleanup_owned(child,s.time.monotonic()+5)['complete'])

    def test_cleanup_deadline_is_terminal(self):
        with patch.object(s.os,'killpg'), patch.object(s,'direct_children',return_value=set()), \
             patch.object(s.os,'waitpid',return_value=(0,0)):
            with self.assertRaisesRegex(RuntimeError,'cleanup deadline'):
                s.cleanup_owned(Mock(pid=1234),s.time.monotonic()-1)

    def test_continuous_reaping_cannot_bypass_deadline(self):
        with patch.object(s.os,'killpg'), patch.object(s,'direct_children',return_value=set()), \
             patch.object(s.os,'waitpid',return_value=(2345,9)), \
             patch.object(s.time,'monotonic',side_effect=[0,0,6]):
            with self.assertRaisesRegex(RuntimeError,'cleanup deadline'):
                s.cleanup_owned(Mock(pid=1234),5)

    def test_preexisting_child_rejected_before_launch(self):
        libc=Mock();libc.prctl.return_value=0
        with patch.object(s.ctypes,'CDLL',return_value=libc), \
             patch.object(s,'direct_children',return_value={2345}), \
             patch.object(s.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'preexisting children'):
                s.supervise([],s.time.monotonic()+5,100,0,'log',{},lambda _:0)
            launch.assert_not_called()

    def test_telemetry_exception_retains_primary_and_cleans_up(self):
        libc=Mock();libc.prctl.return_value=0
        child=Mock(pid=1234);child.poll.side_effect=[None,None,0];child.wait.return_value=-9
        with tempfile.TemporaryDirectory() as td:
            fd=os.open(td,os.O_RDONLY)
            try:
                with patch.object(s.ctypes,'CDLL',return_value=libc), \
                     patch.object(s,'direct_children',return_value=set()), \
                     patch.object(s.subprocess,'Popen',return_value=child), \
                     patch.object(s.os,'getpgid',return_value=1234), \
                     patch.object(s.os,'killpg'), \
                     patch.object(s,'cleanup_owned',return_value={'complete':True}) as cleanup:
                    result=s.supervise([],s.time.monotonic()+5,100,fd,'log',{},Mock(side_effect=ValueError('fixture telemetry')))
                self.assertEqual(result['status'],'telemetry_failure')
                self.assertEqual(result['primary_failure']['reason'],'fixture telemetry')
                cleanup.assert_called_once()
            finally:os.close(fd)

    def test_kill_failure_does_not_overwrite_telemetry_failure(self):
        libc=Mock();libc.prctl.return_value=0
        child=Mock(pid=1234);child.poll.side_effect=[None,None];child.wait.return_value=-9
        with tempfile.TemporaryDirectory() as td:
            fd=os.open(td,os.O_RDONLY)
            try:
                with patch.object(s.ctypes,'CDLL',return_value=libc), \
                     patch.object(s,'direct_children',return_value=set()), \
                     patch.object(s.subprocess,'Popen',return_value=child), \
                     patch.object(s.os,'getpgid',return_value=1234), \
                     patch.object(s.os,'killpg',side_effect=PermissionError('fixture kill')), \
                     patch.object(s,'cleanup_owned',return_value={'complete':True}):
                    result=s.supervise([],s.time.monotonic()+5,100,fd,'log',{},Mock(side_effect=ValueError('fixture telemetry')))
                self.assertEqual(result['primary_failure']['reason'],'fixture telemetry')
                self.assertEqual(result['secondary_failures'][0]['reason'],'fixture kill')
            finally:os.close(fd)

    def test_qualifier_descriptor_reads_reject_symlink_and_fifo(self):
        q=module('qualify_delivery_replay_runtime')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'target').write_text('{}');(root/'link').symlink_to(root/'target');os.mkfifo(root/'fifo')
            fd=os.open(root,os.O_RDONLY)
            try:
                with self.assertRaises(OSError):q.read_fd(fd,'link')
                with self.assertRaisesRegex(ValueError,'regular qualifier'):q.read_fd(fd,'fifo')
                q.write_buffer(fd,'inventory.json',b'{}')
                self.assertEqual(q.read_fd(fd,'inventory.json'),b'{}')
            finally:os.close(fd)

    def test_descriptor_survives_directory_retarget(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'root';root.mkdir();fd=os.open(root,os.O_RDONLY)
            try:
                root.rename(Path(td)/'original');root.mkdir()
                g.write_fd(fd,'receipt.json',{'bound':True})
                self.assertTrue((Path(td)/'original/receipt.json').exists())
                self.assertFalse((root/'receipt.json').exists())
                self.assertIn(b'bound',g.read_fd(fd,'receipt.json'))
                with self.assertRaises(FileExistsError):g.write_fd(fd,'receipt.json',{})
            finally:os.close(fd)

    def test_symlink_and_nonregular_leaf_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'target').write_text('{}');(root/'link').symlink_to(root/'target')
            os.mkfifo(root/'fifo');fd=os.open(root,os.O_RDONLY)
            try:
                with self.assertRaises(OSError):g.read_fd(fd,'link')
                with self.assertRaisesRegex(ValueError,'regular descriptor'):g.read_fd(fd,'fifo')
            finally:os.close(fd)


if __name__=='__main__':unittest.main()
