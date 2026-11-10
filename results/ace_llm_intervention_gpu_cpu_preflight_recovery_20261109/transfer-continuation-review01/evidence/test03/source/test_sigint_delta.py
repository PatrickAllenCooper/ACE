"""One new R1 repeated-SIGINT cleanup boundary; no child, remote or real lock."""
import json
import os
from pathlib import Path
import signal
import types
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
EVIDENCE=Path(os.environ['ACE_CONTINUATION_SIGINT_EVIDENCE']).resolve()

class SigintDelta(unittest.TestCase):
    def test_second_sigint_cannot_interrupt_owned_group_confirmation(self):
        m=types.ModuleType('artificial_sigint_delta');m.__file__=str(HERE/'continue_once.py')
        exec(compile((HERE/'continue_once.py').read_bytes(),m.__file__,'exec'),vars(m))
        before={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
        primary=KeyboardInterrupt('FABRICATED FIRST CTRL-C');calls=[];completed=[]
        def probe(pgid,zero):
            calls.append((pgid,zero))
            if len(calls)==1:signal.raise_signal(signal.SIGINT);return
            raise ProcessLookupError()
        with mock.patch.object(m.os,'killpg',side_effect=probe),mock.patch.object(m.time,'sleep'):
            with self.assertRaises(KeyboardInterrupt) as caught:
                try:raise primary
                finally:
                    with m.cleanup_signals(primary):
                        facts=m.confirm_owned_group_closed(987654321);completed.append(True)
        self.assertIs(caught.exception,primary);self.assertEqual(completed,[True])
        self.assertTrue(facts['closed']);self.assertEqual(facts['observations'],['present','ESRCH'])
        self.assertEqual({s:signal.getsignal(s) for s in before},before)
        with self.assertRaises(m.TerminationRequested):
            with m.cleanup_signals(None):signal.raise_signal(signal.SIGINT);completed.append(True)
        self.assertEqual(completed,[True,True]);self.assertEqual({s:signal.getsignal(s) for s in before},before)
        (EVIDENCE/'facts.json').write_bytes(m.raw(dict(group_probe=facts,primary_exception_preserved=True,both_handlers_restored=True,actual_remote_actions=0,real_locks=0)))

if __name__=='__main__':unittest.main(verbosity=2)
