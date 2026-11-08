"""Bounded CPU child ownership; caller retains OwnedChild through outer finally.

No runtime, resource or acceptance authority. The caller authenticates this source,
sets the original absolute deadlines and calls hold_until_closed before cancelling
its alarm. No process-group action is allowed after observed retirement.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


def save(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')


def require(ok, message):
    if not ok:
        raise ValueError(message)


class OwnedChild:
    def __init__(self, hard_deadline):
        require(type(hard_deadline) in (int, float) and math.isfinite(hard_deadline), 'finite hard deadline required')
        self.hard_deadline = hard_deadline
        self.process = None
        self.launched = self.retired = False
        self.errors = []

    def error(self, stage, exc):
        row = dict(stage=stage, type=type(exc).__name__, message=str(exc))
        if len(self.errors) < 32 and row not in self.errors:
            self.errors.append(row)

    def confirm_closed(self):
        if self.retired or self.process is None:
            return True
        try:
            exited = self.process.poll() is not None
            try:
                os.killpg(self.process.pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
        except BaseException as exc:
            self.error('closure_observation', exc)
            return False
        if exited and not alive:
            self.retired = True
            self.process = None
            return True
        return False

    def kill(self):
        if self.retired or self.process is None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except BaseException as exc:
            self.error('owned_kill', exc)

    def hard_exit(self, sig=None, frame=None):
        try:
            self.kill()
        finally:
            os._exit(124)

    def arm_hard(self):
        signal.signal(signal.SIGALRM, self.hard_exit)
        signal.setitimer(signal.ITIMER_REAL, max(.001, self.hard_deadline-time.monotonic()))

    def hold_until_closed(self):
        """Never surrender an uncertain handle because cleanup/reporting failed."""
        self.arm_hard()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        while not self.confirm_closed():
            if time.monotonic() >= self.hard_deadline:
                self.hard_exit()
            self.kill()
            try:
                time.sleep(.05)
            except BaseException as exc:
                self.error('closure_wait', exc)


def startup_marker(path, *, packet_sha256, pid, started, startup_deadline):
    with os.fdopen(os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK), 'rb') as f:
        raw = f.read(8193)
    require(len(raw) <= 8192, 'bounded startup marker required')
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'duplicate startup marker key')
            result[k] = v
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    require(set(value) == {'packet_sha256', 'pid', 'monotonic'}
        and value['packet_sha256'] == packet_sha256 and type(value['pid']) is int and value['pid'] == pid
        and type(value['monotonic']) in (int, float) and math.isfinite(value['monotonic'])
        and started <= value['monotonic'] < startup_deadline, 'worker startup identity/time differs')
    return value


def supervise(command, env, out, *, started, startup_deadline, work_deadline,
              packet_sha256, owned, kill_owned):
    """Observe readiness before work; retain handle and failures through cleanup.

    Caller must hold_until_closed in its outer finally even when this raises.
    Natural success requires both leader and process group independently closed.
    """
    require(all(type(x) in (int, float) and math.isfinite(x) for x in
        (started, startup_deadline, work_deadline, owned.hard_deadline))
        and started <= time.monotonic() < startup_deadline < work_deadline < owned.hard_deadline
        and owned.hard_deadline-work_deadline >= 5, 'original absolute deadlines required')
    require(not owned.launched and owned.process is None and not owned.retired, 'owned launch is once-only')
    out = Path(out)
    primary = cleanup_error = record_error = None
    cleanup = None
    ready = False
    rc = None
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    def expired(sig, frame):
        raise TimeoutError('CPU startup/work/cleanup deadline exceeded')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, startup_deadline-time.monotonic()))
    try:
        with (out/'worker.stdout.log').open('xb') as stdout, (out/'worker.stderr.log').open('xb') as stderr:
            owned.launched = True
            owned.process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            proc = owned.process
            save(out/'worker.started.json', dict(command=command, pid=proc.pid, pgid=proc.pid,
                started_monotonic=started, startup_deadline=startup_deadline,
                deadline_monotonic=work_deadline, supervisor_deadline_monotonic=owned.hard_deadline))
            while True:
                now = time.monotonic()
                require(now < (work_deadline if ready else startup_deadline), 'absolute startup/work deadline exceeded')
                marker = out/'worker/work-start.json'
                if not ready and marker.exists():
                    startup_marker(marker, packet_sha256=packet_sha256, pid=proc.pid,
                        started=started, startup_deadline=startup_deadline)
                    ready = True
                    signal.setitimer(signal.ITIMER_REAL, max(.001, work_deadline-time.monotonic()))
                rc = proc.poll()
                if rc is not None:
                    require(rc == 0 and ready, 'CPU worker failed or never became ready')
                    break
                time.sleep(.05)
    except BaseException as exc:
        primary = exc
        owned.error('work', exc)
    previous = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM)}
    cleanup_deadline = min(time.monotonic()+5, owned.hard_deadline)
    signal.setitimer(signal.ITIMER_REAL, max(.001, cleanup_deadline-time.monotonic()))
    try:
        if owned.process is not None:
            cleanup = kill_owned(owned.process)
            # Retire authority from independent observations BEFORE receipt I/O.
            closed = owned.confirm_closed()
            save(out/'worker.cleanup.json', cleanup)
            require(closed and cleanup['group_still_exists'] is False and cleanup['errors'] == []
                and cleanup['returncode'] == 0 and cleanup['kill_sent'] is False,
                'natural CPU leader/group closure required')
        require(time.monotonic() < cleanup_deadline, 'bounded cleanup exceeded')
    except BaseException as exc:
        cleanup_error = exc
        owned.error('cleanup', exc)
    finally:
        owned.arm_hard()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = dict(returncode=rc, cleanup=cleanup, startup_observed=ready,
        elapsed_including_cleanup_seconds=time.monotonic()-started,
        child_and_cleanup_cpu_seconds=after.ru_utime+after.ru_stime-usage.ru_utime-usage.ru_stime,
        cumulative_child_peak_rss_bytes=after.ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
        errors=list(owned.errors))
    try:
        save(out/'worker.resources.json', result)
    except BaseException as exc:
        record_error = exc
        owned.error('resource_publication', exc)
    if primary is not None:
        raise primary from (cleanup_error or record_error)
    if cleanup_error is not None:
        raise cleanup_error from record_error
    if record_error is not None:
        raise record_error
    return result
