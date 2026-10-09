"""Derived Linux child custody; one owned group, deadline/RSS and bounded reap."""
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


def direct_children():
    children = set()
    for task in Path('/proc/self/task').iterdir():
        try:
            children.update(map(int, (task/'children').read_text().split()))
        except FileNotFoundError:
            continue
    return children


def cleanup_owned(child, deadline):
    """Called only by the isolated wrapper, after its sampler/watchdog finish."""
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=max(.01, deadline-time.monotonic()))
    killed, reaped = set(), []
    while True:
        if time.monotonic() >= deadline:
            raise RuntimeError('owned descendant cleanup deadline')
        # Linux subreaping places surviving owned descendants under this wrapper,
        # including descendants that started a new session/process group.
        for pid in direct_children():
            if time.monotonic() >= deadline:
                raise RuntimeError('owned descendant cleanup deadline')
            try:
                os.kill(pid, signal.SIGKILL)
                killed.add(pid)
            except ProcessLookupError:
                pass
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return {'complete': True, 'descendants_killed': sorted(killed),
                    'descendants_reaped': reaped}
        if pid:
            reaped.append({'pid': pid, 'status': status})
            continue
        if time.monotonic() >= deadline:
            raise RuntimeError('owned descendant cleanup deadline')
        time.sleep(.02)


def supervise(command, deadline, rss_limit, fd, log_name, env, sampler, input_fds=()):
    if time.monotonic() >= deadline:
        return {'status': 'time_limit', 'exit_code': None, 'samples': []}
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError('Linux child subreaper required')
    if direct_children():
        raise RuntimeError('isolated supervisor must have no preexisting children')
    if Path(log_name).name != log_name:
        raise ValueError('log basename required')
    handle = os.open(log_name, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,
                     0o600, dir_fd=fd)
    samples, reasons, lock = [], [], threading.Lock()
    child = None
    primary, cleanup, code = None, None, None
    secondary = []

    def record_failure(exc):
        nonlocal primary
        item = {'type': type(exc).__name__, 'reason': str(exc)}
        if primary is None:
            primary = item
        else:
            secondary.append(item)
    with os.fdopen(handle, 'w') as stream:
        try:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=env, pass_fds=(fd,*input_fds))
            if os.getpgid(child.pid) != child.pid:
                raise RuntimeError('owned process-group identity differs')

            def stop(why):
                with lock:
                    if child.poll() is None:
                        reasons.append(why)
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

            watchdog = threading.Timer(max(0, deadline-time.monotonic()), stop,
                                       args=('time_limit',))
            watchdog.start()
            try:
                while child.poll() is None:
                    try:
                        rss = sampler(child.pid)
                        samples.append({'monotonic': time.monotonic(), 'rss_bytes': rss})
                        if rss > rss_limit:
                            stop('memory_limit')
                    except BaseException as exc:
                        record_failure(exc)
                        stop('telemetry_failure')
                    time.sleep(.1)
                code = child.wait()
            finally:
                watchdog.cancel()
                watchdog.join()
        except BaseException as exc:
            record_failure(exc)
        finally:
            if child is not None:
                try:
                    cleanup = cleanup_owned(child, min(deadline+5, time.monotonic()+5))
                except BaseException as exc:
                    cleanup = {'complete': False, 'failure_type': type(exc).__name__,
                               'failure_reason': str(exc)}
    status = reasons[0] if reasons else ('failed' if primary or code != 0 else 'complete')
    if cleanup is None or cleanup.get('complete') is not True:
        status = 'cleanup_failure'
    return {'status': status, 'exit_code': code, 'samples': samples,
            'owned_child_pid': None if child is None else child.pid,
            'primary_failure': primary, 'secondary_failures': secondary, 'cleanup': cleanup}
