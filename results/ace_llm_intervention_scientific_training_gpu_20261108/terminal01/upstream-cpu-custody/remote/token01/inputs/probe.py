"""CPU-only fresh-process import diagnosis. Never load weights or query CUDA."""
import argparse
import datetime
import faulthandler
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import re
import signal
import subprocess
import sys
import time
import traceback
import types


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def pinned(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('source/input digest mismatch: '+str(path))
    return raw


def capture_installed_sources(out):
    """Bounded source contents retained before imports, including failed arms."""
    selected = {
        'torch': ['torch/__init__.py'],
        'transformers': ['transformers/'+p for p in (
            '__init__.py', 'models/qwen3/modeling_qwen3.py', 'modeling_utils.py',
            'processing_utils.py', 'loss/loss_for_object_detection.py')],
        'scipy': ['scipy/'+p for p in (
            '__init__.py', 'optimize/__init__.py', 'linalg/__init__.py',
            'linalg/_sketches.py', 'sparse/__init__.py', 'sparse/linalg/__init__.py',
            'sparse/linalg/_eigen/__init__.py', 'sparse/linalg/_eigen/_svds.py',
            'sparse/linalg/_eigen/arpack/__init__.py', 'sparse/linalg/_eigen/arpack/arpack.py')]}
    dest = out/'installed_sources'
    dest.mkdir()
    records = {}
    for package, files in selected.items():
        for relative in files:
            try:
                distribution = importlib.metadata.distribution(package)
                source = Path(distribution.locate_file(relative))
                raw = source.read_bytes()
                saved = dest/relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                with saved.open('xb') as f:
                    f.write(raw)
                records[relative] = dict(path=str(source), version=distribution.version,
                    sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), status='captured')
            except Exception as exc:
                records[relative] = dict(status='evidence_read_failed',
                    exception_type=type(exc).__name__, message=str(exc))
    write(out/'installed_sources.json', records)


def kill_owned(proc):
    result = dict(pid=proc.pid, pgid=proc.pid, kill_sent=False, already_gone=False, errors=[])
    try:
        os.killpg(proc.pid, signal.SIGKILL)
        result['kill_sent'] = True
    except ProcessLookupError:
        result['already_gone'] = True
    except Exception as exc:
        result['errors'].append(type(exc).__name__+': '+str(exc))
    try:
        result['returncode'] = proc.wait(timeout=5)
    except Exception as exc:
        result['returncode'] = None
        result['errors'].append(type(exc).__name__+': '+str(exc))
    try:
        os.killpg(proc.pid, 0)
        result['group_still_exists'] = True
    except ProcessLookupError:
        result['group_still_exists'] = False
    except Exception as exc:
        result['group_still_exists'] = None
        result['errors'].append(type(exc).__name__+': '+str(exc))
    return result


def worker(args):
    out = Path(args.output)
    out.mkdir()
    start, cpu = time.monotonic(), time.process_time()
    count = 0
    def event(phase):
        nonlocal count
        write(out/f'phase-{count:03d}.json', dict(phase=phase,
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-start, process_cpu_seconds=time.process_time()-cpu))
        count += 1
    def timeout(signum, frame):
        event('deadline_signal')
        raise TimeoutError('210 second import worker cap')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(210)
    with (out/'stacks.log').open('x') as stacks:
        faulthandler.dump_traceback_later(15, repeat=True, file=stacks, exit=False)
        try:
            event('authenticate_start')
            source = pinned(args.service, args.service_sha256)
            service = types.ModuleType('llm_intervention_model_service')
            service.__file__ = args.service
            sys.modules[service.__name__] = service
            exec(compile(source, args.service, 'exec'), service.__dict__)
            freeze, captured, adapter, models = service.authenticate(args.freeze, args.freeze_sha256)
            if freeze['treatment'] != 'frozen' or freeze['scope'] != 'artificial' or adapter is not None:
                raise ValueError('diagnostic requires original frozen artificial input')
            sys.meta_path.insert(0, service.CapturedSources(captured, Path(args.freeze).parent))
            event('authenticate_complete')
            capture_installed_sources(out)
            event('installed_source_capture_complete')
            from llm_intervention_persistent_policy import serve_policy
            event('transport_import_complete')
            if args.mode == 'scipy_first':
                event('explicit_scipy_start')
                import scipy.linalg
                import scipy.sparse.linalg
                import scipy.optimize
                event('explicit_scipy_complete')
            event('torch_start')
            import torch
            from torch.nn.attention import sdpa_kernel, SDPBackend
            event('torch_complete')
            from transformers import AutoModelForCausalLM, AutoTokenizer
            event('transformers_auto_complete')
            from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
            event('qwen_class_complete')
            from llm_intervention_language_policy import restore_adapter, generate_task_decision
            from llm_intervention_canonical_scoring import install_canonical_projection
            event('project_helpers_complete')
            imported = {}
            selected = {'torch', 'transformers', 'transformers.models.qwen3.modeling_qwen3',
                'transformers.modeling_utils', 'transformers.processing_utils',
                'transformers.loss.loss_for_object_detection', 'scipy', 'scipy.optimize',
                'scipy.linalg', 'scipy.linalg._sketches', 'scipy.sparse',
                'scipy.sparse.linalg', 'scipy.sparse.linalg._eigen',
                'scipy.sparse.linalg._eigen._svds', 'scipy.sparse.linalg._eigen.arpack',
                'scipy.sparse.linalg._eigen.arpack.arpack'}
            for name, module in sorted(sys.modules.items()):
                path = getattr(module, '__file__', None)
                if name in selected and path and Path(path).is_file():
                    # Only source modules for diagnosis. No claim of complete installed closure.
                    if str(path).endswith('.py'):
                        raw = Path(path).read_bytes()
                        imported[name] = dict(path=path, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
            write(out/'imported_sources.json', imported)
            write(out/'terminal.json', dict(status='complete', mode=args.mode,
                elapsed_seconds=time.monotonic()-start, process_cpu_seconds=time.process_time()-cpu,
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                model_weights_loaded=False, explicit_cuda_api_called=False,
                dependency_capability_checks_instrumented=False,
                training_updates=0, environment_responses=0))
        except BaseException as exc:
            write(out/'failure.json', dict(type=type(exc).__name__, message=str(exc),
                traceback=traceback.format_exc(), mode=args.mode,
                elapsed_seconds=time.monotonic()-start, process_cpu_seconds=time.process_time()-cpu,
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                loaded_scipy=[k for k in sys.modules if k.startswith('scipy')], retry_allowed=False))
            raise
        finally:
            signal.alarm(0)
            faulthandler.cancel_dump_traceback_later()


def main():
    p = argparse.ArgumentParser()
    for name in ('service', 'service-sha256', 'freeze', 'freeze-sha256', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--mode', choices=('matched_import_order','scipy_first'))
    args = p.parse_args()
    if args.mode:
        return worker(args)
    def interrupted(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise InterruptedError('diagnostic wrapper interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    out = Path(args.output)
    out.mkdir()
    own = Path(__file__).absolute()
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit():
        raise ValueError('bounded CPU Slurm allocation required')
    scheduler = subprocess.run(['scontrol','show','job','-o',job],
        capture_output=True,text=True,timeout=10)
    write(out/'allocation_raw.json', dict(stdout=scheduler.stdout,
        stderr=scheduler.stderr,returncode=scheduler.returncode))
    fields = dict(re.findall(r'(\w+)=(\S+)',scheduler.stdout))
    tres = fields.get('AllocTRES','').split(',')
    if (scheduler.returncode or fields.get('JobId') != job
            or fields.get('Account') != 'ucb736_asc2' or fields.get('Partition') != 'acpu'
            or fields.get('QOS') != 'cpu-normal' or fields.get('NumCPUs') != '1'
            or fields.get('NumTasks') != '1' or fields.get('NumNodes') != '1'
            or fields.get('TimeLimit') != '00:08:00' or fields.get('Requeue') != '0'
            or 'cpu=1' not in tres or 'mem=3G' not in tres
            or any('gres/' in item for item in tres)):
        raise ValueError('actual allocation differs from fixed CPU-only bound')
    results = []
    for mode in ('matched_import_order','scipy_first'):
        command = [sys.executable, '-B', '-X', 'importtime', str(own),
            '--service', args.service, '--service-sha256', args.service_sha256,
            '--freeze', args.freeze, '--freeze-sha256', args.freeze_sha256,
            '--output', str(out/mode), '--mode', mode]
        proc = None
        result = dict(mode=mode, command=command, status='not_started')
        try:
            with (out/(mode+'.stdout')).open('xb') as stdout, (out/(mode+'.stderr')).open('xb') as stderr:
                proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
                result.update(pid=proc.pid, pgid=proc.pid,
                    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                write(out/(mode+'.started.json'), result)
                try:
                    result.update(returncode=proc.wait(timeout=220), status='returned')
                except subprocess.TimeoutExpired:
                    result['status'] = 'killed_at_outer_deadline'
                    result['cleanup'] = kill_owned(proc)
                    result['returncode'] = result['cleanup']['returncode']
        except BaseException as exc:
            result.update(status='wrapper_exception', exception_type=type(exc).__name__,
                message=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            if proc is not None and result['status'] != 'returned' and 'cleanup' not in result:
                result['cleanup'] = kill_owned(proc)
            write(out/(mode+'.process.json'), result)
        if result.get('cleanup', {}).get('group_still_exists') is not False and 'cleanup' in result:
            raise RuntimeError('uncertain owned-group cleanup forbids next mode')
        results.append(result)
    write(out/'diagnostic_complete.json', dict(status='diagnostics_retained', results=results,
        gpu_requested=False, runtime_qualified=False, no_automatic_retry=True))
    # A diagnostic's successful wrapper is not successful imports or runtime qualification.
    return 0


if __name__ == '__main__':
    main()
