"""Opaque, separately pinned finalizer report selection. No remote work at import.

Caller must first accept finalizer execution/custody; pins here bind that decision,
not manufacture it. The original 527d backend and its receipt remain unchanged.
"""
import base64
import contextlib
import hashlib
import json
import os
from pathlib import Path
import types

BACKEND_SHA256 = '527d225b8e859c67edb4eb5ca6203e9b38e31aca1e80c8544668ccc09c6dc24b'
PREFIX = 'cpu/cpu/worker/artifacts/campaign/private/'
_BOUND = None


def report_names():
    names = [PREFIX + n for n in ('summary.json', 'report-pins.json')]
    for world in range(24):
        base = PREFIX + f'world-{world:02d}/'
        names.extend(base + n for n in ('summary.json', 'report-pins.json'))
        names.extend(base + f'arm-{arm:02d}/cost-report.json' for arm in range(8))
    return tuple(sorted(names))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def bind_sources(*, extension_source, extension_sha256, backend_source, backend_sha256):
    """Caller loads this extension from captured, independently authenticated bytes."""
    global _BOUND
    sources = (extension_source, extension_sha256, backend_source, backend_sha256)
    for source, expected in ((extension_source, extension_sha256), (backend_source, backend_sha256)):
        _require(type(source) is bytes and 0 < len(source) <= 256 * 1024
                 and type(expected) is str and hashlib.sha256(source).hexdigest() == expected,
                 'captured independent source SHA required before execution')
    _require(backend_sha256 == BACKEND_SHA256, 'original accepted backend required')
    if _BOUND is not None:
        _require(_BOUND[:4] == sources, 'source binding cannot change')
        return
    backend = types.ModuleType('captured_original_remote_custody')
    backend.__file__ = '<captured-original-527d-backend>'
    exec(compile(backend_source, backend.__file__, 'exec'), backend.__dict__)
    backend.bind_source(backend_source, backend_sha256)
    _BOUND = (*sources, backend)


def _backend():
    _require(_BOUND is not None, 'authenticate and bind both sources first')
    return _BOUND[4]


def _selection(b, request):
    receipt = request['receipt']
    b.need(request['action'] == 'batch_finalizer_reports'
           and request['helper_sha256'] == BACKEND_SHA256
           and request['extension_sha256'] == _BOUND[1], 'bound batch operation differs')
    full = b.validate_receipt(receipt)
    b.need(b.pin(request['receipt_sha256']) and b.sha(b.raw(receipt)) == request['receipt_sha256'],
           'independent original receipt byte pin differs; canonical receipt required')
    for key in ('finalizer_acceptance_sha256', 'finalizer_custody_sha256'):
        b.need(b.pin(request[key]), 'independent caller acceptance/custody pin required')
    members = request['members']
    b.need(type(members) is dict and set(members) == set(report_names()),
           'exact 192 cost reports, 24 summaries, 24 world pins, 2 campaign summary/pins required')
    b.need(b.pin(request['members_sha256']) and b.sha(b.raw(members)) == request['members_sha256'],
           'independent additional selection byte pin differs; canonical member map required')
    for name, info in members.items():
        b.need(info == full['files'].get(name) and 0 <= info['bytes'] <= b.LOCAL_MEMBER,
               'report member absent, pin changed, or over32MiB')
    b.need(sum(v['bytes'] for v in members.values()) <= b.LOCAL_TOTAL, 'report batch over512MiB')
    directories = sorted({'/'.join(n.split('/')[:i]) for n in members for i in range(1, len(n.split('/')))})
    return full, b.normalized({'files': members, 'directories': directories})


def _result(b, request, full, verification):
    return dict(schema='ace-finalizer-report-batch-transfer-v1',
        receipt_sha256=request['receipt_sha256'], members_sha256=request['members_sha256'],
        extension_sha256=request['extension_sha256'], helper_sha256=BACKEND_SHA256,
        finalizer_acceptance_sha256=request['finalizer_acceptance_sha256'],
        finalizer_custody_sha256=request['finalizer_custody_sha256'],
        full_inventory_sha256=b.sha(b.raw(full)), verification_sha256=b.sha(b.raw(verification)),
        backup_archive_sha256=request['receipt']['backup_archive_sha256'],
        destination=request['destination'], member_count=len(request['members']),
        payload_bytes=sum(v['bytes'] for v in request['members'].values()),
        full_backup_verified=True, full_archive_local_copy=False,
        original_receipt_unchanged=True, acceptance=False, retry_allowed=False)


def _verification(b, request, full):
    return dict(full_backup_verified=True, full_inventory=full,
        local_inventory=request['receipt']['local_inventory'], receipt_sha256=request['receipt_sha256'],
        backup_archive_sha256=request['receipt']['backup_archive_sha256'],
        helper_sha256=BACKEND_SHA256, acceptance=False)


def _remote_emit(request, output):
    b = _backend()
    full, selection = _selection(b, request)
    # Pass1 includes original durable receipt/inventory and complete archive validation.
    verified = b.remote_verify(dict(receipt=request['receipt'], observation=full, helper_sha256=BACKEND_SHA256))
    b.need(verified == _verification(b, request, full), 'original full verification differs')
    result = _result(b, request, full, verified)
    framed = b.raw(dict(result=result, stream_files=selection['files']))
    b.need(len(framed) <= b.MAX_METADATA, 'bounded batch frame required')
    output.write(len(framed).to_bytes(8, 'big'))
    output.write(framed)
    # Pass2 validates EVERY member again while streaming only exact opaque reports.
    actual = b.check_tar(request['receipt']['backup_archive_path'], full,
                         emit=output.write, wanted=selection['files'])
    b.need(actual == dict(bytes=request['receipt']['backup_archive_bytes'],
                          sha256=request['receipt']['backup_archive_sha256']), 'archive changed during streaming')
    output.flush()


def _specialize(function, **bindings):
    """Reuse captured code with isolated globals; never patch the original backend."""
    return types.FunctionType(function.__code__, dict(function.__globals__, **bindings),
                              function.__name__, function.__defaults__, function.__closure__)


BOOTSTRAP = r'''import sys,hashlib,json,base64
def need(ok,message):
 if not ok:raise ValueError(message)
line=sys.stdin.buffer.readline(32)
need(line.endswith(b'\n') and line[:-1].isdigit(),'bounded source bundle framing required')
n=int(line);need(0<n<=1048576,'source bundle size bound')
source=sys.stdin.buffer.read(n)
need(len(source)==n and hashlib.sha256(source).hexdigest()==sys.argv[1],'independent source bundle SHA differs')
def pairs(items):
 out={}
 for k,v in items:
  need(k not in out,'duplicate source bundle key');out[k]=v
 return out
bundle=json.loads(source,object_pairs_hook=pairs)
need(set(bundle)=={'extension','extension_sha256','backend','backend_sha256'},'source bundle keys differ')
extension=base64.b64decode(bundle['extension'],validate=True)
backend=base64.b64decode(bundle['backend'],validate=True)
for raw,pin in ((extension,bundle['extension_sha256']),(backend,bundle['backend_sha256'])):
 need(0<len(raw)<=262144 and hashlib.sha256(raw).hexdigest()==pin,'individual source SHA differs before exec')
need(bundle['backend_sha256']=='527d225b8e859c67edb4eb5ca6203e9b38e31aca1e80c8544668ccc09c6dc24b','original backend SHA differs')
v={'__name__':'captured_batch_reports','__file__':'<captured-batch-reports>'}
exec(compile(extension,v['__file__'],'exec'),v)
v['bind_sources'](extension_source=extension,extension_sha256=bundle['extension_sha256'],
 backend_source=backend,backend_sha256=bundle['backend_sha256'])
b=v['_backend']();data=sys.stdin.buffer.read(b.MAX_METADATA+1)
need(len(data)<=b.MAX_METADATA,'bounded request metadata required')
v['_specialize'](b.remote_main,remote_emit=v['_remote_emit'])(b.decode(data),sys.stdout.buffer)
'''


def _bundle(b):
    extension, epin, backend, bpin = _BOUND[:4]
    value = b.raw(dict(extension=base64.b64encode(extension).decode(), extension_sha256=epin,
                       backend=base64.b64encode(backend).decode(), backend_sha256=bpin))
    b.need(len(value) <= 1024**2, 'bounded authenticated source bundle required')
    return value


@contextlib.contextmanager
def _transcript(b, logs):
    """Reuse backend owned-file cleanup without masking an earlier transfer error."""
    cleanup = logs/'transcript-cleanup'
    b.owned_dir(cleanup, True)
    owner = b.Receiver({}, {}, lambda result: None)
    primary = None
    duplicate = None
    try:
        handle = b.opened(logs/'stdout.bin', os.O_WRONLY|os.O_CREAT|os.O_EXCL)
        fd = handle.__enter__()
        owner.handle = handle
        duplicate = os.dup(fd)
        owner.fd = os.fdopen(duplicate, 'wb')
        duplicate = None
        yield owner.fd
    except BaseException as exc:
        primary = exc
        raise
    finally:
        first_cleanup = None
        actions = []
        if owner.fd is not None:
            actions += [('transcript_flush', owner.fd.flush),
                        ('transcript_fsync', lambda: os.fsync(owner.fd.fileno()))]
        if duplicate is not None:
            actions.append(('unadopted_duplicate_close', lambda: os.close(duplicate)))
        for phase, action in actions:
            try:
                action()
            except BaseException as exc:
                if first_cleanup is None:
                    first_cleanup = exc
                owner.cleanup_errors.append(dict(phase=phase, type=type(exc).__name__, message=str(exc)))
        # Original close_receiver attempts BOTH buffered and underlying-handle
        # closes, publishes secondary errors, and preserves the supplied primary.
        b.close_receiver(owner, cleanup, primary if primary is not None else first_cleanup)
        if primary is None and first_cleanup is not None:
            raise first_cleanup


def fetch_reports(*, receipt, receipt_sha256, members, members_sha256,
                  finalizer_acceptance_sha256, finalizer_custody_sha256,
                  destination, helper_path, log_dir):
    """ONE batch, no retry. Parent independently accepts finalizer before invocation.

    Both JSON metadata objects must use the backend's canonical encoding and have
    independently supplied byte pins. Report payloads are never JSON-decoded.
    """
    b = _backend()
    # Detach caller containers: begin() may add capacity only to this new request.
    request = b.decode(b.raw(dict(action='batch_finalizer_reports', receipt=receipt,
        receipt_sha256=receipt_sha256, members=members, members_sha256=members_sha256,
        finalizer_acceptance_sha256=finalizer_acceptance_sha256,
        finalizer_custody_sha256=finalizer_custody_sha256, helper_sha256=BACKEND_SHA256,
        extension_sha256=_BOUND[1], destination=str(destination))))
    full, selection = _selection(b, request)
    dest, logs = b.absolute(destination), b.absolute(log_dir)
    b.need(not os.path.lexists(dest) and not os.path.lexists(logs), 'exclusive new destinations required')
    b.need(dest != logs and dest not in logs.parents and logs not in dest.parents, 'disjoint destinations required')
    bundle = _bundle(b)
    # Retain sources, framed stdout (including malformed/partial frames), request,
    # original receipt and operation records. Account on each actual filesystem.
    extra = 3*len(b.raw(request)) + 3*len(bundle) + b.MAX_METADATA + b.LOCAL_TOTAL + 1024**2
    logs = b.begin(logs, request, [('opaque_report_payload', dest.parent,
                   sum(v['bytes'] for v in members.values())), ('additional_evidence', logs.parent, extra)])
    receiver = None
    primary = None
    try:
        b.write_new(logs/'extension.py', _BOUND[0])
        b.write_new(logs/'source-bundle.json', bundle)
        b.write_new(logs/'bootstrap.py', BOOTSTRAP.encode())
        b.write_new(logs/'original-receipt.json', b.raw(request['receipt']))
        b.write_new(logs/'members.json', b.raw(members))
        # Do not inherit backend local_destinations' empty-directory adoption.
        # This exclusive mkdir is the actual claim, after the earlier precheck.
        b.owned_dir(dest, True)
        destinations = b.local_destinations(dest, selection)
        expected = _result(b, request, full, _verification(b, request, full))
        def validate(result):
            b.need(result == expected, 'batch verification identity differs')
        receiver = b.Receiver(selection['files'], destinations, validate)
        transport = _specialize(b.transport, BOOTSTRAP=BOOTSTRAP,
                                _BOUND_SOURCE=(bundle, b.sha(bundle)))
        with _transcript(b, logs) as transcript:
            class RetainedReceiver:
                allowed = receiver.allowed
                def feed(self, data):
                    transcript.write(data); transcript.flush()
                    return receiver.feed(data)
                def finish(self):
                    return receiver.finish()
            transport(request, RetainedReceiver(), helper_path, logs)
        result = receiver.finish()
        b.close_receiver(receiver, logs, None)
        receiver = None
        b.need(b.local_inventory(dest) == selection, 'opaque report readback differs')
        result = dict(result, local_inventory=selection, original_selected_inventory_modified=False,
                      source_bundle_sha256=b.sha(bundle), schema='ace-finalizer-report-batch-retrieval-v1')
        b.write_new(logs/'receipt.json', b.raw(result))
        return result
    except BaseException as exc:
        primary = exc
        b.safe_failure(logs, exc, partial_preserved=True, remote_outcome_ambiguous=True)
        raise
    finally:
        cleanup_error = None
        try:
            b.close_receiver(receiver, logs, primary)
        except BaseException as exc:
            cleanup_error = exc
            raise
        finally:
            try:
                b.write_new(logs/'operation-records.json', b.raw(b.local_inventory(logs)))
            except BaseException as secondary:
                first = primary if primary is not None else cleanup_error
                if first is None:
                    raise
                try:
                    b.write_new(logs/'evidence-publication-error.json', b.raw(dict(
                        primary_type=type(first).__name__, secondary_type=type(secondary).__name__,
                        secondary_message=str(secondary), acceptance=False)))
                except BaseException:
                    pass
