"""Derived first-checkpoint gate; captured source and descriptor output custody."""
import hashlib
import json
import os
import stat
from pathlib import Path


def read_fd(fd, name):
    if Path(name).name != name:
        raise ValueError('basename required')
    handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    with os.fdopen(handle, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('regular descriptor input required')
        return stream.read()


def write_fd(fd, name, value):
    if Path(name).name != name:
        raise ValueError('basename required')
    handle = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=fd)
    with os.fdopen(handle, 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def install(qualifier, reg, fd, manifest, package, source_pins):
    qraw = read_fd(fd, 'runtime_qualification.json')
    invraw = read_fd(fd, 'runtime_inventory.json')
    qualified = json.loads(qraw)
    inventory = json.loads(invraw)
    qpin, invpin = hashlib.sha256(qraw).hexdigest(), hashlib.sha256(invraw).hexdigest()
    if qualified['qualified'] is not True or invpin != qualified['runtime_inventory_sha256']:
        raise ValueError('runtime qualification/inventory differs')
    allowed = {str((Path(package)/e['path']).resolve()): e['sha256']
               for e in manifest['files'] if e['path'].endswith('.py')}
    allowed.update({str((Path('/proc/self/fd/'+str(fd))/name).resolve()): pin
                    for name, pin in source_pins.items()})
    import sys
    torch = sys.modules['torch']
    original_load = torch.load
    first = [True]

    def guarded_load(*args, **kwargs):
        if first[0]:
            if qualifier.dependency_records(reg, reject_cached=False) != qualified['dependencies']:
                raise ValueError('live dependencies changed')
            prefix, base = Path(reg['environment']).resolve(), Path(reg['approved_base_prefix']).resolve()
            qualifier.search_path_check(prefix, base)
            entries = qualifier.verify_environment_inventory(prefix, inventory)
            loaded = qualifier.loaded_module_records(prefix, base, reg['native_backing_pins'], allowed)
            qualifier.bind_loaded_inventory(loaded, prefix, entries)
            if hashlib.sha256(read_fd(fd, 'runtime_qualification.json')).hexdigest()!=qpin or hashlib.sha256(read_fd(fd, 'runtime_inventory.json')).hexdigest()!=invpin:
                raise ValueError('qualification receipt changed')
            write_fd(fd, 'pre_checkpoint_runtime.json', {
                'passed': True, 'runtime_qualification_sha256': qpin,
                'runtime_inventory_sha256': invpin, 'sys_path': sys.path,
                'loaded_modules': loaded, 'environment_objects_revalidated': len(entries),
                'new_fits': 0, 'new_responses': 0})
            first[0] = False
        return original_load(*args, **kwargs)

    torch.load = guarded_load
    return first
