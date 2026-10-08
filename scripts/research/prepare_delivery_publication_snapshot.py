"""Capture a private, revision-pinned manuscript/source snapshot for later release.

No experiments, claim generation, source execution or package mutation. This
snapshot is outside the frozen replay package. It binds current writing bytes;
final B reports, numerical acceptance, anonymity and submission remain separate.
"""
import argparse
import ast
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

PAPER = 'paper/aistats_ace_2027/'
COMPANIONS = ('tmlr.sty', 'fancyhdr.sty', 'tmlr.bst', 'delivery_references.bib',
              'delivery_claims.tex', 'delivery_attribution_table.tex',
              'delivery_history_table.tex', 'delivery_physical_table.tex',
              'delivery_theory.tex', 'delivery_method_diagram.tex',
              'delivery_evaluation_design.tex')
SOURCES = ('scripts/research/sync_delivery_manuscript.py',
           'scripts/research/generate_delivery_claims.py',
           'scripts/research/verify_delivery_table_data.py',
           'docs/development/guidance/delivery_reviewer_commands_2026-10-08.md')
FILES = tuple(PAPER + n for n in ('paper.tex', *COMPANIONS,
                                 'claim_index.json', 'tmlr_style_provenance.json',
                                 'TMLR_STYLE_LICENSE', 'archive/tmlr_official_template.tex')) + SOURCES
BEGIN = '% BEGIN EDITOR COMPANION BUNDLE (generated; edit companion files)'
END = '% END EDITOR COMPANION BUNDLE'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git(repo, *args):
    return subprocess.run(['git', '--no-replace-objects', '-C', str(repo), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def plain_path(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlink in custody location')
    return path


def capture(repo, revision):
    """Read immutable Git blobs, never import a source or read working-tree bytes."""
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('independent full commit pin required')
    if git(repo, 'rev-parse', '--verify', revision + '^{commit}').decode().strip() != revision:
        raise ValueError('commit pin differs')
    committed = git(repo, 'cat-file', 'commit', revision)
    if hashlib.sha1(b'commit ' + str(len(committed)).encode() + b'\0' + committed).hexdigest() != revision:
        raise ValueError('captured commit object differs from independent pin')
    header = committed.split(b'\n', 1)[0]
    if not re.fullmatch(rb'tree [0-9a-f]{40}', header):
        raise ValueError('commit tree identity missing')
    root_tree = header[5:].decode()
    trees = {}
    def entries(object_id):
        if object_id in trees:
            return trees[object_id]
        raw = git(repo, 'cat-file', 'tree', object_id)
        if hashlib.sha1(b'tree ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() != object_id:
            raise ValueError('captured tree bytes differ from Git identity')
        table, offset = {}, 0
        while offset < len(raw):
            stop = raw.index(b'\0', offset)
            mode, name = raw[offset:stop].split(b' ', 1)
            if name in table or stop + 21 > len(raw):
                raise ValueError('malformed or duplicate Git tree entry')
            table[name] = (mode, raw[stop + 1:stop + 21].hex())
            offset = stop + 21
        trees[object_id] = table
        return table
    captured, blobs = {}, {}
    for name in FILES:
        current = root_tree
        parts = name.split('/')
        for index, part in enumerate(parts):
            entry = entries(current).get(part.encode())
            if entry is None:
                raise ValueError('required publication file missing: ' + name)
            mode, current = entry
            if index < len(parts)-1:
                if mode != b'40000':
                    raise ValueError('regular committed publication directory required')
            elif mode not in (b'100644', b'100755'):
                raise ValueError('regular committed publication file required')
        blob = current
        size = int(git(repo, 'cat-file', '-s', blob))
        if size > 16 * 1024 * 1024:
            raise ValueError('publication object exceeds metadata size bound')
        raw = git(repo, 'cat-file', 'blob', blob)
        if len(raw) != size:
            raise ValueError('captured blob size differs')
        if hashlib.sha1(b'blob ' + str(size).encode() + b'\0' + raw).hexdigest() != blob:
            raise ValueError('captured bytes differ from Git blob identity')
        captured[name], blobs[name] = raw, blob
    return captured, blobs


def check_bundle(raw):
    """Check captured companion membership and bytes without executing sync code."""
    tree = ast.parse(raw[SOURCES[0]])
    assignments = [n for n in tree.body if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == 'FILES' for t in n.targets)]
    if len(assignments) != 1 or ast.literal_eval(assignments[0].value) != COMPANIONS:
        raise ValueError('publication companion membership changed; update reviewed inventory')
    text = raw[PAPER + 'paper.tex'].decode()
    if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(BEGIN) >= text.index(END):
        raise ValueError('single ordered editor bundle required')
    chunks = [BEGIN]
    for name in COMPANIONS:
        content = raw[PAPER + name]
        decoded = content.decode()
        if '\\end{filecontents*}' in decoded:
            raise ValueError('nested companion terminator')
        chunks += [f'% {name}: SHA256 {digest(content)}',
                   f'\\begin{{filecontents*}}[overwrite]{{{name}}}',
                   decoded.rstrip('\n'), '\\end{filecontents*}']
    chunks.append(END)
    if text[text.index(BEGIN):text.index(END) + len(END)] != '\n'.join(chunks):
        raise ValueError('captured manuscript/companion bytes differ')
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate publication metadata key')
            value[key] = item
        return value
    def invalid(value):
        raise ValueError('nonfinite publication metadata constant')
    def metadata(name):
        return json.loads(raw[PAPER + name], object_pairs_hook=unique,
                          parse_constant=invalid)
    styles = metadata('tmlr_style_provenance.json')['files']
    expected = {'tmlr.sty', 'fancyhdr.sty', 'tmlr.bst', 'TMLR_STYLE_LICENSE',
                'archive/tmlr_official_template.tex'}
    if set(styles) != expected or any(styles[n]['sha256'] != digest(raw[PAPER + n])
                                       for n in expected):
        raise ValueError('captured official style/notice provenance differs')
    if metadata('claim_index.json')['generator_sha256'] != digest(raw[SOURCES[1]]):
        raise ValueError('captured claim generator/index source binding differs')


DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def reject_allow_acl(fd):
    """macOS descriptor ACL check: permit deny-only ACLs, reject every allow ACE.

    Constants/signatures follow the local Darwin sys/acl.h and acl_get_entry(3).
    This internal custody interface is intentionally macOS-only; unexamined ACL
    implementations fail rather than being treated as permission-qualified.
    """
    if sys.platform != 'darwin':
        raise ValueError('publication custody ACL check requires macOS')
    lib = ctypes.CDLL(None, use_errno=True)
    lib.acl_get_fd_np.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.acl_get_fd_np.restype = ctypes.c_void_p
    lib.acl_valid.argtypes = [ctypes.c_void_p]
    lib.acl_get_entry.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                ctypes.POINTER(ctypes.c_void_p)]
    lib.acl_get_tag_type.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.acl_free.argtypes = [ctypes.c_void_p]
    os.fstat(fd)  # Valid descriptor; ENOENT below means no stored extended ACL.
    ctypes.set_errno(0)
    acl = lib.acl_get_fd_np(fd, 0x100)
    if not acl:
        if ctypes.get_errno() == errno.ENOENT:
            return
        raise ValueError('custody ACL retrieval failed')
    try:
        if lib.acl_valid(acl) != 0:
            raise ValueError('invalid custody ACL')
        for index in range(1024):
            entry = ctypes.c_void_p()
            ctypes.set_errno(0)
            status = lib.acl_get_entry(acl, index, ctypes.byref(entry))
            if status == -1 and ctypes.get_errno() == errno.EINVAL:
                return
            if status != 0:
                raise ValueError('custody ACL entry check failed')
            tag = ctypes.c_int()
            if lib.acl_get_tag_type(entry, ctypes.byref(tag)) != 0 or tag.value != 2:
                raise ValueError('allow or unsupported custody ACL entry')
        raise ValueError('custody ACL exceeds bounded entry count')
    finally:
        lib.acl_free(acl)


def trusted_directory(directory, final=False):
    """Require trusted owners and reject write access by other principals.

    The current user and root are trusted; malicious same-UID/root processes
    can alter any owned files or Git repository and are outside this boundary.
    Root-owned sticky temporary ancestors protect owned child entries, but may
    not be the final custody parent.
    """
    info = os.fstat(directory)
    sticky_ancestor = not final and info.st_uid == 0 and info.st_mode & stat.S_ISVTX
    if info.st_uid not in (0, os.geteuid()) or info.st_mode & 0o022 and not sticky_ancestor:
        raise ValueError('trusted non-shared-writable custody parent required')
    reject_allow_acl(directory)


def exclusive_directory(path):
    """Traverse with directory handles; an ancestor symlink swap cannot redirect I/O."""
    current = os.open(path.anchor, DIRECTORY_FLAGS)
    try:
        for part in path.parts[1:-1]:
            trusted_directory(current)
            try:
                child = os.open(part, DIRECTORY_FLAGS, dir_fd=current)
            except FileNotFoundError:
                os.mkdir(part, mode=0o700, dir_fd=current)
                child = os.open(part, DIRECTORY_FLAGS, dir_fd=current)
            os.close(current)
            current = child
        trusted_directory(current, final=True)
        os.mkdir(path.name, mode=0o700, dir_fd=current)
        child = os.open(path.name, DIRECTORY_FLAGS, dir_fd=current)
        try:
            reject_allow_acl(child)
            os.fchmod(child, 0o700)
        except BaseException:
            os.close(child)
            raise
        return child
    finally:
        os.close(current)


def store(directory, name, raw):
    current = os.dup(directory)
    parents = []
    try:
        for part in Path(name).parts[:-1]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=current)
            except FileExistsError:
                pass
            child = os.open(part, DIRECTORY_FLAGS, dir_fd=current)
            reject_allow_acl(child)
            os.fchmod(child, 0o700)
            parents.append(current)
            current = child
        fd = os.open(Path(name).name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     mode=0o600, dir_fd=current)
        with os.fdopen(fd, 'wb') as stream:
            reject_allow_acl(stream.fileno())
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o400)
    finally:
        os.close(current)
        for parent in parents:
            os.close(parent)


def seal_directory(directory):
    for name in os.listdir(directory):
        try:
            child = os.open(name, DIRECTORY_FLAGS, dir_fd=directory)
        except NotADirectoryError:
            continue
        try:
            seal_directory(child)
        finally:
            os.close(child)
    os.fchmod(directory, 0o500)


def prepare(repo, revision, destination):
    repo, destination = plain_path(repo), plain_path(destination)
    repo_root = Path(git(repo, 'rev-parse', '--show-toplevel').decode().strip())
    if repo_root != repo.resolve():
        raise ValueError('explicit repository root required')
    if destination.resolve().is_relative_to(repo_root) or destination.exists():
        raise ValueError('new exclusive external snapshot required')
    raw, blobs = capture(repo, revision)
    check_bundle(raw)
    records = [{'path': name, 'sha256': digest(raw[name]), 'bytes': len(raw[name]),
                'git_blob': blobs[name]} for name in FILES]
    inventory = {
        'schema': 'delivery-publication-source-snapshot-v1', 'revision': revision,
        'scope': 'private writing/source bytes; not empirical or release qualification',
        'files': records, 'companions': list(COMPANIONS),
        'scientific_inputs_included': False, 'entry_points_executed': False,
        'prospective_report_qualification_performed': False, 'anonymous_release_approved': False,
        'submission_approved': False,
        'lineage_requirement': 'Final reports need qualified replay/report pins and a new successor manifest; this snapshot never changes the frozen replay input manifest.'}
    encoded = (json.dumps(inventory, indent=2, sort_keys=True) + '\n').encode()
    directory = exclusive_directory(destination)
    try:
        for name in FILES:
            store(directory, 'source/' + name, raw[name])
        store(directory, 'inventory.json', encoded)
        seal_directory(directory)
        final = os.stat(plain_path(destination), follow_symlinks=False)
        opened = os.fstat(directory)
        if (final.st_dev, final.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError('snapshot custody location changed during writes')
    finally:
        os.close(directory)
    return {'inventory_sha256': digest(encoded), 'revision': revision,
            'files': len(records), 'companions': len(COMPANIONS),
            'destination': str(destination), 'release_qualified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--revision', required=True, help='Independently pinned full commit SHA')
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.repo, args.revision, args.destination), indent=2))
