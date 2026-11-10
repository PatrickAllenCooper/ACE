"""Copy an existing runtime without installing or rewriting any of its files.

CPU-only preparation. The caller supplies exclusive owned output locations,
resource bounds and process supervision. A verified copy is not runtime or
model qualification. Absolute or escaping symlinks fail rather than change.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile

VERSION = "ace-runtime-archive-v1"
MAX_OBJECTS = 100000
MAX_BYTES = 16 * 1024**3
CHUNK = 1024**2


def _save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def _name(value):
    if (type(value) is not str or not value or value.startswith("/")
            or any(ord(c) < 32 for c in value) or "\\" in value
            or str(PurePosixPath(value)) != value
            or any(p in (".", "..", "") for p in value.split("/"))):
        raise ValueError("noncanonical archive path")
    return value


def _link(name, target):
    if (type(target) is not str or not target or target.startswith("/")
            or "\\" in target or any(ord(c) < 32 for c in target)):
        raise ValueError("absolute or invalid runtime link")
    parts = list(PurePosixPath(name).parent.parts)
    for part in target.split("/"):
        if part == "..":
            if not parts:
                raise ValueError("runtime link escapes root")
            parts.pop()
        elif part not in ("", "."):
            parts.append(part)
    return target


def _unchanged(before, after):
    return (before.st_dev, before.st_ino, before.st_mode, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns) == (
            after.st_dev, after.st_ino, after.st_mode, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns)


class _HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.count += len(data)
        return data


def pack(source, output, *, progress=lambda value: None):
    """Archive captured bytes once, retaining any incomplete output on failure."""
    source, output = Path(source).resolve(strict=True), Path(output).absolute()
    output = output.parent.resolve(strict=True)/output.name
    if not source.is_dir() or source == output or source in output.parents:
        raise ValueError("external exclusive output required")
    output.mkdir(mode=0o700, exist_ok=False)
    records, total = [], 0
    # No following directory links. Unsupported or escaping objects fail first.
    def walk_error(error):
        raise error
    for parent, dirs, files in os.walk(source, followlinks=False, onerror=walk_error):
        for name in sorted(dirs + files):
            p = Path(parent)/name
            rel = _name(p.relative_to(source).as_posix())
            s = p.lstat()
            if len(records) >= MAX_OBJECTS:
                raise ValueError("runtime object cap")
            kind = ("file" if stat.S_ISREG(s.st_mode) else "directory" if stat.S_ISDIR(s.st_mode)
                    else "symlink" if stat.S_ISLNK(s.st_mode) else None)
            if kind is None or s.st_mode & (0o5000 if kind == "directory" else 0o7000):
                raise ValueError("unsupported runtime object or special permissions: " + rel)
            row = dict(path=rel, kind=kind, mode=stat.S_IMODE(s.st_mode) & 0o777,
                       source_mode=stat.S_IMODE(s.st_mode))
            if kind == "file":
                total += s.st_size
                if total > MAX_BYTES:
                    raise ValueError("runtime byte cap")
                row["bytes"] = s.st_size
            if kind == "symlink":
                row["target"] = _link(rel, os.readlink(p))
                if not p.resolve().is_relative_to(source):
                    raise ValueError("resolved runtime link escapes root: " + rel)
            row["_stat"] = s
            records.append(row)
        if len(records) % 500 < len(dirs) + len(files):
            progress(dict(phase="inventory", objects=len(records), declared_bytes=total))
    records.sort(key=lambda r: r["path"])
    progress(dict(phase="inventory_complete", objects=len(records), declared_bytes=total))
    with (output/"runtime.tar").open("xb") as raw:
        with tarfile.open(fileobj=raw, mode="w|", format=tarfile.PAX_FORMAT) as tar:
            for i, row in enumerate(records):
                p = source/row["path"]
                before = row.pop("_stat")
                if not _unchanged(before, p.lstat()):
                    raise ValueError("runtime changed before capture: " + row["path"])
                info = tarfile.TarInfo(row["path"])
                info.mode = row["mode"]
                if row["kind"] == "file":
                    info.size = row["bytes"]
                    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
                    with os.fdopen(fd, "rb") as stream:
                        if not _unchanged(before, os.fstat(stream.fileno())):
                            raise ValueError("runtime changed on open")
                        reader = _HashReader(stream)
                        tar.addfile(info, reader)
                        if reader.count != info.size or not _unchanged(before, os.fstat(stream.fileno())):
                            raise ValueError("runtime changed during capture")
                        row["sha256"] = reader.digest.hexdigest()
                elif row["kind"] == "directory":
                    info.type = tarfile.DIRTYPE
                    tar.addfile(info)
                else:
                    info.type = tarfile.SYMTYPE
                    info.linkname = row["target"]
                    tar.addfile(info)
                if not _unchanged(before, p.lstat()):
                    raise ValueError("runtime changed after capture")
                if i % 500 == 0:
                    progress(dict(phase="archive", completed_objects=i+1, objects=len(records)))
        raw.flush()
        os.fsync(raw.fileno())
    archive = output/"runtime.tar"
    manifest = dict(version=VERSION, source=str(source), objects=len(records),
                    file_bytes=total, archive_bytes=archive.stat().st_size,
                    archive_sha256=_hash(archive), records=records)
    _save(output/"manifest.json", manifest)
    return manifest


def _validate(manifest):
    if type(manifest) is not dict or set(manifest) != {"version", "source", "objects", "file_bytes", "archive_bytes", "archive_sha256", "records"}:
        raise ValueError("exact runtime manifest required")
    records = manifest["records"]
    if (manifest["version"] != VERSION or type(records) is not list
            or not 0 < len(records) <= MAX_OBJECTS or manifest["objects"] != len(records)
            or type(manifest["objects"]) is not int
            or type(manifest["source"]) is not str or not manifest["source"].startswith("/")
            or type(manifest["archive_bytes"]) is not int or not 0 < manifest["archive_bytes"] <= MAX_BYTES+MAX_OBJECTS*2048
            or type(manifest["archive_sha256"]) is not str or len(manifest["archive_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in manifest["archive_sha256"])
            or type(manifest["file_bytes"]) is not int or not 0 <= manifest["file_bytes"] <= MAX_BYTES):
        raise ValueError("runtime manifest bounds")
    seen, total = {}, 0
    for r in records:
        if type(r) is not dict:
            raise ValueError("invalid runtime record")
        name = _name(r.get("path"))
        kind = r.get("kind")
        expected = {"path", "kind", "mode", "source_mode"} | ({"bytes", "sha256"} if kind == "file" else {"target"} if kind == "symlink" else set())
        if (set(r) != expected or kind not in ("file", "directory", "symlink") or name in seen
                or type(r["mode"]) is not int or not 0 <= r["mode"] <= 0o777
                or type(r["source_mode"]) is not int
                or r["source_mode"] not in ((r["mode"], r["mode"] | 0o2000) if kind == "directory" else (r["mode"],))):
            raise ValueError("invalid runtime record fields")
        if any(seen.get(str(p), {}).get("kind") != "directory" for p in PurePosixPath(name).parents if str(p) != "."):
            raise ValueError("non-directory or missing ancestor")
        if kind == "file":
            if (type(r["bytes"]) is not int or r["bytes"] < 0 or type(r["sha256"]) is not str
                    or len(r["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in r["sha256"])):
                raise ValueError("invalid captured file pin")
            total += r["bytes"]
        if kind == "symlink":
            _link(name, r["target"])
        seen[name] = r
    if list(seen) != sorted(seen) or total != manifest["file_bytes"]:
        raise ValueError("manifest order/byte count differs")
    return records


def stage(archive, manifest_path, manifest_sha256, output, *, progress=lambda value: None):
    """Verify archive and each extracted object. Never execute the copied runtime."""
    raw = Path(manifest_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("independent manifest pin differs")
    manifest = json.loads(raw)
    records = _validate(manifest)
    archive, output = Path(archive), Path(output).absolute()
    output = output.parent.resolve(strict=True)/output.name
    if archive.stat().st_size != manifest["archive_bytes"] or _hash(archive) != manifest["archive_sha256"]:
        raise ValueError("archive pin differs")
    output.mkdir(mode=0o700, exist_ok=False)
    links = []
    with tarfile.open(archive, "r|") as tar:
        for i, row in enumerate(records):
            item = tar.next()
            kind = "file" if item is not None and item.isfile() else "directory" if item is not None and item.isdir() else "symlink" if item is not None and item.issym() else None
            if (item is None or item.name != row["path"] or kind != row["kind"]
                    or item.mode != row["mode"] or item.uid != 0 or item.gid != 0):
                raise ValueError("archive object differs from manifest")
            p = output/row["path"]
            if kind == "directory":
                p.mkdir(mode=0o700)
            elif kind == "symlink":
                if item.linkname != row["target"]:
                    raise ValueError("archive link differs")
                links.append((p, item.linkname))
            else:
                if item.size != row["bytes"]:
                    raise ValueError("archive file size differs")
                h = hashlib.sha256()
                with tar.extractfile(item) as src, p.open("xb") as dst:
                    for data in iter(lambda: src.read(CHUNK), b""):
                        h.update(data)
                        dst.write(data)
                if h.hexdigest() != row["sha256"] or p.stat().st_size != row["bytes"]:
                    raise ValueError("staged bytes differ")
                p.chmod(row["mode"])
            if i % 500 == 0:
                progress(dict(phase="stage", completed_objects=i+1, objects=len(records)))
        if tar.next() is not None:
            raise ValueError("extra archive object")
    for p, target in links:
        p.symlink_to(target)
    # A later link can change resolution of an earlier link, including through
    # '..'. Check the completed graph, not only each partial construction.
    for p, target in links:
        if not p.resolve().is_relative_to(output):
            raise ValueError("staged link escapes root")
    for row in reversed(records):
        if row["kind"] == "directory":
            (output/row["path"]).chmod(row["mode"])
    return dict(version=VERSION, manifest_sha256=manifest_sha256,
                archive_sha256=manifest["archive_sha256"], objects=len(records),
                file_bytes=manifest["file_bytes"], staged_prefix=str(output),
                runtime_qualified=False, model_loaded=False)
