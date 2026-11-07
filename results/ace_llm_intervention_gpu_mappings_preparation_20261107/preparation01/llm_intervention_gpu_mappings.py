"""Narrow classification of observed GPU device mappings, not library paths.

The trusted caller must save the unmodified runtime observation and this
independent device inventory before using the returned check view. This is a
kernel/host-trusting consistency check, not a hostile-process sandbox, proof of
GPU assignment, or numerical runtime qualification. No devices are opened.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import re
import stat

VERSION = "ace-gpu-device-map-observation-v1"
PATHS = frozenset({"/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/zero"})
PHASES = frozenset({"after-model-load", "after-requests"})
FIELDS = {"uid", "mode", "inode", "dev_major", "dev_minor", "rdev_major", "rdev_minor"}


def need(value, message):
    if not value:
        raise ValueError(message)


def capture():
    """Call only in the GPU worker, then save these bytes as observation evidence."""
    records = {}
    for name in sorted(PATHS):
        path = Path(name)
        need(path.resolve(strict=True)==path, "device path alias rejected")
        before = path.lstat()
        record = {"uid":before.st_uid,"mode":before.st_mode,"inode":before.st_ino,
            "dev_major":os.major(before.st_dev),"dev_minor":os.minor(before.st_dev),
            "rdev_major":os.major(before.st_rdev),"rdev_minor":os.minor(before.st_rdev)}
        after = path.lstat()
        need(before==after, "device metadata changed during capture")
        records[name] = record
    with Path("/proc/devices").open("rb") as handle:
        registry = handle.read(64*1024+1)
    need(len(registry)<=64*1024,"device registry exceeds bound")
    return {"schema":VERSION,"devices":records,"proc_devices":registry.decode("ascii"),
            "proc_devices_sha256":hashlib.sha256(registry).hexdigest()}


def character_registry(raw):
    need(type(raw) is str and len(raw)<=64*1024,"bounded character registry required")
    section = None
    values = {}
    for line in raw.splitlines():
        if line in ("Character devices:","Block devices:"):
            section=line
            continue
        if not line.strip() or section!="Character devices:":
            continue
        fields = line.split()
        need(len(fields)==2 and fields[0].isdigit(),"malformed character registry")
        major = int(fields[0])
        name = fields[1]
        need(name not in values and 0<major<2**20,"duplicate/invalid character driver")
        values[name]=major
    need("nvidia-uvm" in values,"registered UVM major required")
    return values


def inventory(value):
    need(type(value) is dict and set(value)=={"schema","devices","proc_devices","proc_devices_sha256"}
         and value["schema"]==VERSION and type(value["devices"]) is dict
         and set(value["devices"])==PATHS,"exact GPU device inventory required")
    registry = character_registry(value["proc_devices"])
    need(hashlib.sha256(value["proc_devices"].encode("ascii")).hexdigest()==value["proc_devices_sha256"],
         "device registry digest differs")
    expected = {"/dev/nvidia0":(195,0),"/dev/nvidiactl":(195,255),
                "/dev/nvidia-uvm":(registry["nvidia-uvm"],0),"/dev/zero":(1,5)}
    for name,r in value["devices"].items():
        need(type(r) is dict and set(r)==FIELDS and all(type(v) is int and v>=0 for v in r.values()),
             "complete integer device metadata required")
        need(r["uid"]==0 and stat.S_ISCHR(r["mode"]) and r["inode"]>0
             and (r["rdev_major"],r["rdev_minor"])==expected[name],
             "root-owned exact character device required")
    return value["devices"]


def mapping_rows(maps):
    need(type(maps) is str and len(maps)<=8*1024**2,"bounded process maps required")
    rows=[]
    for line in maps.splitlines():
        fields=line.split(None,5)
        need(5<=len(fields)<=6,"malformed process map")
        match=re.fullmatch(r"([0-9a-f]+)-([0-9a-f]+)",fields[0])
        need(match is not None and int(match[1],16)<int(match[2],16)
             and re.fullmatch(r"[r-][w-][x-][ps]",fields[1]) is not None
             and re.fullmatch(r"[0-9a-f]+",fields[2]) is not None
             and re.fullmatch(r"[0-9a-f]+:[0-9a-f]+",fields[3]) is not None
             and fields[4].isdigit(),"invalid process map fields")
        if len(fields)==5 or not fields[5].startswith("/"):
            continue
        path=fields[5]
        deleted=path.endswith(" (deleted)")
        if deleted:path=path[:-10]
        dev_major,dev_minor=(int(x,16) for x in fields[3].split(":"))
        rows.append({"line":line,"path":path,"deleted":deleted,"permissions":fields[1],
            "offset":int(fields[2],16),"dev_major":dev_major,"dev_minor":dev_minor,
            "inode":int(fields[4])})
    return rows


def checked_view(snapshot, device_inventory, phase):
    """Return a separate guard view. Never mutate or relabel the original maps."""
    need(phase in PHASES,"device exception forbidden before model loading")
    need(type(snapshot) is dict and type(snapshot.get("mixed_native_paths")) is list,
         "original complete runtime snapshot required")
    mixed=snapshot["mixed_native_paths"]
    need(all(type(p) is str for p in mixed) and sorted(set(mixed))==mixed
         and set(mixed)<=PATHS,"unknown mapped path remains rejected")
    devices=inventory(device_inventory)
    rows=mapping_rows(snapshot.get("process_maps"))
    selected=[r for r in rows if r["path"] in PATHS]
    need({r["path"] for r in selected}==set(mixed),"raw maps and rejected device roster disagree")
    for row in selected:
        name=row["path"]
        r=devices[name]
        need(re.fullmatch(r"[r-][w-]-s",row["permissions"]) is not None and row["inode"]>0,
             "device mapping must be non-executable and shared")
        if name=="/dev/zero":
            # Exact observed Linux anonymous-shared shape. Its deleted backing
            # inode differs from today's character device, so never equate them.
            need(row["deleted"] and row["permissions"]=="rw-s" and row["offset"]==0
                 and (row["dev_major"],row["dev_minor"])==(0,1),
                 "unknown zero-device/shared-memory mapping")
        else:
            need(not row["deleted"] and (row["dev_major"],row["dev_minor"],row["inode"])==
                 (r["dev_major"],r["dev_minor"],r["inode"]),"mapped GPU device identity differs")
    view=copy.deepcopy(snapshot)
    view["mixed_native_paths"]=[]
    evidence={"schema":VERSION,"phase":phase,"classified_rows":selected,
              "original_mixed_native_paths":list(mixed),"device_inventory":copy.deepcopy(device_inventory),
              "original_snapshot_preserved":True,"gpu_assignment_proved":False,
              "scope":"Non-executable device/shared mappings only. Other runtime guards still required."}
    return view,evidence
