"""CPU-only reproduction of the failed service parent's threading inheritance.

Two fresh original-runtime parents import the captured numerical task closure.
Their torch-first children use the authenticated copied runtime. One reproduces
the previous environment handoff, the other supplies explicit GNU threading.
These are import diagnostics, never model-service or scientific qualification.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
import traceback
import types

VERSION = "ace-threading-inheritance-diagnostic-v1"
CPU_SHA = "c827a53f0fe03204c447e8e7d406e6d8a2c9318b4d6cd5fdc82beb84b576b6b7"
GPU_SHA = "ee75aead983f30393774cd239af53f9ef1caf7876bdef23089b9c0ae3f1c1174"
RESOURCES = {"account":"ucb736_asc2", "partition":"acpu", "qos":"cpu-normal",
             "cpus":2, "memory_mib":6144, "wall_seconds":600, "gpus":0, "requeue":False}
KEYS = ("MKL_THREADING_LAYER", "MKL_SERVICE_FORCE_INTEL", "LD_LIBRARY_PATH",
        "LD_PRELOAD", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "PYTHONHOME")
MODES = ("inherited", "explicit_gnu")


def need(value, message):
    if not value:
        raise ValueError(message)


def save(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with Path(path).open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(raw).hexdigest()


def module(name, path, raw):
    result = types.ModuleType(name)
    result.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), vars(result))
    return result


def environment():
    """Only this fixed allowlist is read. C values can differ from os.environ."""
    libc = ctypes.CDLL(None)
    libc.getenv.argtypes = [ctypes.c_char_p]
    libc.getenv.restype = ctypes.c_char_p
    c_values = {}
    for key in KEYS:
        value = libc.getenv(key.encode())
        need(value is None or len(value) <= 8192, "environment value exceeds bound")
        c_values[key] = None if value is None else value.decode("utf-8", "strict")
    return {"python": {k: os.environ.get(k) for k in KEYS}, "c": c_values}


def observe(out, name):
    maps = Path("/proc/self/maps").read_text()
    need(len(maps) <= 4*1024**2, "process map exceeds bound")
    return save(out/(name+".json"), {"environment": environment(), "pid":os.getpid(),
        "python":sys.version, "executable":sys.executable, "prefix":sys.prefix,
        "sys_path":sys.path, "maps":maps, "monotonic":time.monotonic(),
        "process_cpu_seconds":time.process_time()})


def explicit_environment(inherited, prefix):
    env = dict(inherited)
    for key in ("PYTHONHOME", "PYTHONPATH", "LD_PRELOAD", "MKL_SERVICE_FORCE_INTEL"):
        env.pop(key, None)
    env.update(LD_LIBRARY_PATH=str(prefix/"lib"), CUDA_VISIBLE_DEVICES="",
               MKL_THREADING_LAYER="GNU", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    return env


def allocation(raw, job):
    fields = dict(re.findall(r"(\w+)=(\S+)", raw))
    expected = {"JobId":job,"JobState":"RUNNING","Account":"ucb736_asc2",
        "Partition":"acpu","QOS":"cpu-normal","NumCPUs":"2","NumTasks":"1",
        "NumNodes":"1","TimeLimit":"00:10:00","Requeue":"0","Restarts":"0"}
    tres = fields.get("AllocTRES", "").split(",")
    need(all(fields.get(k)==v for k,v in expected.items()) and "cpu=2" in tres
         and "mem=6G" in tres and not any("gres/" in x for x in tres), "CPU diagnostic allocation differs")
    return fields


def authenticate(args):
    root = Path(args.packet).absolute().parent
    # This small bootstrap is independently pinned before it executes.
    path = root/"qualify_llm_intervention_service_cpu.py"
    with path.open("rb") as handle:
        raw = handle.read(128*1024+1)
    need(len(raw)<=128*1024 and hashlib.sha256(raw).hexdigest()==CPU_SHA, "CPU helper pin differs")
    cpu = module("pinned_cpu", path, raw)
    packet = json.loads(cpu.bounded_input(args.packet, args.packet_sha256, 128*1024))
    expected = cpu.INPUTS | {"qualify_llm_intervention_threading.py", "cpu-preflight.json",
                            "qualify_llm_intervention_model_service.py"}
    need(set(packet)=={"schema","files","resources"} and packet["schema"]==VERSION
         and packet["resources"]==RESOURCES and set(packet["files"])==expected,
         "complete diagnostic packet required")
    data = {n:cpu.bounded_input(root/n,pin,8*1024**2) for n,pin in packet["files"].items()}
    need(packet["files"][path.name]==CPU_SHA and packet["files"]["qualify_llm_intervention_model_service.py"]==GPU_SHA
         and Path(__file__).absolute()==root/"qualify_llm_intervention_threading.py", "entrypoint/helper differs")
    service = module("pinned_service",root/"llm_intervention_model_service.py",data["llm_intervention_model_service.py"])
    d = module("pinned_diagnostic",root/"qualify_llm_intervention_runtime_metadata.py",data["qualify_llm_intervention_runtime_metadata.py"])
    preflight = service.decode(data["cpu-preflight.json"])
    cpu.authenticate(root,preflight,{n:data[n] for n in cpu.INPUTS},service,d)
    need(all(packet["files"][n]==pin for n,pin in preflight["files"].items()), "CPU predecessor input drift")
    return root, packet, data, cpu, service, d, preflight


def worker(args, out):
    prefix = Path(args.prefix).resolve(strict=True)
    need(Path(sys.prefix).resolve()==prefix, "child must use copied interpreter")
    observe(out,"before-torch")
    import torch
    observe(out,"after-torch")
    import numpy as np
    import scipy
    observe(out,"after-numerical-imports")
    versions = {"torch":torch.__version__,"numpy":np.__version__,"scipy":scipy.__version__}
    need(versions=={"torch":"2.9.1+cu128","numpy":"2.2.5","scipy":"1.15.3"}, "imported versions differ")
    origins = {n:str(Path(m.__file__).resolve()) for n,m in (("torch",torch),("numpy",np),("scipy",scipy))}
    need(all(Path(p).is_relative_to(prefix) for p in origins.values()), "mixed numerical origins")
    a = np.array([[1.,2.],[3.,4.]])
    need(np.array_equal(a@a,[[7.,10.],[15.,22.]]) and torch.equal(torch.tensor(a)@torch.tensor(a),
         torch.tensor([[7.,10.],[15.,22.]],dtype=torch.float64)), "CPU arithmetic differs")
    save(out/"terminal.json",{"status":"imports_and_cpu_arithmetic_complete","versions":versions,
         "origins":origins,"model_loaded":False,"explicit_cuda_api_called":False,
         "training_updates":0,"environment_responses":0})


def parent(args, out, context):
    root, packet, data, cpu, service, d, _ = context
    need(args.variant in MODES and sys.executable==d.RUNTIME_SOURCE+"/bin/python", "original-runtime parent required")
    observe(out,"before-parent-imports")
    need(environment()["c"]["MKL_SERVICE_FORCE_INTEL"] is None, "force-Intel is prohibited")
    sys.meta_path.insert(0, service.CapturedSources({n:data[n] for n in service.SOURCE_NAMES},root))
    from llm_intervention_persistent_policy import PersistentPolicy
    from llm_intervention_task import unpack_envelope
    observe(out,"after-parent-imports")
    gpu = module("failed_gpu_caller",root/"qualify_llm_intervention_model_service.py",data["qualify_llm_intervention_model_service.py"])
    # The original helper requires nonempty visibility. A sentinel is supplied
    # only to this pure dictionary function, then removed before any subprocess.
    inherited = dict(os.environ)
    inherited["CUDA_VISIBLE_DEVICES"] = "diagnostic-dictionary-only"
    env = gpu.child_environment(Path(args.prefix),inherited)
    env["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.clear()
    os.environ.update(env)
    observe(out,"after-original-handoff")
    child_env = None if args.variant=="inherited" else explicit_environment(os.environ,Path(args.prefix))
    if child_env is not None:
        save(out/"explicit-child-environment.json",{k:child_env.get(k) for k in KEYS})
    command = [args.prefix+"/bin/python","-I","-B","-c",d.BOOTSTRAP,str(root/Path(__file__).name),
        packet["files"][Path(__file__).name],"--packet",args.packet,"--packet-sha256",args.packet_sha256,
        "--mode","worker","--prefix",args.prefix,"--output",str(out/"child")]
    proc = None
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        with (out/"child.stdout").open("xb") as stdout,(out/"child.stderr").open("xb") as stderr:
            proc = subprocess.Popen(command,stdout=stdout,stderr=stderr,env=child_env)
            save(out/"child-started.json",{"command":command,"pid":proc.pid,"pgid":os.getpgid(proc.pid),
                "parent_pgid":os.getpgrp(),"explicit_environment":child_env is not None})
            rc = proc.wait(timeout=90)
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    save(out/"terminal.json",{"status":"parent_child_observation_complete","variant":args.variant,
        "child_returncode":rc,"child_receipt_present":(out/"child/terminal.json").is_file(),
        "elapsed_seconds":time.monotonic()-started,"child_cpu_seconds":after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        "cumulative_child_maxrss_bytes":after.ru_maxrss*1024,"model_service_qualified":False,
        "model_loaded":False,"training_updates":0,"environment_responses":0})


def supervisor(args, out, context, start, cpu_start):
    root, packet, data, cpu, service, d, preflight = context
    job = os.environ.get("SLURM_JOB_ID","")
    need(job.isdigit() and not os.environ.get("SLURM_JOB_GPUS") and not os.environ.get("SLURM_STEP_GPUS"), "CPU job required")
    result = subprocess.run(["scontrol","show","job","-o",job],capture_output=True,text=True,timeout=10)
    save(out/"allocation.json",{"stdout":result.stdout,"stderr":result.stderr,"returncode":result.returncode})
    need(result.returncode==0,"scheduler query failed")
    allocation(result.stdout,job)
    scratch = Path(os.environ.get("SLURM_SCRATCH",""))
    need(scratch.is_absolute(),"absolute node scratch required")
    scratch = scratch.resolve(strict=True)
    need(scratch.is_dir() and scratch.stat().st_uid==os.getuid(),"owned scratch required")
    local = scratch/("ace-threading-"+job)
    local.mkdir(mode=0o700,exist_ok=False)
    helper = module("archive_helper",root/"llm_intervention_runtime_archive.py",data["llm_intervention_runtime_archive.py"])
    probe = module("owned_process",root/"probe.py",data["probe.py"])
    event = d.events(out,start,cpu_start,512*1024**2)
    event("stage_start")
    stage = d.stage_runtime(preflight,helper,local/"runtime",event)
    stage_pin = save(out/"stage.json",stage)
    event("stage_complete")
    results = []
    for variant in MODES:
        directory = out/variant
        directory.mkdir(mode=0o700)
        command = [d.RUNTIME_SOURCE+"/bin/python","-I","-B","-c",d.BOOTSTRAP,str(root/Path(__file__).name),
            packet["files"][Path(__file__).name],"--packet",args.packet,"--packet-sha256",args.packet_sha256,
            "--mode","parent","--variant",variant,"--prefix",stage["staged_prefix"],"--output",str(directory/"parent")]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        event("parent_start",variant=variant)
        def validate():
            raw = (directory/"parent/terminal.json").read_bytes()
            value = json.loads(raw)
            need(value["status"]=="parent_child_observation_complete" and value["variant"]==variant
                 and type(value["child_returncode"]) is int,"parent observation missing")
            return hashlib.sha256(raw).hexdigest()
        results.append({"variant":variant,**d.run_child(command,env,directory,start+580,probe.kill_owned,validate)})
        event("parent_complete",variant=variant)
    save(out/"terminal.json",{"schema":VERSION,"status":"threading_observations_complete",
        "packet_sha256":args.packet_sha256,"stage_sha256":stage_pin,"sessions":results,
        "elapsed_seconds":time.monotonic()-start,"supervisor_cpu_seconds":time.process_time()-cpu_start,
        "model_service_qualified":False,"scientific_qualified":False,"model_loaded":False,
        "training_updates":0,"environment_responses":0})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("packet","packet-sha256","output"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--mode",choices=("supervisor","parent","worker"),default="supervisor")
    p.add_argument("--variant",choices=MODES)
    p.add_argument("--prefix")
    args = p.parse_args()
    out = Path(args.output).absolute()
    out.mkdir(mode=0o700,exist_ok=False)
    start,cpu_start = time.monotonic(),time.process_time()
    def interrupted(signum,frame):
        raise InterruptedError("threading diagnostic signal "+str(signum))
    for s in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):
        signal.signal(s,interrupted)
    signal.setitimer(signal.ITIMER_REAL,580 if args.mode=="supervisor" else 165)
    try:
        context = authenticate(args)
        if args.mode=="worker":
            worker(args,out)
        elif args.mode=="parent":
            parent(args,out,context)
        else:
            supervisor(args,out,context,start,cpu_start)
    except BaseException as exc:
        save(out/"failure.json",{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":
    main()
