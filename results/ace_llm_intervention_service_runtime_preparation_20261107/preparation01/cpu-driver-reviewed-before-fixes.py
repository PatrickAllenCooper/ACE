"""Bounded CPU-only preflight of two pinned prospective model-service manifests.

Stages the existing archive once, then checks imports and tokenization in two
fresh isolated processes. No model weights are loaded. External scheduler
completion, complete custody and independent result acceptance remain required.
"""
from __future__ import annotations

import argparse
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

VERSION = "ace-model-service-cpu-preflight-v1"
DIAGNOSTIC_SHA = "31bb109e2ce9603e72fe97e1cb3fcfdff2b1a72b3f7eaf74431ca92f1cd7c329"
RESOURCES = {"account":"ucb736_asc2", "partition":"acpu", "qos":"cpu-normal",
             "cpus":1, "memory_gib":4, "wall_seconds":600, "gpus":0, "requeue":False}
SERVICES = ["frozen.json", "adapter.json"]
FIXTURES = {"case0.json", "case3.json", "stress.json"}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    raw=(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with Path(path).open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return sha(raw)


def captured_module(name, path, raw):
    module=types.ModuleType(name)
    module.__file__=str(path)
    exec(compile(raw,str(path),"exec"),vars(module))
    return module


def allocation_fields(raw, job):
    fields=dict(re.findall(r"(\w+)=(\S+)",raw))
    expected={"JobId":job,"JobState":"RUNNING","Account":"ucb736_asc2",
        "Partition":"acpu","QOS":"cpu-normal","NumCPUs":"1","NumTasks":"1",
        "NumNodes":"1","TimeLimit":"00:10:00","Requeue":"0","Restarts":"0"}
    tres=fields.get("AllocTRES","").split(",")
    need(all(fields.get(k)==v for k,v in expected.items()) and "cpu=1" in tres
         and "mem=4G" in tres and not any("gres/" in x for x in tres),
         "allocation differs from fixed one-CPU preflight")
    return fields


def authenticate(root, packet, data, service, diagnostic):
    need(set(packet)=={"schema","files","services","archive","manifest","runtime_source","resources"}
         and packet["schema"]==VERSION and packet["services"]==SERVICES
         and packet["resources"]==RESOURCES, "CPU packet contract differs")
    needed=set(service.SOURCE_NAMES)|{"qualify_llm_intervention_service_cpu.py",
        "llm_intervention_runtime_archive.py","probe.py","frozen.json","adapter.json",
        "trained_adapter.pt","gpu09_terminal.json"}|FIXTURES
    need(set(data)==needed and set(packet["files"])==needed, "complete CPU input roster differs")
    need(packet["files"]["qualify_llm_intervention_runtime_metadata.py"]==DIAGNOSTIC_SHA
         and packet["files"]["llm_intervention_runtime_archive.py"]==diagnostic.HELPER_SHA
         and packet["files"]["probe.py"]==diagnostic.PROBE_SHA, "accepted helper source differs")
    for key,pin in (("archive",diagnostic.ARCHIVE_SHA),("manifest",diagnostic.MANIFEST_SHA)):
        record=packet[key]
        need(set(record)=={"path","sha256"} and record["sha256"]==pin
             and type(record["path"]) is str and Path(record["path"]).is_absolute(),
             "original runtime input pin differs")
    need(packet["runtime_source"]==diagnostic.RUNTIME_SOURCE, "original runtime identity differs")
    freezes=[service.validate_freeze(service.decode(data[n])) for n in SERVICES]
    for freeze in freezes:
        need(freeze["scope"]=="artificial" and freeze["sources"]=={n:packet["files"][n] for n in service.SOURCE_NAMES}
             and freeze["limits"]=={"startup_seconds":180,"request_seconds":30,
                 "idle_seconds":30,"total_seconds":240,"max_requests":4},
             "service source, scope or deadline differs")
    need(freezes[0]["treatment"]=="frozen" and freezes[1]["treatment"]=="artificial_adapter"
         and freezes[0]["model_files"]==freezes[1]["model_files"]
         and freezes[0]["model_directory"]==freezes[1]["model_directory"]
         and freezes[1]["adapter"]=={"file":"trained_adapter.pt",
             "sha256":packet["files"]["trained_adapter.pt"],"training_receipt":"gpu09_terminal.json",
             "training_receipt_sha256":packet["files"]["gpu09_terminal.json"],"training_scope":"artificial"},
         "frozen and artificial-adapter bindings differ")
    previous=service.decode(data["gpu09_terminal.json"])
    need(previous["status"]=="complete" and previous["adapter_sha256"]==packet["files"]["trained_adapter.pt"],
         "saved artificial adapter lineage differs")
    return freezes


def validate_session(out, freeze_sha, stage_sha, freeze, files, service, envelope_pins):
    """Verify saved CPU receipts, without model imports or replaying tokenization."""
    def read(name):
        return service.decode((out/name).read_bytes())
    need(not list(out.rglob("failure.json")), "service failure cannot qualify")
    terminal=read("terminal.json")
    need(terminal.get("status")=="cpu_preflight_complete"
         and terminal.get("freeze_sha256")==freeze_sha
         and terminal.get("runtime_stage_sha256")==stage_sha
         and all(terminal.get(k) is False for k in ("model_loaded","adapter_deserialized",
             "explicit_cuda_api_called","service_qualified","scientific_execution_qualified"))
         and terminal.get("new_training_updates")==0 and terminal.get("environment_responses")==0,
         "CPU terminal scope/binding differs")
    authenticated=read("authenticated.json")
    need(authenticated["freeze_sha256"]==freeze_sha and authenticated["identity"]==service.identity(freeze)
         and authenticated["runtime"]==service.RUNTIME, "authenticated service identity differs")
    need(set(authenticated["models"])==service.MODEL_NAMES
         and all(row["sha256"]==freeze["model_files"][name] and type(row["bytes"]) is int and row["bytes"]>0
                 for name,row in authenticated["models"].items()), "model input authentication differs")
    bound=read("runtime-stage-binding.json")
    need(bound["stage_sha256"]==stage_sha and bound["contract"]==service.RUNTIME,
         "runtime stage receipt differs")
    for name in ("runtime-after-service-imports.json","runtime-after-cpu-tokenization.json"):
        observed=read(name)
        versions={k:v["version"] for k,v in observed["packages"].items()}
        versions.update(observed["additional_package_versions"])
        need(versions==service.RUNTIME["imported_versions"]
             and observed["generated_provenance"]["verified"] is True,
             "actual imported runtime or generated provenance differs")
        projects=observed["captured_project_modules"]
        expected={Path(n).stem:freeze["sources"][n] for n in service.SOURCE_NAMES
                  if n!="llm_intervention_model_service.py"}
        need(set(expected)<=set(projects) and set(projects)<=set(expected)|{"__main__","__mp_main__"}
             and all(projects[n]["sha256"]==pin for n,pin in expected.items()),
             "complete service import closure differs")
    tokens=read("cpu-tokenization.json")
    rows=tokens["fixtures"]
    need(len(rows)==3 and {r["name"] for r in rows}==FIXTURES
         and all(tokens[k] is False for k in ("model_weights_loaded","adapter_deserialized","explicit_cuda_api_called"))
         and tokens["new_training_updates"]==tokens["environment_responses"]==0,
         "CPU tokenization scope/roster differs")
    for r in rows:
        need(r["fixture_sha256"]==files[r["name"]]
             and r["envelope_sha256"]==envelope_pins[r["name"]]
             and r["max_tokens"]==8192 and r["max_new_tokens"]==32 and r["enable_thinking"] is False
             and len(r["prompt_ids"])==r["prompt_tokens"] and r["prompt_tokens"]+32<=8192
             and r["completion_ids"] and len(r["completion_ids"])==r["legal_commands"]
             and max(map(len,r["completion_ids"]))==r["longest_command_tokens"]<=32
             and all(ids and ids[-1]==151645 for ids in r["completion_ids"]),
             "saved token counts/pins differ")
    phases=[read(p.name)["phase"] for p in out.glob("phase-*.json")]
    need(all(phases.count(n)==1 for n in ("service_cpu_import_start","service_cpu_import_complete",
         "cpu_tokenizer_load_start","cpu_tokenization_complete"))
         and not any(n in phases for n in ("model_cpu_load_start","model_ready","generation_complete")),
         "CPU phase sequence differs")
    return sha((out/"terminal.json").read_bytes())


def child_command(prefix, root, name, files, directory, stage_path, stage_pin, bootstrap):
    command=[str(prefix/"bin/python"),"-I","-B","-c",bootstrap,
        str(root/"llm_intervention_model_service.py"),files["llm_intervention_model_service.py"],
        "--freeze",str(root/name),"--freeze-sha256",files[name],"--output",str(directory/"worker"),
        "--runtime-stage",str(stage_path),"--runtime-stage-sha256",stage_pin,"--verify-only"]
    for fixture in sorted(FIXTURES):
        command.extend(["--fixture",fixture+"="+files[fixture]])
    return command


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("packet","packet-sha256","output"):
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    start,cpu=time.monotonic(),time.process_time()
    out=Path(args.output).absolute()
    out.mkdir(mode=0o700,exist_ok=False)
    def interrupted(signum,frame):
        raise InterruptedError("CPU preflight signal "+str(signum))
    for s in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):
        signal.signal(s,interrupted)
    signal.setitimer(signal.ITIMER_REAL,580)
    results=[]
    try:
        root=Path(args.packet).absolute().parent
        raw=Path(args.packet).read_bytes()
        need(sha(raw)==args.packet_sha256,"independent CPU packet pin differs")
        # The diagnostic source has an independent fixed pin before it executes.
        diagnostic_raw=(root/"qualify_llm_intervention_runtime_metadata.py").read_bytes()
        need(sha(diagnostic_raw)==DIAGNOSTIC_SHA,"accepted diagnostic source differs")
        d=captured_module("cpu_preflight_diagnostic",root/"qualify_llm_intervention_runtime_metadata.py",diagnostic_raw)
        packet=d.decode(raw)
        data={}
        for name,expected in packet["files"].items():
            need(type(name) is str and Path(name).name==name and name not in ("",".",".."),
                 "plain packet filename required")
            data[name]=d.pinned(root/name,expected)
        need(Path(__file__).absolute()==root/"qualify_llm_intervention_service_cpu.py",
             "entrypoint must be frozen beside packet")
        service=captured_module("cpu_preflight_service",root/"llm_intervention_model_service.py",
                                 data["llm_intervention_model_service.py"])
        freezes=authenticate(root,packet,data,service,d)
        save(out/"packet.json",packet)
        job=os.environ.get("SLURM_JOB_ID","")
        need(job.isdigit() and not os.environ.get("SLURM_JOB_GPUS") and not os.environ.get("SLURM_STEP_GPUS"),
             "CPU Slurm job required")
        result=subprocess.run(["scontrol","show","job","-o",job],capture_output=True,text=True,timeout=10)
        save(out/"allocation.json",{"stdout":result.stdout,"stderr":result.stderr,"returncode":result.returncode})
        need(result.returncode==0,"scheduler query failed")
        allocation_fields(result.stdout,job)
        scratch=Path(os.environ.get("SLURM_SCRATCH",""))
        need(scratch.is_absolute(),"absolute node scratch required")
        scratch=scratch.resolve(strict=True)
        need(scratch.is_dir() and scratch.stat().st_uid==os.getuid(),"owned node scratch required")
        local=scratch/("ace-service-cpu-"+job)
        local.mkdir(mode=0o700,exist_ok=False)
        helper=captured_module("cpu_archive_helper",root/"llm_intervention_runtime_archive.py",data["llm_intervention_runtime_archive.py"])
        probe=captured_module("cpu_owned_process",root/"probe.py",data["probe.py"])
        event=d.events(out,start,cpu,512*1024**2)
        event("stage_start")
        stage=d.stage_runtime(packet,helper,local/"runtime",event)
        stage_pin=save(out/"stage.json",stage)
        event("stage_complete",staged_prefix=stage["staged_prefix"])
        prefix=Path(stage["staged_prefix"])
        env=dict(os.environ)
        for name in ("PYTHONHOME","PYTHONPATH","LD_PRELOAD"):
            env.pop(name,None)
        env.update(LD_LIBRARY_PATH=str(prefix/"lib"),CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",
            HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
        for ordinal,(name,freeze) in enumerate(zip(SERVICES,freezes)):
            directory=out/("session-"+str(ordinal))
            directory.mkdir(mode=0o700,exist_ok=False)
            command=child_command(prefix,root,name,packet["files"],directory,out/"stage.json",stage_pin,d.BOOTSTRAP)
            event("service_preflight_start",ordinal=ordinal,manifest=name)
            before_child=resource.getrusage(resource.RUSAGE_CHILDREN)
            result=d.run_child(command,env,directory,start+580,probe.kill_owned,
                lambda:validate_session(directory/"worker",packet["files"][name],stage_pin,freeze,packet["files"],service,
                    {n:service.digest(service.decode(data[n])) for n in FIXTURES}))
            after_child=resource.getrusage(resource.RUSAGE_CHILDREN)
            result.update(child_and_cleanup_cpu_seconds=after_child.ru_utime+after_child.ru_stime
                -before_child.ru_utime-before_child.ru_stime,
                cumulative_child_peak_rss_bytes=after_child.ru_maxrss*1024)
            results.append({"manifest":name,**result})
            event("service_preflight_complete",ordinal=ordinal,**result)
        need(time.monotonic()-start<580,"CPU preflight exceeded absolute deadline")
        save(out/"terminal.json",{"schema":VERSION,"status":"cpu_preflight_complete",
            "packet_sha256":args.packet_sha256,"stage_sha256":stage_pin,"sessions":results,
            "elapsed_seconds":time.monotonic()-start,"supervisor_cpu_seconds":time.process_time()-cpu,
            "supervisor_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            "model_weights_loaded":False,"training_updates":0,"environment_responses":0,
            "service_qualified":False,"scientific_execution_qualified":False})
    except BaseException as exc:
        save(out/"failure.json",{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc(),
            "completed_sessions":results,"model_weights_loaded":False,"training_updates":0,"environment_responses":0})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":
    main()
