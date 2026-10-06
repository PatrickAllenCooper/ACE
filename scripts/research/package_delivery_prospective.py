"""Package committed workers and frozen descriptors; generate no responses."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile

from delivery_prospective_batch import WORKERS,load_descriptors,PILOT_FAILURE_RECEIPT
from runner_delivery_confirmation import read,write,sha,utc


def package(project,source,draft,gate,destination,revision):
    project,source,draft,gate,destination=map(lambda p:Path(p).resolve(),(project,source,draft,gate,destination))
    actual=subprocess.check_output(['git','rev-parse',revision],cwd=project,text=True).strip()
    files=sorted({'scripts/research/'+n for n in WORKERS}|{
        'scripts/research/delivery_prospective_pilot.py','scripts/research/package_delivery_prospective.py',
        'baselines.py','experiments/large_scale_scm.py',PILOT_FAILURE_RECEIPT})
    hashes={}
    for name in files:
        committed=subprocess.check_output(['git','show',actual+':'+name],cwd=project)
        h=hashlib.sha256(committed).hexdigest()
        if sha(project/name)!=h:raise ValueError('uncommitted worker/generator: '+name)
        hashes[name]=h
    # Immutable gate and original source must agree with the accepted campaign.
    if sha(gate)!='a81d0ac51965f71123ecc915764cf18c8166537a1d69f1053d096dd36ed3310f':
        raise ValueError('immutable attribution gate changed')
    pilot_registration=read(project/'results/delivery_prospective_preparation_20261006/pilot_registration.json')
    for name,h in pilot_registration['source_hashes'].items():
        if sha(source/name)!=h:raise ValueError('original source changed')
    manifest=read(draft/'manifest.json');load_descriptors(draft,manifest)
    destination.mkdir(parents=True,exist_ok=False);bundle=destination/'bundle';bundle.mkdir()
    for name in files:
        target=bundle/'project'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(project/name,target)
    write(bundle/'project/source_commit_receipt.json',{'at':utc(),'source_revision':actual,'files':hashes,
        'verification':'each worker/generator byte hash compared against git show revision:path before packaging',
        'confirmation_responses_evaluated':0})
    for name in pilot_registration['source_hashes']:
        target=bundle/'runner'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,target)
    for case in manifest['worlds']:
        target=bundle/'draft'/case.replace(':','-');target.mkdir(parents=True)
        for name in ('world.json','actions.json'):shutil.copyfile(draft/case.replace(':','-')/name,target/name)
    shutil.copyfile(draft/'manifest.json',bundle/'draft/manifest.json')
    shutil.copyfile(gate,bundle/'attribution_gate.json')
    with tarfile.open(destination/'bundle.tar','w') as archive:archive.add(bundle,arcname='bundle')
    receipt={'at':utc(),'source_revision':actual,'bundle_sha256':sha(destination/'bundle.tar'),
        'worker_generator_hashes':hashes,'original_source_hashes':pilot_registration['source_hashes'],
        'descriptor_manifest_sha256':sha(draft/'manifest.json'),'gate_sha256':sha(gate),
        'source_commit_receipt_sha256':sha(bundle/'project/source_commit_receipt.json'),
        'confirmation_responses_evaluated':0,'allocations_created':0}
    write(destination/'bundle_receipt.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('project','source','draft','gate','destination'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--revision',required=True)
    args=parser.parse_args();result=package(args.project,args.source,args.draft,args.gate,args.destination,args.revision)
    print({'source_revision':result['source_revision'],'bundle_sha256':result['bundle_sha256']})
