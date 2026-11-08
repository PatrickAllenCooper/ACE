"""Prepare a new private notice/metadata plan; no public approval or study work.

Original hash-bound pyproject metadata is retained privately. Only three optional
organizational metadata fields are projected; license/dependencies stay intact.
Unknown Runner copyright/grant authority remains an explicit release blocker.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from build_delivery_release import write, reconcile
from verify_delivery_release import sha, project_toml_bytes, VERIFIER_SHA256

OMIT = ['tool.poetry.authors', 'tool.poetry.homepage', 'tool.poetry.repository']
RUNNER_REVISION = 'e9f811fbc68fb70681c3d89182d8ebe024882ce3'
RUNNER_METADATA = '852305cfad1672b2ec2aba5690d7fb5c919a3ec9978ad8f2c9fe0e9d07ed0e9f'
CHAMBERS_REVISION = '0fa8222dc761829270c8959e0ba53b261b075e1c'
CHAMBERS_ARCHIVE = '584490fc05191c21debd75c70c94ee80358f66482694f98c21f405d695f1b3e9'
REVIEWED = {
    'LICENSE':'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',
    'results/causal_chambers_source_audit_20261001/source_receipt.json':'c788c12d80a6f388e522e0f2bb9d55997904cfd7055c62235e36b6bb67a4e60f',
    'results/causal_chambers_source_audit_20261001/README.md':'d704cac78de98118087a8c3a5e0837cf9acf723deafdb8450ef1a90cd37b49f5',
    'results/causal_chambers_source_audit_20261001/malus.py':'9f00fb619263e2ea8565b377fd6adf8fd5edc1e5997fc5998a0fb12975f6f7d8',
    'results/causal_chambers_archive_audit_20261001/audit.json':'ebe2648a5a16b7fe88889c7933e92c764068879b61afb9bb26013fea43796b69'}


def captured(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('notice/provenance snapshot changed')
    return raw


def outputs(private_dir, out, inputs):
    private_dir, out = map(lambda p:Path(p).resolve(), (private_dir,out))
    if private_dir.exists() or out.exists() or out.is_relative_to(private_dir) or private_dir.is_relative_to(out):
        raise ValueError('exclusive separate outputs required')
    for p in map(lambda p:Path(p).resolve(), inputs):
        if any(a==p or a.is_relative_to(p) or p.is_relative_to(a) for a in (private_dir,out)):
            raise ValueError('outputs overlap original sources')
    return private_dir,out


def extend(plan_file, expected_plan, repo, private_dir, out):
    raw = captured(plan_file, expected_plan); plan = copy.deepcopy(json.loads(raw))
    repo = Path(repo).resolve()
    private_dir,out = outputs(private_dir,out,[plan_file,repo])
    indexed = {f['path']:f for f in plan['files']}
    if len(indexed)!=len(plan['files']) or any(n.startswith('notices/') for n in indexed):
        raise ValueError('duplicate or already extended plan')
    name = 'source/runner/pyproject.toml'; entry = indexed[name]
    if entry['original_sha256'] != RUNNER_METADATA or entry.get('transform',{'kind':'identity'}) != {'kind':'identity'}:
        raise ValueError('exact original archived runner metadata required')
    source = reconcile(entry['sources'],RUNNER_METADATA)
    metadata = captured(source,RUNNER_METADATA)
    transform = {'kind':'project-toml','omit':OMIT}
    projected = project_toml_bytes(metadata,transform)
    notice_snapshots = {n:captured(repo/n,h) for n,h in REVIEWED.items()}
    source_receipt = json.loads(notice_snapshots['results/causal_chambers_source_audit_20261001/source_receipt.json'])
    if source_receipt['revision'] != CHAMBERS_REVISION:
        raise ValueError('pinned Chambers source revision differs')
    records = {Path(f['path']).name:f for f in source_receipt['files']}
    audit_root = repo/'results/causal_chambers_source_audit_20261001'
    readme = notice_snapshots['results/causal_chambers_source_audit_20261001/README.md']
    malus = notice_snapshots['results/causal_chambers_source_audit_20261001/malus.py']
    if records['README.md']['sha256'] != hashlib.sha256(readme).hexdigest() or records['malus.py']['sha256'] != hashlib.sha256(malus).hexdigest():
        raise ValueError('pinned declaration source differs')
    if b'CC BY 4.0' not in readme or b'MIT license' not in readme:
        raise ValueError('recorded Chambers license categories differ')
    archive = json.loads(notice_snapshots['results/causal_chambers_archive_audit_20261001/audit.json'])
    if archive['archive_sha256'] != CHAMBERS_ARCHIVE:
        raise ValueError('recorded Chambers archive differs')
    # Copy the complete existing comment notice, no invented author/permission.
    prefix = malus.decode().split('\n\n\n',1)[0]
    if not prefix.startswith('# MIT License') or '# SOFTWARE.' not in prefix:
        raise ValueError('complete original generator notice required')
    notice = '\n'.join(line[2:] if line.startswith('# ') else '' if line=='#' else line
                        for line in prefix.splitlines())+'\n'
    ace_license = notice_snapshots['LICENSE']
    if b'Apache License' not in ace_license or b'Version 2.0' not in ace_license:
        raise ValueError('ACE license differs')
    inputs = [plan_file,repo,source,*[s for f in plan['files'] for s in f['sources']]]
    private_dir,out = outputs(private_dir,out,inputs)
    bindings=[b for b in plan['bindings'] if b['artifact']==name]
    if len(bindings)!=1 or bindings[0]['record']!='F/protocol.json' or bindings[0]['pointer']!='/source_hashes/pyproject.toml' or bindings[0]['digest']!='sha256':
        raise ValueError('exact original F metadata binding required')
    verifier_bytes = captured(repo/'scripts/research/verify_delivery_release.py',VERIFIER_SHA256)
    private_dir.mkdir(parents=True,exist_ok=False)
    (private_dir/'input_plan.json').write_bytes(raw)
    (private_dir/'original_pyproject.toml').write_bytes(metadata)
    for original_name, snapshot in notice_snapshots.items():
        target = private_dir/'original_notices'/original_name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(snapshot)
    (private_dir/'verify_delivery_release.py').write_bytes(verifier_bytes)
    (private_dir/'projected_pyproject.toml').write_bytes(projected)
    def add(name, content, role):
        target=private_dir/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content if isinstance(content,bytes) else content.encode())
        plan['files'].append({'path':name,'sources':[str(target)],'original_sha256':sha(target),
                             'role':role,'transform':{'kind':'identity'}})
    add('notices/ACE_APACHE_2_0.txt',ace_license,'license-notice')
    add('notices/CHAMBERS_MIT_GENERATOR.txt',notice,'third-party-license-notice')
    add('notices/CHAMBERS_DATASET.md', f'''# Causal Chambers dataset attribution\n\nDataset: lt_malus_v1. Authors requested by the pinned source citation:\nJuan L. Gamella, Jonas Peters and Peter Bühlmann.\nCausal chambers as a real-world physical testbed for AI methodology,\nNature Machine Intelligence (2025), DOI10.1038/s42256-024-00964-x.\n\nCSV/image data: Creative Commons Attribution4.0 International,\nhttps://creativecommons.org/licenses/by/4.0/ . Software has separate MIT terms.\nSource: https://github.com/juangamella/causal-chamber/tree/{CHAMBERS_REVISION}/datasets/lt_malus_v1\nData: https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/lt_malus_v1.zip\nArchive SHA256:{CHAMBERS_ARCHIVE}. Original archive bytes retained unchanged;\nanalysis/splits/checkpoints are separate derived study artifacts.\nThe original source README accompanies these notices; its hash is recorded\nin the metadata provenance object. These notices do not license Runner source.\n''','third-party-dataset-attribution')
    add('notices/CHAMBERS_SOURCE_README.md',readme,'third-party-original-license-declaration')
    status={'public_release_approved':False,'runner_grant_authority_resolved':False,
        'runner_revision':RUNNER_REVISION,'runner_license_declaration':'MIT; authoritative copyright notice unresolved',
        'metadata_projection':{'path':name,'original_sha256':RUNNER_METADATA,
            'derived_sha256':hashlib.sha256(projected).hexdigest(),'omit':OMIT,
            'method':'explicit optional TOML metadata projection; all other parsed values unchanged; original private bytes retained'},
        'chambers_revision':CHAMBERS_REVISION,'chambers_archive_sha256':CHAMBERS_ARCHIVE,
        'chambers_original_README_sha256':records['README.md']['sha256'],
        'chambers_original_generator_sha256':records['malus.py']['sha256'],
        'remaining':['authoritative Runner copyright/license grant','full worker disposition and human anonymity review',
                     'original B full acceptance and actual release replay','complete accounting and human submission approval'],
        'new_fits':0,'new_responses':0}
    add('notices/provenance.json',json.dumps(status,indent=2)+'\n','derived-notice-provenance')
    # Preserve the ORIGINAL digest in F's protocol; bind it to original_sha256,
    # not the different newly projected bytes. No historical receipt is edited.
    entry['sources']=[str(private_dir/'original_pyproject.toml')];entry['transform']=transform
    bindings[0]['digest']='original_sha256'
    plan['bindings'].extend([
        {'record':'notices/provenance.json','pointer':'/metadata_projection/original_sha256','artifact':name,'digest':'original_sha256'},
        {'record':'notices/provenance.json','pointer':'/metadata_projection/derived_sha256','artifact':name,'digest':'sha256'},
        {'record':'notices/provenance.json','pointer':'/chambers_original_README_sha256','artifact':'notices/CHAMBERS_SOURCE_README.md','digest':'sha256'}])
    verifier=indexed['verify_delivery_release.py']
    verifier.update(sources=[str(private_dir/'verify_delivery_release.py')],
                    original_sha256=VERIFIER_SHA256,transform={'kind':'identity'})
    plan['status']='private notice/metadata preparation; Runner grant, B acceptance and public approval pending'
    write(out,plan)
    write(private_dir/'derivation.json',{'input_plan_sha256':expected_plan,'output_plan_sha256':sha(out),
        'metadata_projection':status['metadata_projection'],'original_candidate_changed':False,
        'notice_authority_scope':'ACE root license and pinned Chambers declarations; no inferred Runner grant',
        'notice_source_snapshots':REVIEWED,'executing_verifier_sha256':VERIFIER_SHA256,
        'public_release_approved':False,'new_fits':0,'new_responses':0})
    return {'plan_sha256':sha(out),'files':len(plan['files']),'bindings':len(plan['bindings']),
            'public_release_approved':False,'runner_grant_authority_resolved':False,'new_fits':0,'new_responses':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan-file','repo','private-dir','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-plan',required=True)
    print(json.dumps(extend(**vars(parser.parse_args())),indent=2))
