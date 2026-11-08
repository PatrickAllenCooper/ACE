from pathlib import Path
import argparse,json,hashlib,shutil
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
REPO=Path('/Users/pat/code/ACE')
FREEZE='15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d'
SOURCE='a9346022f145cc29439325b6f291597b2e6ec037'
CLAIM='58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29'
def raw(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def checked(path,pin):
 b=Path(path).read_bytes();assert sha(b)==pin;return b
ap=argparse.ArgumentParser();ap.add_argument('--review',required=True);ap.add_argument('--review-sha256',required=True);a=ap.parse_args()
f=checked(REPO/'results/ace_llm_intervention_scientific_contract_20261108/candidate03/scientific-freeze.json',FREEZE)
review=checked(a.review,a.review_sha256)
claim=checked(BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace.json',CLAIM)
assert json.loads(f)['source_commit']==SOURCE
root='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_02'
local_registry=str(BASE/'ACE-LLM-intervention-scientific-screen-20261107.namespace-recovery02.json')
remote_registry='/scratch/alpine/paco0228/ACE/results/.ace-scientific-screen-20261107.namespace-recovery02.json'
authority=dict(schema='ace-scientific-realization-recovery-authority-v1',status='authorized',recovery_id='realization02',
 failed_job_id='35402725',failure_acceptance_sha256='5da56ce061706f29a23c9a37a898bcbac128d16421f6cfd30731655208021a8d',
 no_draw_proof_sha256='5fb94f9d07da3acd25c979250471037e8981066e0b201716f8c32d0b681fb9ef',
 predecessor_freeze_sha256='57a3e6dd27ded5cec007be235b799a2bec711fc2c5630ff16fef87d74273cd28',
 successor_freeze_sha256=FREEZE,source_commit=SOURCE,original_namespace_claim_sha256=CLAIM,
 accounting_sha256='63cd36dfd9c0dcda4ef8ceec7b99d91ecfa696676cd1d99149256c6a0fdb3727',
 original_root='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_01',
 recovery_root=root,local_recovery_registry=local_registry,remote_recovery_registry=remote_registry,
 one_submission_only=True,automatic_retry_allowed=False,replacement_worlds_allowed=False)
adoption=dict(schema='ace-scientific-contract-adoption-v1',status='adopted',freeze_sha256=FREEZE,
 review_sha256=a.review_sha256,required_findings=0,adopted_before_first_draw=True)
namespace=dict(schema='ace-scientific-screen-namespace-descriptor-v1',status='predeclared',freeze_sha256=FREEZE,
 proposal_sha256='8e3214958f671dfee5ef0a2d7adc70e8f5cdc26edf0fbeff02e7925921ff13da',
 proposal_semantic_sha256='585536b1408054ecebc7b7bb0e80055fa153b5bc7a699d8f2927db25ee745ea8',
 world_ids=dict(policy_train=list(range(202611070000,202611070032)),development=list(range(202611070100,202611070124))),
 namespaces_from='exact original proposal bindings',local_registry=local_registry,remote_registry=remote_registry,
 realization_root=root,retry_allowed=False,replacement_worlds=False,recovery_authority_sha256=sha(raw(authority)),
 original_namespace_claim_sha256=CLAIM,failed_job_id='35402725')
local=REPO/'results/ace_llm_intervention_scientific_contract_20261108/adoption02'
ext=BASE/'ACE-LLM-intervention-scientific-contract-adoption-20261108-02'
assert not local.exists() and not ext.exists();ext.mkdir(mode=0o700)
files={'adoption.json':raw(adoption),'namespace-admission.json':raw(namespace),'recovery-authority.json':raw(authority),
 'freeze-review.md':review,'source.py':Path(__file__).read_bytes()}
for n,b in files.items():
 with (ext/n).open('xb') as h:h.write(b)
inventory={n:sha(b) for n,b in files.items()}
with (ext/'inventory.json').open('xb') as h:h.write(raw(dict(files=inventory,scope='explicit parent adoption after independent candidate review; no namespace claimed or submission')))
shutil.copytree(ext,local)
assert all((local/n).read_bytes()==b for n,b in files.items())
print(json.dumps(inventory))
