"""Build a new anonymous text-revision packet. Preserve all prior packets and studies."""
from pathlib import Path
import argparse, hashlib, json, zipfile
REPO=Path('/Users/pat/code/ACE')
BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
PREVIOUS=BASE/'ACE-TMLR-submission-review-20261101-01'
PREVIOUS_PIN='369cd061eede65896c54b79776aa75971d764a9bb949b5160f0153bfbf35fdd2'
MEMBERSHIP=BASE/'ACE-submission-review-20261024-02/source_membership.json'
MEMBERSHIP_PIN='148357c90b4b789ce6e76373b202dc590b100e6ea859a93f14643e2eb434795b'
BUNDLE_PIN='2cca8a3699bcee68adef1a7fe610df87a6518a39513f71b310e990d73183cd84'

def sha(raw): return hashlib.sha256(raw).hexdigest()
def encode(v):return (json.dumps(v,indent=2,sort_keys=True)+'\n').encode()
def read_pin(p,pin):
 b=p.read_bytes()
 if sha(b)!=pin:raise ValueError('pin mismatch: '+str(p))
 return json.loads(b)

def main():
 a=argparse.ArgumentParser();a.add_argument('--capture',required=True,type=Path);a.add_argument('--capture-sha256',required=True);a.add_argument('--render',required=True,type=Path);a.add_argument('--destination',required=True,type=Path);args=a.parse_args()
 inv=read_pin(args.capture/'inventory.json',args.capture_sha256)
 prev=read_pin(PREVIOUS/'submission_manifest.json',PREVIOUS_PIN)
 old=read_pin(MEMBERSHIP,MEMBERSHIP_PIN)
 entries={e['path']:e for e in inv['files']};payload={}
 for f in old['files']:
  path=f['path'];prefix='paper/aistats_ace_2027/'
  if not path.startswith(prefix) or '..' in Path(path).parts:raise ValueError('invalid source path')
  b=(args.capture/'source'/path).read_bytes();e=entries[path]
  if sha(b)!=e['sha256'] or len(b)!=e['bytes'] or b!=(REPO/path).read_bytes():raise ValueError('captured source mismatch')
  payload[path[len(prefix):]]=b
 if len(payload)!=29:raise ValueError('source/report membership')
 if sha((REPO/'paper/aistats_ace_2027/delivery_prospective_bundle.json').read_bytes())!=BUNDLE_PIN:raise ValueError('prospective bundle changed')
 comp=json.loads((args.render/'compile.json').read_text())
 if comp['returncode']!=0 or comp['source_sha256']!=sha(payload['paper.tex']) or (args.render/'paper.tex').read_bytes()!=payload['paper.tex']:raise ValueError('export source mismatch')
 # The original index remains unchanged in the repository and committed capture.
 original_index=payload['claim_index.json']; idx=json.loads(original_index)
 omitted={k:idx.pop(k) for k in ('pending_claims','removed_claims')}
 idx['publication_projection']={'scope':'Empirical claim mapping for the current manuscript. Earlier agenda annotations remain in the preserved historical index. No empirical entry or adverse result is removed.','historical_index_sha256':sha(original_index),'omitted_annotation_fields':sorted(omitted)}
 for k,v in json.loads(original_index).items():
  if k not in omitted and idx[k]!=v:raise ValueError('empirical index modification')
 payload['claim_index.json']=encode(idx)
 projection={'original_sha256':sha(original_index),'published_sha256':sha(payload['claim_index.json']),'omitted_fields':sorted(omitted),'all_other_original_fields_equal':True,'historical_original_preserved':True}
 payload['README_REVIEW.md']=b'''# Anonymous manuscript sources and saved reports

This supplement accompanies the revised manuscript. Compile paper.tex with a
standard LaTeX distribution supporting the included official TMLR style.
The style files and required notices are unchanged.

All original prospective CSVs and generated reports are retained byte for byte.
They include 640 scored cells, 320 fixed-init0 arm/history errors, 80 system log
ratios, history summaries and floor counts. The manuscript displays editorial
copies of generated tables with clearer labels and captions. Their original
numerical entries are unchanged. A new, explicitly descriptive initialization
summary averages saved log errors, without new tests or model selection.
The new absolute-scale summaries use the median of sqrt(NMSE) across the forty
system/history cells per arm and family.

claim_index.json is a publication projection of the preserved historical index.
Only obsolete pending/removed-agenda annotations are omitted. Its projection
record identifies the historical digest. Every empirical mapping, accounting
field and adverse-result boundary is retained. This projection is not the
original frozen index and does not change the underlying study evidence.
SOURCE_MANIFEST.json records hashes for internal integrity, not independent
proof of origin or of when the original protocol was frozen.

This is a source/report supplement, not the executable reproduction archive.
Original checkpoints, response ledgers, runtime closure and private execution
records are retained separately with their required named attribution. That
full archive is withheld from anonymous upload and needs a suitable later
release route. The saved reports permit statistical reconstruction, not an
independent reconstruction of original data collection or training chronology.
No new model fit, response, or inferential test was performed for this revision.
'''
 members={n:{'sha256':sha(b),'bytes':len(b)} for n,b in payload.items()}
 payload['SOURCE_MANIFEST.json']=encode({'scope':'Revised writing and unchanged saved reports, with explicitly identified historical-index projection','files':members})
 out=args.destination;out.mkdir(mode=0o700)
 (out/'ACE_manuscript.pdf').write_bytes((args.render/'paper.pdf').read_bytes())
 with zipfile.ZipFile(out/'ACE_anonymous_supplement.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for n,b in sorted(payload.items()):
   zi=zipfile.ZipInfo(n,(2026,11,2,0,0,0));zi.external_attr=0o100644<<16;zi.compress_type=zipfile.ZIP_DEFLATED;z.writestr(zi,b)
 if (out/'ACE_anonymous_supplement.zip').stat().st_size>100_000_000:raise ValueError('upload size')
 with zipfile.ZipFile(out/'ACE_anonymous_supplement.zip') as z:
  if len(z.namelist())!=len(payload) or set(z.namelist())!=set(payload):raise ValueError('archive membership')
  for n,b in payload.items():
   if z.read(n)!=b:raise ValueError('archive readback')
 author='''# Revised TMLR packet for final author review

Only ACE_manuscript.pdf and ACE_anonymous_supplement.zip are intended for
anonymous submission. Keep this README and submission_manifest.json private.
The November 1 packet and full attributed reproduction archive remain unchanged.

This revision addresses the November 2 model panel. It clarifies pooled versus
history-specific results, unequal training budgets, conditional initialization,
absolute errors and the extreme sensitivity failure. It explains the archived
4,803 paid rows, adds the warm-starting precedent, shortens process prose, and
moves quantization proofs to an appendix. It preserves all adverse evidence.
No new experiments or inferential tests were conducted.

The small supplement retains all original saved scientific reports. A separately
identified projection of the historical claim index omits obsolete agenda
annotations while preserving all empirical fields. The full executable archive
with mandatory named notices remains private and is not an anonymous attachment.

Confirmed by Patrick: Patrick Cooper (patrick.cooper@colorado.edu), followed by
Alvaro Velasquez (alvaro.velasquez@colorado.edu), both University of Colorado
Boulder. Alvaro is the only coauthor and has agreed to coauthorship. No funding,
competing interests, or overlapping submission were reported. Agreement to
coauthorship does not establish approval of this final manuscript.

Both authors must verify the final paper and AI-assisted work, finalize their
OpenReview profiles and applicable declarations, and approve this exact PDF
and supplement before submission. No upload or submission has occurred.
Model reviews are internal feedback and do not predict a human venue decision.
'''
 (out/'README_AUTHOR.md').write_text(author)
 manifest={'schema':'ace-tmlr-panel-revision-successor-v1','scope':'Text revision and descriptive saved-data summaries, not new science or full reproduction','writing_revision':inv['revision'],'writing_inventory_sha256':args.capture_sha256,'previous_submission_manifest_sha256':PREVIOUS_PIN,'full_attributed_archive_predecessor_sha256':prev['full_attributed_archive_predecessor_sha256'],'replay_input_manifest_sha256':prev['replay_input_manifest_sha256'],'replay_acceptance_sha256':prev['replay_acceptance_sha256'],'report_acceptance_sha256':prev['report_acceptance_sha256'],'prospective_bundle_sha256':BUNDLE_PIN,'claim_index_projection':projection,'export_receipt_sha256':sha((args.render/'compile.json').read_bytes()),'builder_sha256':sha(Path(__file__).read_bytes()),'artifacts':{n:{'sha256':sha((out/n).read_bytes()),'bytes':(out/n).stat().st_size} for n in ['ACE_manuscript.pdf','ACE_anonymous_supplement.zip','README_AUTHOR.md']},'supplement_members':len(payload),'new_model_fits':0,'new_responses':0,'new_inferential_tests':0,'descriptive_saved_data_summaries_added':True,'human_author_details_confirmed':True,'both_authors_final_version_approved':False,'human_submission_approved':False,'venue_accepted':False}
 (out/'submission_manifest.json').write_bytes(encode(manifest))
 print(json.dumps({'destination':str(out),'manifest_sha256':sha(encode(manifest)),'artifacts':manifest['artifacts'],'supplement_members':len(payload)},indent=2))
if __name__=='__main__':main()
