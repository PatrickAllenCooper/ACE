"""Actual pinned saved-metadata capture only; never installs the history callback."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
pins = json.loads((root/'input-pins.json').read_bytes())
data = {}
for name, record in pins.items():
    value = (root/name).read_bytes()
    if hashlib.sha256(value).hexdigest() != record['sha256']:
        raise ValueError('saved qualification input pin differs: '+name)
    data[name] = value
path = root/'llm_intervention_cpu_history_admission.py'
spec = importlib.util.spec_from_file_location('qualified_history_adapter', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)  # Only the new stdlib adapter, not original project source.
history = module.capture_history(
    acceptance_raw=data['cpu-acceptance.json'], acceptance_sha256=pins['cpu-acceptance.json']['sha256'],
    bundle_raw=data['campaign-bundle.json'], bundle_sha256=pins['campaign-bundle.json']['sha256'],
    readset_raw=data['cpu-history-readset.json'], readset_sha256=pins['cpu-history-readset.json']['sha256'],
    campaign_source_raw=data['original-campaign.py'], campaign_source_sha256=pins['original-campaign.py']['sha256'],
)
result = {
    'status': 'actual_saved_metadata_capture_passed',
    'readset_sha256': history.readset_sha256,
    'cpu_acceptance_sha256': history.acceptance_sha256,
    'bundle_sha256': history.bundle_sha256,
    'original_campaign_source_sha256': history.campaign_source_sha256,
    'campaign_sha256': history.campaign_sha256,
    'file_count': len(history._files),
    'raw_metadata_bytes': sum(map(len, history._files.values())),
    'cpu_attempt_count': len(history._attempt_pins),
    'install_called': False,
    'original_project_source_executed': False,
    'live_metadata_checked': False,
    'full_preflight_qualified': False,
    'scientific_qualified': False,
}
with (root/'qualification.json').open('x') as f:
    f.write(json.dumps(result, sort_keys=True, indent=2)+'\n')
print(json.dumps(result, sort_keys=True))
