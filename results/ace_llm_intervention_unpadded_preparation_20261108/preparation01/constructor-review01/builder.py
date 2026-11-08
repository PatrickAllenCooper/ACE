#!/usr/bin/env python3
"""CPU preparation from accepted saved tokens. No tokenizer/model/response calls.

Labels are deliberately fabricated for an artificial arithmetic/shape check.
This is neither collected supervision nor a scientific training dataset.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

REPO = Path('/Users/pat/code/ACE')
ROOT = REPO/'results/ace_llm_intervention_screen_prompt_cpu_20261108/terminal01/remote/launch'
PACKET_PIN = '0979cb607643b4b5d01fbf56c6e701a3e474d4d2bbda1e1c9f4e78e91251ccfa'
TOKENS_PIN = 'fb663a035eef20420fe461bc4dfa4fa6afa8462b38e32e1262c5661f666ff72c'
LONG = 'collider-uniform-noise0.25-long_spelling-last_all_actions'
TERMINAL = 'collider-uniform-noise0.25-long_spelling-full_history_terminal'

def need(ok, message):
    if not ok:
        raise ValueError(message)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read(path, pin):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'regular input required')
    raw = path.read_bytes()
    need(sha(raw) == pin, 'independent input pin mismatch: '+str(path))
    return raw

def save(root, name, raw):
    with (root/name).open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--acceptance', required=True)
    p.add_argument('--acceptance-sha256', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    started, cpu = time.monotonic(), time.process_time()
    packet_raw = read(ROOT/'inputs/packet.json', PACKET_PIN)
    token_raw = read(ROOT/'cpu01/worker/artifacts/tokenization.json', TOKENS_PIN)
    acceptance_raw = read(args.acceptance, args.acceptance_sha256)
    packet, tokens, acceptance = map(json.loads, (packet_raw, token_raw, acceptance_raw))
    # The caller additionally reviews the exact independently pinned acceptance.
    need(acceptance.get('job_id') == '35391340', 'wrong tokenizer execution acceptance')
    rows = {r['fixture_id']: r for r in tokens['rows']}
    need(len(rows) == 120 and all(r['status'] == 'fits' for r in rows.values()), 'incomplete actual tokenization')
    need(len(rows[LONG]['prompt_ids']) == 4638 and len(rows[TERMINAL]['prompt_ids']) == 4744,
         'fixed maximum-shape fixture changed')
    envelopes = {name: read(ROOT/'inputs'/(name+'.json'), packet['files'][name+'.json'])
                 for name in (LONG, TERMINAL)}
    prompts = {name: read(ROOT/'inputs'/(name+'.prompt.txt'), packet['files'][name+'.prompt.txt'])
               for name in (LONG, TERMINAL)}
    root = Path(args.output).absolute()
    root.mkdir(mode=0o700, exist_ok=False)
    def dump(name, value):
        save(root, name, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode())
    dump('intent.json', {'scope':'artificial', 'scientific_worlds':0, 'environment_responses':0,
        'tokenizer_calls':0, 'inference_calls':0, 'optimizer_updates':0,
        'capacity_packet_sha256':PACKET_PIN, 'tokenization_sha256':TOKENS_PIN,
        'capacity_acceptance_sha256':args.acceptance_sha256,
        'labels':'hand-written artificial labels, not scored supervision'})
    try:
        save(root, 'builder.py', Path(__file__).read_bytes())
        save(root, 'capacity-acceptance.json', acceptance_raw)
        save(root, 'capacity-packet.json', packet_raw)
        save(root, 'tokenization.json', token_raw)
        sys.path.insert(0, str(REPO/'scripts/research'))
        from llm_intervention_contract import canonical, digest
        from llm_intervention_supervision import VERSION as RECORD_VERSION, matched_views
        from llm_intervention_training_plan import qwen_adapter_shapes
        import llm_intervention_training_engine as engine
        names = ('llm_intervention_canonical_scoring.py', 'llm_intervention_training_engine.py',
                 'llm_intervention_training_plan.py', 'llm_intervention_supervision.py',
                 'llm_intervention_task.py', 'llm_intervention_contract.py')
        source_pins = {}
        (root/'source').mkdir()
        for name in names:
            raw = (REPO/'scripts/research'/name).read_bytes()
            save(root, 'source/'+name, raw)
            source_pins[name] = sha(raw)
        specifications = (
            (LONG, 'observe_all', ('observe_all', 'set_x_neg')),
            (LONG, 'set_x_neg', ('set_x_neg', 'observe_all')),
            (TERMINAL, None, None),
            (LONG, 'set_xy', ('set_xy', 'observe_all')),
        )
        prepared = []
        for index, (name, supervised, pair) in enumerate(specifications):
            rid = 'shape-'+str(index)
            envelope, row = json.loads(envelopes[name]), rows[name]
            need(sha(prompts[name]) == row['metadata']['prompt_sha256'], 'saved prompt pin changed')
            receipt = {'scope':'artificial', 'record_id':rid, 'world_id':990100+index,
                'fixture_id':name, 'envelope_sha256':digest(envelope),
                'tokenization_sha256':TOKENS_PIN, 'capacity_acceptance_sha256':args.acceptance_sha256,
                'labels':'fabricated arithmetic qualification only', 'environment_responses':0}
            receipt_raw = (canonical(receipt)+'\n').encode()
            receipt_pin = sha(receipt_raw)
            record = {'version':RECORD_VERSION, 'scope':'artificial', 'world_id':990100+index,
                'envelope':envelope, 'supervised_action_id':supervised,
                'preferences':([] if pair is None else [{'left':pair[0], 'right':pair[1], 'tie':False,
                    'chosen':pair[0], 'rejected':pair[1]}]), 'supervision_receipt_sha256':receipt_pin}
            raw = canonical(record).encode()
            views = matched_views(raw, record_sha256=sha(raw), supervision_receipt_sha256=receipt_pin,
                                  allowed_scope='artificial')
            need(views['language']['prompt'].encode() == prompts[name], 'unchanged full prompt required')
            rec = {'version':engine.VERSION, 'record_id':rid, 'scope':'artificial',
                'raw_record_utf8':raw.decode(), 'record_sha256':sha(raw),
                'supervision_receipt_sha256':receipt_pin, 'tokenizer_sha256':digest(packet['model_files']),
                'token_metadata':row['metadata'], 'prompt_ids':row['prompt_ids'], 'commands':row['commands'],
                'pad_token_id':151643, 'eos_token_id':151645,
                'supervised_command':views['language']['supervised_completion'],
                'strict_pairs':[list(x) for x in views['language']['preference_completions']],
                'tied_pairs':views['alignment']['tied_pairs'], 'provenance_qualified':False}
            prepared.append(rec)
            save(root, rid+'.record.json', raw)
            save(root, rid+'.receipt.json', receipt_raw)
        schedule = tuple([{'stage':'sft','record_id':r['record_id'],'pair_index':None} for r in prepared]+
                         [{'stage':'dpo','record_id':'shape-'+str(i),'pair_index':0} for i in (0,1)])
        plan = engine.prepare_training_plan(records=tuple(prepared),
            record_pins={r['record_id']:digest(r) for r in prepared}, schedule=schedule,
            numeric_recipe=engine.screen_numeric_recipe(), scope='artificial',
            execution_kind='qwen_screen_singletons', expected_adapter_shapes=qwen_adapter_shapes(),
            max_trainable_parameters=engine.MAX_PARAMETERS, checkpoint_steps=(3,5),
            outer_provenance_receipt_sha256=sha((root/'intent.json').read_bytes()))
        result = {'schema':'ace-screen-training-shape-preparation-v1', 'scope':'artificial',
            'training_plan':plan, 'training_plan_sha256':digest(plan),
            'capacity_acceptance_sha256':args.acceptance_sha256, 'capacity_packet_sha256':PACKET_PIN,
            'tokenization_sha256':TOKENS_PIN, 'source_sha256':source_pins,
            'scientific_qualified':False, 'training_execution_qualified':False,
            'labels':'fabricated only', 'environment_responses':0, 'tokenizer_calls':0}
        save(root, 'prepared.json', (canonical(result)+'\n').encode())
        dump('terminal.json', {'status':'prepared_only', 'prepared_sha256':sha((root/'prepared.json').read_bytes()),
            'plan_sha256':digest(plan), 'source_sha256':source_pins,
            'prompt_tokens':[len(r['prompt_ids']) for r in prepared],
            'dpo_command_lengths':[[len(r['commands'][c]) for c in r['strict_pairs'][0]] for r in prepared[:2]],
            'sft_steps':4, 'dpo_steps':2, 'process_cpu_seconds':time.process_time()-cpu,
            'elapsed_seconds':time.monotonic()-started, 'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'cost_scope':'this CPU preparation process only, not whole-cycle CPU',
            'model_loaded':False, 'scientific_worlds':0, 'environment_responses':0,
            'training_execution_qualified':False})
        print((root/'terminal.json').read_text())
    except BaseException as exc:
        dump('failure.json', {'type':type(exc).__name__, 'message':str(exc),
                             'elapsed_seconds':time.monotonic()-started})
        raise

if __name__ == '__main__':
    main()
