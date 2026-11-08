#!/usr/bin/env python3
"""Offline two-stage open-model canary; reads public task and reveal files only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


MODEL_REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
STAGE1_SCHEMA = ('Return exactly one JSON object with keys '
    'candidate_set_doX1_mean_interval (two numbers or rational strings), '
    'observations_identify_candidate (Boolean), and action '
    '(one legal action code or abstain_unresolved). No prose.')
STAGE2_SCHEMA = ('Return exactly one JSON object with posterior_A_B '
    '(two numbers or rational strings summing to one) and '
    'posterior_supported_candidates (ordered subset of ["A","B"] with positive posterior). No prose.')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_rows(path: Path) -> dict[str, dict]:
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = {item['id']:item for item in entries}
    if len(rows) != len(entries):
        raise ValueError('duplicate task ID')
    return rows


def parse_object(raw: str) -> tuple[dict | None, str | None]:
    try:
        value = json.loads(raw.strip())
        if not isinstance(value, dict):
            raise ValueError('response is not a JSON object')
        return value, None
    except (ValueError, TypeError) as error:
        return None, str(error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--reveals', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if revision != args.source_revision:
        raise ValueError('source revision mismatch')
    if os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1':
        raise RuntimeError('offline flags required')
    if not args.model.is_dir() or not (args.model/'model.safetensors').is_file():
        raise FileNotFoundError('pinned local checkpoint absent')
    if args.model.name != MODEL_REVISION:
        raise ValueError('checkpoint revision mismatch')
    public, reveals = read_rows(args.public), read_rows(args.reveals)
    if set(args.ids) - set(public) or set(args.ids) - set(reveals) or len(args.ids) != len(set(args.ids)):
        raise ValueError('invalid task IDs')
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model),local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(args.model),local_files_only=True,
                                                torch_dtype=torch.float16).to('cuda').eval()

    def generate(prompt: str) -> tuple[str, int]:
        messages = [{'role':'user','content':prompt}]
        chat = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        tokens = tokenizer(chat,return_tensors='pt').to('cuda')
        with torch.inference_mode():
            output = model.generate(**tokens,max_new_tokens=160,do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        continuation = output[0,tokens.input_ids.shape[1]:]
        return tokenizer.decode(continuation,skip_special_tokens=True),int(continuation.shape[0])

    args.output.mkdir(parents=True)
    responses = []
    trace = []
    start = time.monotonic()
    for task_id in args.ids:
        public_task, reveal = public[task_id], reveals[task_id]
        prompt1 = json.dumps(public_task,sort_keys=True)+'\n'+STAGE1_SCHEMA
        raw1, tokens1 = generate(prompt1)
        parsed1, error1 = parse_object(raw1)
        prompt2 = (json.dumps(public_task,sort_keys=True)+'\n'
                   +json.dumps(reveal,sort_keys=True)+'\n'+STAGE2_SCHEMA)
        raw2, tokens2 = generate(prompt2)
        parsed2, error2 = parse_object(raw2)
        responses.append({'id':task_id,'stage1':parsed1,'stage2':parsed2})
        trace.append({'id':task_id,'stage1_prompt_sha256':digest(prompt1.encode()),
                      'stage2_prompt_sha256':digest(prompt2.encode()),
                      'stage1_raw':raw1,'stage2_raw':raw2,
                      'stage1_parse_error':error1,'stage2_parse_error':error2,
                      'stage1_generated_tokens':tokens1,'stage2_generated_tokens':tokens2})
    for name,entries in (('responses.jsonl',responses),('trace.jsonl',trace)):
        (args.output/name).write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in entries))
    files = {name:digest((args.output/name).read_bytes()) for name in ('responses.jsonl','trace.jsonl')}
    receipt = {'source_revision':revision,'model_revision':MODEL_REVISION,
               'model_path':str(args.model),'public_sha256':digest(args.public.read_bytes()),
               'reveals_sha256':digest(args.reveals.read_bytes()),'task_ids':args.ids,
               'model_calls':2*len(args.ids),'closed_model_calls':0,'private_files_read':0,
               'max_new_tokens_per_call':160,'do_sample':False,'seconds':time.monotonic()-start,
               'file_sha256':files}
    (args.output/'complete.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(f'completed {len(args.ids)} tasks; {2*len(args.ids)} offline model calls')


if __name__ == '__main__':
    main()
