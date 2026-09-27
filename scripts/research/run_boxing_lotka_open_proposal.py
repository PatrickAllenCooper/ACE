#!/usr/bin/env python3
"""One offline, public-only open-model proposal call for a frozen world."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from boxing_lotka_proposal import build_prompt, parse_proposal


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--public', type=Path, required=True)
    p.add_argument('--condition', choices=('descriptive', 'anonymous'), required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--model-revision', required=True)
    p.add_argument('--source-revision', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not a.model.is_dir() or not (a.model / 'model.safetensors').is_file():
        raise FileNotFoundError('local model weights missing')
    if os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1':
        raise RuntimeError('offline mode required')
    prompt = build_prompt(a.public, a.condition)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU allocation required')
    model = AutoModelForCausalLM.from_pretrained(str(a.model), local_files_only=True,
                                                torch_dtype=torch.float16).to('cuda').eval()
    chat = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                         tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors='pt').to('cuda')
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=180, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'prompt.txt').write_text(prompt)
    (a.output / 'raw_response.txt').write_text(raw)
    try:
        proposal = parse_proposal(raw)
        valid, problem = True, None
        (a.output / 'proposal.json').write_text(json.dumps(proposal, indent=2) + '\n')
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        valid, problem = False, str(exc)
    receipt = {'seed': a.seed, 'condition': a.condition,
               'source_revision': a.source_revision, 'model_revision': a.model_revision,
               'prompt_sha256': sha(prompt.encode()), 'raw_sha256': sha(raw.encode()),
               'model_calls': 1, 'closed_model_calls': 0,
               'valid_proposal': valid, 'parse_error': problem,
               'input_observations': 8, 'private_files_read': 0}
    if valid:
        receipt['proposal_sha256'] = sha((a.output / 'proposal.json').read_bytes())
    (a.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('proposal valid:', valid, 'family:', proposal['family'] if valid else None)


if __name__ == '__main__':
    main()
