#!/usr/bin/env python3
"""
One-shot prefetch of every Qwen2.5 size used by the model-scale sweep into
HF_HOME, so ACE jobs can load from local snapshot paths and never hit the
HuggingFace Hub API (which rate-limits CURC's shared campus IP with 429s
when ~60 jobs start together).

Usage (on a CURC login node, with HF_TOKEN set):
  export HF_HOME=/projects/paco0228/cache/huggingface
  export HF_TOKEN=hf_...   # required; anonymous IP is already rate-limited
  conda activate ace
  python scripts/runners/prefetch_qwen_models.py

Or via the CPU SLURM helper:
  bash jobs/curc_prefetch_hf_models.sh
"""
from __future__ import annotations

import argparse
import os
import sys


MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-32B",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--hf-home", default=os.environ.get(
        "HF_HOME", "/projects/paco0228/cache/huggingface"))
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN is not set. Anonymous downloads from CURC's "
              "campus IP are rate-limited (429). Create a free HF token at "
              "https://huggingface.co/settings/tokens and export it.",
              file=sys.stderr)
        sys.exit(2)

    os.environ["HF_HOME"] = args.hf_home
    os.makedirs(args.hf_home, exist_ok=True)

    from huggingface_hub import snapshot_download

    # Import after HF_HOME is set so the resolver sees the same tree jobs use.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from ace_experiments import resolve_local_hf_snapshot

    print(f"HF_HOME={args.hf_home}")
    print(f"Prefetching {len(args.models)} models (authenticated)...")
    for mid in args.models:
        existing = resolve_local_hf_snapshot(mid, hf_home=args.hf_home)
        if existing != mid and os.path.isdir(existing):
            print(f"  SKIP (already cached): {mid} -> {existing}")
            continue
        print(f"  Downloading {mid} ...")
        # Default huggingface_hub layout under HF_HOME/hub/models--org--name/.
        path = snapshot_download(mid, token=token)
        resolved = resolve_local_hf_snapshot(mid, hf_home=args.hf_home)
        print(f"    done: snapshot_download -> {path}")
        print(f"    resolve_local_hf_snapshot -> {resolved}")
        if resolved == mid:
            print(f"  WARNING: resolver still cannot see a local snapshot for {mid}. "
                  f"Jobs may fall back to Hub ids and hit 429s.", file=sys.stderr)

    print("\nPrefetch complete. Resubmit ACE jobs; HuggingFacePolicy will load "
          "from local snapshots and skip Hub API calls.")


if __name__ == "__main__":
    main()
