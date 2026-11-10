"""Process entrypoint for all numerical arms of the shared prediction task."""
import argparse
import json
import sys

import numpy as np

from llm_intervention_contract import canonical, digest, integer
from llm_intervention_numerical_policy import choose
from llm_intervention_target_risk import choose_target_risk
from llm_intervention_task import unpack_envelope, numerical_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
        choices=("uniform_random", "menu_coverage", "local_information", "target_risk", "learned_ranker"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--training-receipt-sha256")
    args = parser.parse_args()
    envelope = json.load(sys.stdin)
    state, task, _, _, _ = unpack_envelope(envelope)
    if not integer(args.seed):
        raise ValueError("fixed nonnegative policy seed required")
    seed = int(np.random.SeedSequence([args.seed, state["requests"]]).generate_state(1, dtype=np.uint64)[0])
    if args.method == "target_risk":
        decision, metadata = choose_target_risk(state, **task.risk_arguments(), seed=seed, draws=args.draws)
    elif args.method == "learned_ranker":
        if not all((args.checkpoint, args.checkpoint_sha256, args.training_receipt_sha256)):
            raise ValueError("all independent numerical checkpoint pins required")
        import torch
        from llm_intervention_ranker import load_ranker, feature_tensor
        torch.set_num_threads(1)
        ids, x, metadata = numerical_features(envelope)
        model = load_ranker(args.checkpoint, sha256=args.checkpoint_sha256,
                            training_receipt_sha256=args.training_receipt_sha256, feature_width=x.shape[1])
        with torch.no_grad():
            scores = model(feature_tensor(x))
        # Action rows are sorted by ID with stop last. Ties are deterministic.
        index = int(torch.argmax(scores))
        decision = {"stop": True} if ids[index] is None else {"action_id": ids[index]}
        metadata.update({"scores": scores.tolist(), "action_ids": ids, "environment_responses": 0})
    else:
        decision, metadata = choose(state, method=args.method, seed=seed, draws=args.draws)
    metadata.update({"envelope_sha256": digest(envelope), "task_sha256": digest(task.payload()),
                     "method": args.method, "decision_seed": seed})
    print(canonical(metadata), file=sys.stderr)
    print(canonical(decision))


if __name__ == "__main__":
    main()
