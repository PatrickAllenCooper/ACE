"""Finite legal-command decoding, independent of model scores or outcomes.

This is a new constrained-generation interface. It does not repair a rejected
completion, select a preferred action, or turn old failures into successes.
One grammar binds one unpadded prompt. Multi-prompt batching needs a separate
binding layer. The strict executor remains responsible for charging/execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from llm_intervention_contract import Action, canonical, integer, parse_completion


def _tokens(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(values)
    if not result or any(not integer(token) for token in result):
        raise ValueError("nonempty nonnegative integer token sequence required")
    return result


@dataclass(frozen=True)
class FiniteActionGrammar:
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[tuple[int, ...], ...]
    eos_token_id: int

    def __post_init__(self):
        prompt = _tokens(self.prompt_ids)
        completions = tuple(_tokens(x) for x in self.completion_ids)
        if (not integer(self.eos_token_id) or not completions
                or len(set(completions)) != len(completions)
                or any(len(x) < 2 or x[-1] != self.eos_token_id
                       or self.eos_token_id in x[:-1] for x in completions)):
            raise ValueError("distinct nonempty commands followed by exactly one EOS required")
        object.__setattr__(self, "prompt_ids", prompt)
        object.__setattr__(self, "completion_ids", completions)

    def allowed_next(self, sequence_ids: Sequence[int]) -> tuple[int, ...]:
        sequence = _tokens(sequence_ids)
        if sequence[:len(self.prompt_ids)] != self.prompt_ids:
            raise ValueError("generation prefix differs from bound prompt")
        suffix = sequence[len(self.prompt_ids):]
        if suffix in self.completion_ids:
            raise ValueError("completion already ended with EOS")
        allowed = {x[len(suffix)] for x in self.completion_ids
                   if len(suffix) < len(x) and x[:len(suffix)] == suffix}
        if not allowed:
            raise ValueError("generated suffix is outside legal command grammar")
        return tuple(sorted(allowed))

    def validate_complete(self, sequence_ids: Sequence[int]) -> None:
        sequence = _tokens(sequence_ids)
        if (sequence[:len(self.prompt_ids)] != self.prompt_ids
                or sequence[len(self.prompt_ids):] not in self.completion_ids):
            raise ValueError("generation is not one complete bound command")


def build_action_grammar(prompt_ids: Sequence[int], menu: tuple[Action, ...],
                         remaining_units: int, encode: Callable[[str], Sequence[int]],
                         decode: Callable[[Sequence[int]], str], eos_token_id: int):
    """Include every affordable legal action and stop, with no model preference.

    `encode` must omit added special tokens. `decode` receives command tokens
    only. Exact round trips prevent a tokenizer from changing action identity.
    The menu is the already validated public menu supplied to every policy.
    """
    if (not integer(remaining_units) or type(menu) is not tuple or not menu
            or any(not isinstance(a, Action) for a in menu)
            or len({a.action_id for a in menu}) != len(menu)):
        raise ValueError("validated nonempty menu and nonnegative remaining budget required")
    commands = [canonical({"action_id": a.action_id}) for a in sorted(menu, key=lambda a: a.action_id)
                if a.cost_units <= remaining_units]
    commands.append(canonical({"stop": True}))
    completions = []
    for command in commands:
        tokens = _tokens(encode(command))
        if decode(tokens) != command:
            raise ValueError("command token round trip changes exact text")
        parse_completion(command, menu)
        completions.append(tokens + (eos_token_id,))
    return FiniteActionGrammar(tuple(prompt_ids), tuple(completions), eos_token_id)
