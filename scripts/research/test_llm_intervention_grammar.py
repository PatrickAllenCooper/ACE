"""Pure arithmetic/token fixtures. No tokenizer/model/scientific response."""
import unittest

from llm_intervention_contract import Action
from llm_intervention_grammar import FiniteActionGrammar, build_action_grammar


class GrammarTests(unittest.TestCase):
    def setUp(self):
        self.prompt = [1000, 1001]
        self.menu = (Action("a0", "intervene", (("X", -1.),), ("Y",), 1),
                     Action("a1", "intervene", (("X", 1.),), ("Y",), 2))
        self.encode = lambda x: list(map(ord, x))
        self.decode = lambda x: "".join(map(chr, x))

    def grammar(self, budget=2):
        return build_action_grammar(self.prompt, self.menu, budget, self.encode, self.decode, 0)

    def test_every_action_and_stop_reachable_without_preference(self):
        grammar = self.grammar()
        self.assertEqual({self.decode(x[:-1]) for x in grammar.completion_ids},
                         {'{"action_id":"a0"}', '{"action_id":"a1"}', '{"stop":true}'})
        for completion in grammar.completion_ids:
            prefix = self.prompt[:]
            for token in completion:
                self.assertIn(token, grammar.allowed_next(prefix))
                prefix.append(token)
            grammar.validate_complete(prefix)

    def test_missing_closing_quote_cannot_be_generated(self):
        grammar = self.grammar()
        prefix = self.prompt + self.encode('{"action_id":"a0')
        self.assertEqual(grammar.allowed_next(prefix), (ord('"'),))
        with self.assertRaises(ValueError):
            grammar.validate_complete(prefix + self.encode('}') + [0])

    def test_exact_allowed_set_against_independent_text_oracle(self):
        grammar = self.grammar()
        texts = ('{"action_id":"a0"}', '{"action_id":"a1"}', '{"stop":true}')
        prefixes = {text[:length] for text in texts for length in range(len(text)+1)}
        for prefix in prefixes:
            expected = {0 if len(text) == len(prefix) else ord(text[len(prefix)])
                        for text in texts if text.startswith(prefix)}
            self.assertEqual(set(grammar.allowed_next(self.prompt+self.encode(prefix))), expected)

    def test_budget_prunes_actions_but_preserves_stop(self):
        self.assertEqual(len(self.grammar(1).completion_ids), 2)
        self.assertEqual(self.grammar(0).completion_ids, (tuple(self.encode('{"stop":true}')+[0]),))

    def test_prompt_binding_partial_unknown_and_after_eos_rejected(self):
        grammar = self.grammar()
        for invalid in ([999]+self.prompt, self.prompt[:1], self.prompt+self.encode('bad')):
            with self.assertRaises(ValueError):
                grammar.allowed_next(invalid)
        complete = self.prompt + list(grammar.completion_ids[0])
        with self.assertRaises(ValueError): grammar.allowed_next(complete)
        with self.assertRaises(ValueError): grammar.validate_complete(complete+[1])
        with self.assertRaises(ValueError): grammar.validate_complete(complete[:-1])

    def test_token_identity_and_duplicate_grammar_rejected(self):
        with self.assertRaises(ValueError):
            build_action_grammar(self.prompt, self.menu, 2, self.encode, lambda _: 'wrong', 0)
        for completions, eos in ((((1, 0), (1, 0)), 0), (((1, 0, 2, 0),), 0),
                                  (((True, 0),), 0), (((1, -1, 0),), 0), (((1, 0),), True)):
            with self.assertRaises(ValueError): FiniteActionGrammar((1000,), completions, eos)

    def test_inputs_copied_and_menu_order_does_not_choose_action(self):
        prompt, completions = [1000], [[1, 0], [2, 0]]
        grammar = FiniteActionGrammar(prompt, completions, 0)
        prompt[0] = 999
        completions[0][0] = 9
        self.assertEqual(grammar.allowed_next([1000]), (1, 2))
        reversed_menu = build_action_grammar(self.prompt, self.menu[::-1], 2, self.encode, self.decode, 0)
        self.assertEqual(self.grammar(), reversed_menu)


if __name__ == "__main__":
    unittest.main()
