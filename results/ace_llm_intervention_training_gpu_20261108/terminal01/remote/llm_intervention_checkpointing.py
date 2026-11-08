"""Explicit non-reentrant decoder checkpointing for eval-mode action scoring.

Recomputation checks the captured mode/reference state. This is a single-device
recipe, not an automatic memory rescue or a proof of numerical equivalence.
"""
from contextlib import contextmanager, nullcontext
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class DecoderCheckpointing:
    def __init__(self, model, *, expected_layers):
        layers = getattr(getattr(model, 'model', None), 'layers', None)
        if (type(expected_layers) is not int or expected_layers < 1
                or not isinstance(layers, nn.ModuleList) or len(layers) != expected_layers
                or getattr(model, 'is_gradient_checkpointing', False)
                or any(hasattr(layer, '_ace_checkpointing') for layer in layers)):
            raise ValueError('explicit unwrapped decoder layout required')
        self.enabled = False
        self.forward_calls = 0
        self.recompute_calls = 0
        for layer in layers:
            original = layer.forward
            layer.forward = self._wrap(layer, original)
            layer._ace_checkpointing = True

    @staticmethod
    def _state(layer):
        return tuple((name, child.training, getattr(child, 'reference', None))
                     for name, child in layer.named_modules())

    def _wrap(self, layer, original):
        def forward(*args, **kwargs):
            if not self.enabled or not torch.is_grad_enabled():
                return original(*args, **kwargs)
            if (kwargs.get('use_cache', False) or kwargs.get('past_key_values') is not None
                    or kwargs.get('past_key_value') is not None):
                raise ValueError('checkpointing requires stateless use_cache=False')
            state = self._state(layer)
            if any(training for _, training, _ in state):
                raise ValueError('checkpointed scoring requires eval mode')
            self.forward_calls += 1

            @contextmanager
            def recompute():
                if self._state(layer) != state:
                    raise ValueError('decoder mode/reference changed before recomputation')
                self.recompute_calls += 1
                yield

            return checkpoint(original, *args, use_reentrant=False,
                              preserve_rng_state=True, determinism_check='default',
                              context_fn=lambda: (nullcontext(), recompute()), **kwargs)
        return forward

    def set_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError('explicit checkpoint switch required')
        self.enabled = enabled

    def evidence(self):
        return {'enabled': self.enabled, 'forward_calls': self.forward_calls,
                'recompute_calls': self.recompute_calls, 'use_reentrant': False,
                'preserve_rng_state': True, 'native_checkpointing': False}
