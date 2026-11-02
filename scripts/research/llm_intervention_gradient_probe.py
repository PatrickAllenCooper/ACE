"""Observe action-loss gradients without changing their propagation or weights."""
from __future__ import annotations

import math
import torch

from llm_intervention_action_scoring import action_positions, supervised_loss_selected
from llm_intervention_learning import supervised_loss


def capture_backward(model, batch, *, selected, loss_function=None):
    if type(selected) is not bool or not isinstance(getattr(model, "lm_head", None), torch.nn.Module):
        raise ValueError("explicit path and output projection required")
    positions, _, _ = action_positions(batch)
    captured = {}
    handles = []
    def before_projection(module, args):
        if len(args) != 1 or not isinstance(args[0], torch.Tensor) or args[0].ndim != 3:
            raise ValueError("one rank-three projection input required")
        x = args[0]
        if "projection_inputs" in captured or not x.requires_grad:
            raise ValueError("exactly one differentiable projection call required")
        captured["projection_inputs"] = x.detach().cpu().clone()
        def gradient_observer(gradient):
            if "projection_input_gradients" in captured:
                raise ValueError("exactly one projection-input gradient required")
            captured["projection_input_gradients"] = gradient.detach().cpu().clone()
            return None  # Tensor hook observes only. It never replaces a gradient.
        handles.append(x.register_hook(gradient_observer))
        return None  # Pre-hook never replaces a module argument.
    model.zero_grad(set_to_none=True)
    handles.append(model.lm_head.register_forward_pre_hook(before_projection))
    try:
        loss = ((supervised_loss_selected if selected else supervised_loss)(model, batch)
                if loss_function is None else loss_function(model, batch, selected))
        if not torch.isfinite(loss):
            raise ValueError("nonfinite loss")
        loss.backward()
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if not trainable or any(p.grad is None for p in trainable.values()):
            raise ValueError("every trainable tensor requires a recorded gradient")
        gradients = {n: p.grad.detach().cpu().clone() for n, p in trainable.items()}
        if (set(captured) != {"projection_inputs", "projection_input_gradients"}
                or any(not torch.isfinite(t).all() for t in [*gradients.values(), *captured.values()])
                or not any(bool((g != 0).any()) for g in gradients.values())):
            raise ValueError("missing, nonfinite or all-zero gradient evidence")
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise ValueError("frozen parameter received a gradient")
        return {"path": "selected" if selected else "full", "loss": float(loss.detach().cpu()),
                "positions": positions.cpu().clone(), "gradients": gradients, **captured}
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


def tensor_comparison(a, b, *, atol, rtol):
    if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
            or a.shape != b.shape or a.dtype != b.dtype or a.device.type != 'cpu' or b.device.type != 'cpu'
            or a.numel() == 0 or not torch.isfinite(a).all() or not torch.isfinite(b).all()
            or not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0):
        raise ValueError("finite aligned CPU tensors and nonnegative tolerances required")
    x, y = a.double(), b.double()
    difference = x-y
    return {"shape": list(a.shape), "dtype": str(a.dtype), "bitwise_equal": bool(torch.equal(a, b)),
            "within_tolerance": bool(torch.allclose(x, y, atol=atol, rtol=rtol)),
            "max_absolute_difference": float(difference.abs().max()),
            "difference_l2": float(torch.linalg.vector_norm(difference)),
            "left_l2": float(torch.linalg.vector_norm(x)), "right_l2": float(torch.linalg.vector_norm(y))}


def compare_captures(a, b, *, atol=.0001, rtol=.01):
    if (set(a["gradients"]) != set(b["gradients"]) or not torch.equal(a["positions"], b["positions"])
            or a["path"] not in ('full', 'selected') or b["path"] not in ('full', 'selected')):
        raise ValueError("capture identities or selected positions differ")
    def align(capture, key):
        value = capture[key]
        if capture['path'] == 'full':
            return value.index_select(1, capture['positions'])
        if value.shape[1] != len(capture['positions']):
            raise ValueError("selected capture shape differs")
        return value
    gradients = {n: tensor_comparison(a['gradients'][n], b['gradients'][n], atol=atol, rtol=rtol)
                 for n in a['gradients']}
    return {'left_path': a['path'], 'right_path': b['path'], 'left_loss': a['loss'], 'right_loss': b['loss'],
            'loss_absolute_difference': abs(a['loss']-b['loss']),
            'gradient_tensors': len(gradients),
            'failed_gradient_tensors': sum(not v['within_tolerance'] for v in gradients.values()),
            'max_gradient_absolute_difference': max(v['max_absolute_difference'] for v in gradients.values()),
            'gradients': gradients, 'atol': atol, 'rtol': rtol,
            'projection_inputs': tensor_comparison(align(a,'projection_inputs'), align(b,'projection_inputs'), atol=0, rtol=0),
            'projection_input_gradients': tensor_comparison(align(a,'projection_input_gradients'), align(b,'projection_input_gradients'), atol=atol, rtol=rtol)}
