import torch
import math
from typing import Optional

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # targets shape: [B,T], logits shape: [B,T, vocab_size]
    logsumexp = torch.logsumexp(logits, dim=-1) # [seq,1]
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (logsumexp - target_logits).mean() # subtraction order swapped for NEGATIVE logp

def softmax(x: torch.Tensor, dim: int = -1, temp: Optional[float] = None) -> torch.Tensor:
    max_el = x.max(dim=dim,keepdim=True).values
    x_stable = x - max_el
    if temp is not None:
        x_stable /= temp
    x_exp = torch.exp(x_stable)
    x_out = x_exp / x_exp.sum(dim=dim, keepdim=True)
    return x_out

def cosine_annealing_lr(
    max_lr: float, 
    min_lr: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    it: int
) -> float:
    if it < warmup_iters:
        return (it / warmup_iters) * max_lr
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        return min_lr + 0.5 * (1 + math.cos( math.pi * (it-warmup_iters)/(cosine_cycle_iters-warmup_iters) )) * (max_lr - min_lr)
    else:
        assert it > cosine_cycle_iters
        return min_lr


def grad_clip(params, max_l2_norm:float, eps: float=1e-6):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    l2_norm = torch.tensor(0.0, device=grads[0].device)
    for g in grads:
        l2_norm += (g ** 2).sum()
    l2_norm = l2_norm.sqrt()
    down_scale = min(1, max_l2_norm/(l2_norm + eps))
    for g in grads:
        g *= down_scale

