import torch
import math
from typing import Callable, Optional

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas: tuple[float, float], eps: float, weight_decay: int):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1,b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                adjusted_lr = lr * (math.sqrt(1 - (b2**t)) / (1 - (b1**t)))

                # weight decay
                p.data -= lr * weight_decay * p.data 

                # update first moment
                m = state.get("m", torch.zeros_like(p.grad))
                m = b1 * m + (1 - b1) * p.grad
                state["m"] = m

                # update second moment
                v = state.get("v", torch.zeros_like(p.grad))
                v = b2 * v + (1 - b2) * torch.square(p.grad)
                state["v"] = v

                p.data -= adjusted_lr * m / (torch.sqrt(v) + eps)

                state["t"] = t + 1
        return loss
