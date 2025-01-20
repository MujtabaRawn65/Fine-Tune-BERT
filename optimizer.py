from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        # Validate hyperparameters
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        # Store default hyperparameters
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        # Initialize loss if a closure is provided
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Access gradient of the parameter
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # Step 1: Initialize state if not already initialized
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)  # First moment vector (m)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)  # Second moment vector (v)

                # Step 2: Retrieve state and hyperparameters
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]

                # Step 3: Increment step
                state["step"] += 1

                # Step 4: Update biased first moment estimate (m_t)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Step 5: Update biased second raw moment estimate (v_t)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Step 6: Compute bias-corrected first moment estimate (m_t_hat)
                bias_correction1 = 1 - beta1 ** state["step"]
                corrected_exp_avg = exp_avg / bias_correction1

                # Step 7: Compute bias-corrected second moment estimate (v_t_hat)
                bias_correction2 = 1 - beta2 ** state["step"]
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                # Step 8: Compute denominator (sqrt(v_t_hat) + eps)
                denom = corrected_exp_avg_sq.sqrt().add_(eps)

                # Step 9: Apply parameter update using learning rate (alpha)
                p.data.addcdiv_(corrected_exp_avg, denom, value=-lr)

                # Step 10: Apply weight decay after parameter update
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss
