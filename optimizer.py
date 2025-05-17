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

                #  Initialize state if not already initialized
                state = self.state[p]
                # Access hyperparameters from the `group` dictionary.

                 # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # To complete this implementation:
                # 1. Update the first and second moments of the gradients.
                # 2. Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3. Update parameters (p.data).
                # 4. Apply weight decay after the main gradient-based updates.
                # Refer to the default project handout for more details.

                ### TODO  TAYYAB
                ## The state dictionary is initialized for each parameter if it doesn't already exist
                if len(state) == 0:
                    state["step"] = 0    ## Tracks the number of updates (initially set to 0).
                    state["exp_avg"] = torch.zeros_like(p.data)  ## First moment vector (m)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)  ## Second moment vector (v)
                    # p.data is a multi-dimensional tensor that represents the weights of the model parameters

                # Retrieve state and hyperparameters
                # The state values (e.g., exp_avg, exp_avg_sq, step) are read from the state dictionary during parameter updates
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]

                # Increment step
                state["step"] += 1

                #  Update biased first moment estimate (m_t)=((B_1).(m_t-1)) + ((1-B_t).g_t)   
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                #  Update biased second raw moment estimate (v_t)=((B_2).(v_t-1)) + ((1-B_t).g_t) 
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                #  Compute bias-corrected first moment estimate (m_t_hat)
                # using the "efficient version" given in https://arxiv.org/abs/1412.6980
                bias_correction1 = 1 - beta1 ** state["step"] #adjusts for the fact that the moving average m_t starts small (biased toward zero) when training begins
                corrected_exp_avg = exp_avg / bias_correction1

                #  Compute bias-corrected second moment estimate (v_t_hat)= (v_t)/(1-(B_1)*t)
                bias_correction2 = 1 - beta2 ** state["step"]
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2


                # Compute the adaptive learning rate directly (efficient version)
                scaled_lr = lr * (1 - beta2 ** step) ** 0.5 / (1 - beta1 ** step)

               
                #The denominator is used to normalize the gradient making the updates invariant to the scale of the gradients.
                denom = corrected_exp_avg_sq.sqrt().add_(eps)

                p.data.addcdiv_(exp_avg, denom, value=-scaled_lr)  # Update parameters: θ_t = θ_t−1 − α_t · mt / (√vt + ϵ)


                #  Apply weight decay after parameter update
                # If weight_decay is not zero, this means parameter update setup (L2 regularization) is enabled
                if weight_decay != 0:
                    # θ_t=(θ_t) - lr.λ.θ_t 


                    p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss
