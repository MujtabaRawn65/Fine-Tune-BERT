from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer
###  AdamW updates model parameters
## Outputs: Creates internal state variables for each model parameter.
##          Stores hyperparameters in self.defaults.
## Uses dictionaries (self.state[p]) to keep track of gradient history for each parameter
class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter], ##Model parameters that need optimization.
            lr: float = 1e-3, ## Learning rate (step size for updates) float
            betas: Tuple[float, float] = (0.9, 0.999),   ##Coefficients controlling momentum and variance (tuple of floats)
            eps: float = 1e-6,   ##Small number to prevent division by zero.  (float)
            weight_decay: float = 0.0,  ## Controls L2 regularization  (float)
            correct_bias: bool = True,   ## Whether to apply bias correction    (bool)
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


        ###   Performs One Optimization Step
##    Inputs:
    ##       closure (Callable, optional) → A function that recomputes the loss for second-order optimizers. Not used here.
##    Outputs:
##            Updates model parameters in place.
##            Returns loss (if closure is provided).
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
                grad = p.grad.data   ## Tensor

                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                #  Initialize state if not already initialized
                ## Creates state[p] dictionary
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0  # Tracks the number of updates (initially set to 0)
                    state["exp_avg"] = torch.zeros_like(p.data)  # First moment vector (m)   Tensor
                    state["exp_avg_sq"] = torch.zeros_like(p.data)  # Second moment vector (v)   Tensor

                #  Retrieve state and hyperparameters
                ## # The state values (e.g., exp_avg, exp_avg_sq, step) are read from the state dictionary during parameter updates
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]   ##(Scalars)
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

                # Compute denominator (sqrt(v_t_hat) + eps)
                denom = corrected_exp_avg_sq.sqrt().add_(eps)

                #  Apply parameter update using learning rate (alpha)
                #The denominator is used to normalize the gradient making the updates invariant to the scale of the gradients.
                denom = corrected_exp_avg_sq.sqrt().add_(eps)


                #  Apply weight decay after parameter update
                # If weight_decay is not zero, this means parameter update setup (L2 regularization) is enabled
                if weight_decay != 0:
                    # θ_t=(θ_t) - lr.λ.θ_t 


                    p.data.add_(p.data, alpha=-lr * weight_decay)
        return loss
