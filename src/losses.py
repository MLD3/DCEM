import numpy as np
import torch
import torch.nn as nn

from utils import _enforce_bounds

from typing import Optional

CLAMP_MAX = np.log(torch.finfo(torch.float32).max / 3) - 1.0e-5 # for numerical stability
CLAMP_MIN = np.log(torch.finfo(torch.float32).tiny)

def causal_regularization_loss(
        loss_fn: nn.Module,
        y_obs: torch.Tensor,
        y_post: torch.Tensor,
        y_est: torch.Tensor,
        t_est: torch.Tensor,
        reg: Optional[float] = 1.
    ):
    """The M-step loss (causal regularization loss).

    Args:
        loss_fn (nn.Module): The loss function used (i.e., cross entropy)
        y_obs (torch.Tensor): A tensor of observed labels.
        y_post (torch.Tensor): The E-step estimates (`y_post` = y-posterior)
        y_est (torch.Tensor): *Logits* for the outcome predictor (P(Y|X))
        t_est (torch.Tensor): *Logits* for the propensity model (P(T|X,A))
        reg (Optional[float], optional): Regularization parameter -- should not need to be adjusted. Defaults to 1.

    Returns:
        float: Loss, to be back-propagated.
    """
    _enforce_bounds(y_post, "E-step estimate", _min=0., _max=1.)

    # compute logits directly -- for numerical stability 
    loss_1 = loss_fn(y_est, torch.ones_like(y_post, dtype=torch.long)) * y_post
    loss_0 = loss_fn(y_est, torch.zeros_like(y_post, dtype=torch.long)) * (1 - y_post)
    vanilla_ce = loss_1 + loss_0

    # Need to clamp to ensure the `exp` doesn't get out of bounds (logits must be ~87.6 at most) 
    y_obs_logits = torch.stack([ 
        torch.log(
            torch.exp(torch.clamp(y_est[:, 0] + t_est[:, 1], min=CLAMP_MIN, max=CLAMP_MAX)) \
            + torch.exp(torch.clamp(y_est[:, 1] + t_est[:, 0], min=CLAMP_MIN, max=CLAMP_MAX)) \
            + torch.exp(torch.clamp(y_est[:, 0] + t_est[:, 0], min=CLAMP_MIN, max=CLAMP_MAX))), # P(Y_obs = 0 | Y, T). For some cases where testing rate is 100%, need to clamp manually to avoid overconfident predictions (log(inf) -> nan loss)
        y_est[:, 1] + t_est[:, 1], # P(Y_obs = 1 | Y, T)
        ], dim=-1)
    causal_reg = loss_fn(y_obs_logits, y_obs) * y_post
    return (vanilla_ce + reg * causal_reg).mean()