from functools import partial, update_wrapper

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def weighted_loss_wrapped(loss_fn, out, y, weights, reduction=torch.mean):
    """
        A general weighted loss. Use with `partial()`.
    """
    return reduction(loss_fn(out, y) * weights)

def weighted_loss_noreduce(loss_fn, out, y, weights):
    return loss_fn(out, y) * weights

def lq_loss(q, k, out, y, reduction=torch.mean, aux_weights=None):
    if aux_weights is None:
        aux_weights = torch.ones_like(y)
    probs = F.softmax(out, dim=-1)[torch.arange(len(out)), y]
    k_ = torch.tensor(k, device=probs.device, dtype=torch.float)
    lk = (1 - k ** q) / q
    lq_raw = (1 - (probs ** q)) / q
    weights = (lk >= lq_raw)

    losses = (lq_raw - lk) * weights
    return reduction(losses * aux_weights)

def js_scaled_loss(out, y_, weights=torch.tensor([0.5, 0.5])):
    weights = weights.to(out.device)
    probs = F.softmax(out, dim=1)
    oh_labels = F.one_hot(y_, len(weights)).float()
    distribs = torch.stack([probs, oh_labels], dim=0) # shape: 2, N, 2
    
    log_mean = (weights.view(-1, 1, 1) * distribs).sum(dim=0).clamp(1e-7, 1.0).log() # shape: N, 2
    scale = -1 / ((1 - weights[0]) * (1 - weights[0]).log())
    kls = weights[0] * F.kl_div(log_mean, distribs[0], reduction='batchmean') + weights[1] * F.kl_div(log_mean, distribs[1], reduction='batchmean')
    jsw = kls * scale
    return jsw

def semi_loss(out, y_, mask):
    probs_u = F.softmax(out[~mask], dim=1)
    lx = -torch.mean(torch.sum(F.log_softmax(out[mask], dim=1) * y_[mask], dim=1))
    lu = torch.mean((probs_u - y_[~mask]) ** 2)
    return lx, lu

def negentropy(out):
    probs = F.softmax(out, dim=1)
    return torch.mean(torch.sum(probs.log() * probs, dim=1))


CLAMP_MAX = np.log(torch.finfo(torch.float32).max / 3) - 1.0e-5 # for numerical stability
CLAMP_MIN = np.log(torch.finfo(torch.float32).tiny)
def consistency_regularized_loss(out, y_, y_post, t_logits, reg=1., loss_fn=nn.CrossEntropyLoss(reduction='none'), con_loss_fn=nn.CrossEntropyLoss(reduction='none'), no_y_loss=False, hard_t_labels=False, aux_weights=None, mean_reduce=True):
    """
        Consistency regularized loss, or the loss derived from the M-step under
        disparate censorship:

        L_ce(y_hat, f(x); 1) + reg * L_ce(y_obs, f(x) * e(x); y_hat)
    """

    if mean_reduce:
        w_loss = partial(weighted_loss_wrapped, loss_fn)
        w_con_loss = partial(weighted_loss_wrapped, con_loss_fn)
    else:
        w_loss = partial(weighted_loss_noreduce, loss_fn)
        w_con_loss = partial(weighted_loss_noreduce, con_loss_fn)
    if aux_weights is None:
        aux_weights = torch.ones_like(y_post)

    l1 = w_loss(out, torch.ones_like(y_post, dtype=torch.long), y_post * aux_weights)
    l0 = w_loss(out, torch.zeros_like(y_post, dtype=torch.long), (1 - y_post) * aux_weights)
    y_model_loss = l0 + l1
 
    if hard_t_labels:
        # no change to y_obs_logits, and regularization is gated -- only occurs for T = 1; o/w the 2nd term is const
        aux_weights = aux_weights * t_logits.float() # t_logits should be [0, 1] if that keyword is true
        y_obs_logits = out
    else:
        # compute logits directly -- for numerical stability. Need to clamp to ensure the `exp` doesn't get out of bounds (should only happen under very extreme logits) 
        y_obs_logits = torch.stack([ 
           torch.log(
               torch.exp(torch.clamp(out[:, 0] + t_logits[:, 1], min=CLAMP_MIN, max=CLAMP_MAX)) \
                + torch.exp(torch.clamp(out[:, 1] + t_logits[:, 0], min=CLAMP_MIN, max=CLAMP_MAX)) \
                + torch.exp(torch.clamp(out[:, 0] + t_logits[:, 0], min=CLAMP_MIN, max=CLAMP_MAX))), # P(Y_obs = 0 | Y, T). For some cases where testing rate is 100%, need to clamp manually to avoid overconfident predictions (log(inf) -> nan loss)
           out[:, 1] + t_logits[:, 1], # P(Y_obs = 1 | Y, T)
           ], dim=-1)

    consistency_loss = w_con_loss(y_obs_logits, y_, y_post * aux_weights)
    return int(not no_y_loss) * y_model_loss + reg * consistency_loss, y_model_loss, consistency_loss
