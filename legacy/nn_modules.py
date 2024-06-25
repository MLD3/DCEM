import copy
from collections import defaultdict
import operator as op
from functools import partial, reduce
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_ema import ExponentialMovingAverage
from tqdm.auto import tqdm

from losses import weighted_loss_wrapped, consistency_regularized_loss, semi_loss, negentropy
import metrics
from utils import linear_warmup_multiplier

# TODO: refactor the specialized techniques into a new subdirectory (one new estimator per file) + make a master module for importing everything easily

class SimpleMLP(nn.Module, BaseEstimator):
    """
        SimpleMLP is a base model class that combines torch module (for autograd) 
        and scikit-learn functionality (for scikit-learn built-in visualizations).
        As hacky as it is, all that is necessary is to inherit from a BaseEstimator
        and ensure that a field with a trailing underscore is set following `.fit()`.

        Each module has to have (at the very least) a `.fit()` method, which should 
        (optionally, but recommended) call a `validate` method to assess goodness of fit.
        The `forward`, `predict` and `predict_proba` methods are used by other methods.
    """

    def __init__(self,
                 input_size,
                 hidden_sizes=[64, 64],
                 loss_fn=nn.CrossEntropyLoss(),
                 seed=42,
                 device=f"cuda" if torch.cuda.is_available() else "cpu",
                 metric_names=["AUC", "xAUC", "ROCGap"],
                 parent=None,
                 **kwargs
                 ):
        super().__init__()
        self.seed = seed
        torch.manual_seed(self.seed)
        
        
        self.lin1 = nn.Linear(input_size, hidden_sizes[0])
        lins = [self.lin1, nn.ReLU()]
        
        for i in range(len(hidden_sizes) - 1):
            lins.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            lins.append(nn.ReLU())
        self.fc = nn.Linear(hidden_sizes[-1], 2)
        lins.append(self.fc)

        self.lins = nn.Sequential(*lins)

        self.hidden_sizes = hidden_sizes
        self.hparams = kwargs
        self.device = device
        self.constraints = None  # TODO
        self.loss_fn = loss_fn
        self.metric_names = metric_names
        if parent is None:
            self.parent = self.__class__.__name__
        else:
            self.parent = parent
        self.global_step = 0
        self.to(self.device)

        self.extra_info = defaultdict(dict)

    def forward(self, X):
        if hasattr(self, "lins"):
            out = self.lins(X)
        else: # backward compatibility w/ earlier implementations
            out = self.lin1(X)
            out = F.relu(out)
            out = self.lin2(out)
            out = F.relu(out)
            out = self.fc(out)
        return out

    def train_one_epoch(self, X_, y_, A=None, T=None, loss_weights=None, logger=None):
        out = self(X_)
        if loss_weights is None:
            loss = self.loss_fn(out, y_)
        else:
            loss = self.loss_fn(out, y_, loss_weights)
        if self.constraints is not None:
            loss += self.constraints(out, y_)
        return loss


    def log_fairness_metrics_for_split(self, logger, pred_probs, Y, A, split):

        def _cpu_if_not_cpu(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy()
            else:
                return obj

        pred_probs, Y, A = _cpu_if_not_cpu(pred_probs), _cpu_if_not_cpu(Y), _cpu_if_not_cpu(A)
        for m in self.metric_names:
            metric_dict = getattr(metrics, m)()(pred_probs, Y, A) 
            prefix = "/".join([m.lower(), self.parent, split]) 
            if logger is not None:
                logger.add_scalar(f"{prefix}/group0",  metric_dict["Group 0 value"], self.global_step) 
                logger.add_scalar(f"{prefix}/group1",  metric_dict["Group 1 value"], self.global_step)
                logger.add_scalar(f"{prefix}/diff",  metric_dict["diff"], self.global_step)
                logger.add_scalar(f"{prefix}/abs_diff",  np.abs(metric_dict["diff"]), self.global_step)
                if metric_dict["overall"] is not None:
                    logger.add_scalar(prefix, metric_dict["overall"], self.global_step) 

    def fit(self, X, y, A=None, T=None, X_val=None, y_val=None, A_val=None, T_val=None, loss_weights=None, val_loss_weights=None, no_validate=False, logger=None):

        if not no_validate:
            assert X_val is not None
            assert y_val is not None
        optimizer = getattr(optim, self.hparams["optimizer"])(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        if "initializer" in self.hparams:
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    getattr(torch.nn.init, self.hparams["initializer"])(m.weight)
            self.model.lins.apply(init_weights)
                
        if "scheduler" in self.hparams:
            scheduler_params = self.hparams["scheduler"]
            scheduler = getattr(optim.lr_scheduler, scheduler_params["name"])(
                optimizer, **scheduler_params["scheduler_kwargs"])
        else:
            scheduler = None
        pbar = tqdm(range(self.hparams["epochs"]), disable=no_validate)
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y)
        X_, y_ = X.float().to(self.device), y.long().to(self.device)

        n_epochs_no_improve = 0
        best_test_loss = float('inf')
        for i in pbar:
            self.train()
            loss = self.train_one_epoch(X_, y_, A=A, loss_weights=loss_weights, logger=logger)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc = accuracy_score(self.predict(X_, A).cpu(), y_.cpu())
            results_dict = {"L_tr": loss.item(), "A_tr": train_acc}
            if logger is not None and self.global_step % 20 == 0 and not no_validate: # doesn't play well with auxilliary models and computing fairness metrics
                logger.add_scalar(f"loss/{self.parent}/train", loss.item(), self.global_step)
                logger.add_scalar(f"acc/{self.parent}/train", train_acc, self.global_step)


                pred_probs = self.predict_proba(X_, A)

                # Evaluate on true labels
                with torch.no_grad():
                    self.log_fairness_metrics_for_split(logger, pred_probs, y_, A, "train")
            if no_validate: 
                test_loss, test_acc = None, None
            else:
                test_loss, test_acc = self.validate(X_val, y_val, A_val=A_val, loss_weights=val_loss_weights)
                results_dict |= {"L_ts": test_loss, "A_ts": test_acc}

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_loss)
                else:
                    scheduler.step()

            if logger is not None and self.global_step % 20 == 0 and not no_validate:
                logger.add_scalar(f"loss/{self.parent}/val", test_loss, self.global_step)
                logger.add_scalar(f"acc/{self.parent}/val", test_acc, self.global_step)
                pred_probs = self.predict_proba(X_val, A_val)

                # Evaluate on true labels
                with torch.no_grad():
                    self.log_fairness_metrics_for_split(logger, pred_probs, y_val, A_val, "val")
            if self.hparams.get("early_stopping", False) and test_loss is not None:
                if test_loss + 1e-8 < best_test_loss:
                    n_epochs_no_improve = 0
                    best_test_loss = test_loss
                    self.best_state_dict = self.state_dict()
                else:
                    n_epochs_no_improve += 1
                    if n_epochs_no_improve % 20 == 0:
                        print()
                        print(n_epochs_no_improve, f"epochs with no improvement (max", str(self.hparams["early_stop_patience"]) + ")")
                    if n_epochs_no_improve == self.hparams["early_stop_patience"]:
                        print()
                        print(self.hparams["early_stop_patience"], f"epochs with no improvement -- patience reached")
                        self.load_state_dict(self.best_state_dict)
                        break

            if pbar.disable:
                print(
                    f"\rEpoch {i+1}/{len(pbar)}: L_tr={loss.item():.3f}, A_tr={train_acc:.3f}", end="")
            else:
                pbar.set_postfix(results_dict)
            self.global_step += 1
        self.fitted_flag_ = True
        return loss.item(), test_loss

    def validate(self, X_val, y_val, A_val=None, T_val=None, loss_weights=None):
        self.eval()
        test_loss = 0
        if not isinstance(X_val, torch.Tensor):
            X_val = torch.from_numpy(X_val)
        if not isinstance(y_val, torch.Tensor):
            y_val = torch.from_numpy(y_val)
        X_val, y_val = X_val.float().to(self.device), y_val.long().to(self.device)

        with torch.no_grad():
            out = self(X_val)
            if loss_weights is None:
                test_loss = self.loss_fn(out, y_val).item()
            else:
                if loss_weights.size(0) != out.size(0):
                    # warnings.warn("Weights don't match prediction size. If you are validating on non-training labels, this is OK. Defaulting to all-ones (un-weighted).")
                    loss_weights = torch.ones(out.size(0))
                test_loss = self.loss_fn(out, y_val, loss_weights.to(self.device)).item()
        test_acc = accuracy_score(self.predict(X_val, A_val).cpu(), y_val.cpu())
        return test_loss, test_acc

    def predict(self, X, A=None):
        with torch.no_grad():
            if not torch.is_tensor(X):
                X = torch.from_numpy(X).to(self.get_device()).float()
            _, preds = torch.max(self(X).data, 1)
            return preds

    def predict_proba(self, X, A=None):
        with torch.no_grad():
            if not torch.is_tensor(X):
                # hacky, but otherwise doesn't play nice with the plotting code
                X = torch.from_numpy(X).to(self.get_device()).float()
            return F.softmax(self(X), dim=-1)

    def get_device(self):
        return next(self.parameters()).device

class ITECorrectedMLP(SimpleMLP):
    def __init__(self, input_size, **kwargs):
        super().__init__(input_size, **kwargs)
        self.prob_estimator = SimpleMLP(input_size, parent=self.parent, **self.hparams)
        self.ite_estimator = DragonNetEstimator(input_size, parent=self.parent + "_ite", **self.hparams)

    def fit(self, *args, **kwargs):
        self.ite_estimator.fit(*args, no_validate=True, **kwargs)
        self.prob_estimator.fit(*args, **kwargs)
        self.fitted_flag_ = True

    def predict(self, X, A=None):
        assert A is not None
        with torch.no_grad():
            _, preds = torch.max(self.predict_proba(X, A), 1)
            return preds

    def predict_proba(self, X, A=None):
        assert A is not None
        with torch.no_grad():
            if not torch.is_tensor(X):
                X = torch.from_numpy(X).to(self.get_device()).float()
            if not torch.is_tensor(A):
                A = torch.from_numpy(A).to(self.get_device()).bool().view(-1, 1)
            probs = F.softmax(self.prob_estimator(X), dim=-1)
            ite = self.ite_estimator.get_ite(X)
            diff = torch.stack([-ite, ite], dim=-1)
            zeros = torch.zeros_like(diff)
            return torch.clamp(probs - torch.where(A == 0, zeros, diff), min=1e-8, max=1 - 1e-8)

class PeerLossMLP(SimpleMLP):
    def __init__(self, input_size, alpha=1., **kwargs):
        super().__init__(input_size, **kwargs)
        self.alpha = alpha

    def get_peers(self, X_, y_):
        with torch.no_grad():
            n = X_.size(0)
            out = self(X_)
            peer_out = self(X_[torch.randperm(n)])
            peer_y = y_[torch.randperm(n)]
        return peer_out, peer_y

    def train_one_epoch(self, X_, y_, A, T=None, logger=None, loss_weights=None, peer_reduction=torch.mean):

        peer_out, peer_y = self.get_peers(X_, y_)
        out = self(X_)

        if loss_weights is None:
            loss = self.loss_fn(out, y_) 
            peer_loss = self.loss_fn(peer_out, peer_y)
        else:
            # get the thing inside the partial -- only apply constraint to the CE-proper, not additional constraints (maybe refactor later)
            loss = self.loss_fn(out, y_, loss_weights)
            peer_loss = peer_reduction(self.loss_fn.args[0](peer_out, peer_y))
        total_loss = loss - self.alpha * peer_loss
        if logger is not None and self.global_step % 20 == 0:
            logger.add_scalar(f"ce_loss/{self.parent}/train", loss, self.global_step)
            logger.add_scalar(f"peer_loss/{self.parent}/train", loss, self.global_step)
        # total loss is logged in the outer .fit() call
        return total_loss

class GroupPeerLossMLP(PeerLossMLP):
    """
        Jialu Wang, Yang Liu, and Caleb Levy. 2021. Fair Classification with Group-Dependent Label Noise, FAccT '21.
    """
    def __init__(self, input_size, alpha, noise0, noise1, loss_fn=partial(weighted_loss_wrapped, nn.CrossEntropyLoss(reduction='none')), **kwargs):
        super().__init__(input_size, alpha=alpha, loss_fn=loss_fn, **kwargs)
        self.noise0 = noise0
        self.noise1 = noise1

    def train_one_epoch(self, X_, y_, A, loss_weights=None, peer_reduction=torch.mean, logger=None):
        peer_out, peer_y = self.get_peers(X_, y_)
        out = self(X_)
        loss_weights = torch.from_numpy(1 / np.where(A == 0, self.noise0, self.noise1)).to(self.device)
        loss = self.loss_fn(out, y_, loss_weights)
        peer_loss = self.loss_fn(peer_out, peer_y, loss_weights)
        total_loss = loss - self.alpha * peer_loss
        if logger is not None and self.global_step % 20 == 0:
            logger.add_scalar(f"ce_loss/{self.parent}/train", loss, self.global_step)
            logger.add_scalar(f"peer_loss/{self.parent}/train", total_loss, self.global_step)
        return total_loss 

    def validate(self, X_val, y_val, A_val=None, T_val=None, loss_weights=None):
        self.eval()
        test_loss = 0
        X_val, y_val = torch.from_numpy(X_val).float().to(
            self.device), torch.from_numpy(y_val).long().to(self.device)
        loss_weights = torch.from_numpy(1 / np.where(A_val == 0, self.noise0, self.noise1)).to(self.device)

        with torch.no_grad():
            X_, y_ = X_val.to(self.device), y_val.to(self.device)
            out = self(X_)
            test_loss = self.loss_fn(self(X_), y_, loss_weights).item()
        test_acc = accuracy_score(self.predict(X_).cpu(), y_.cpu())
        return test_loss, test_acc

class DragonNetEstimator(SimpleMLP):
    """
        Adapted with reference to https://github.com/farazmah/dragonnet-pytorch and https://github.com/claudiashi57/dragonnet.
    """
    def __init__(self, input_size, reg_loss_fn=nn.MSELoss(), **kwargs):
        super().__init__(input_size, **kwargs)
        self.input_size = input_size
        self.reg_loss_fn = reg_loss_fn

        self.y0_model = SimpleMLP(self.hidden_sizes[-1], parent=self.parent + "_dragonnet_y0", **self.hparams)
        self.y1_model = SimpleMLP(self.hidden_sizes[-1], parent=self.parent + "_dragonnet_y1", **self.hparams)
        self.epsilon = nn.Linear(in_features=1, out_features=1, bias=False, device=self.device)
        torch.nn.init.xavier_normal_(self.epsilon.weight)

    def forward(self, x):
        all_but_final = self.lins[:-1]
        z = all_but_final(x) 
        a_out = self.lins[-1](z) 

        y0_out = self.y0_model(z)
        y1_out = self.y1_model(z)
        eps = self.epsilon(torch.ones_like(a_out)[:, 0:1])

        return a_out, y0_out, y1_out, eps

    def get_ite(self, x):
        t_a1 = self.predict_proba(x, a=1)[:, 1] 
        t_a0 = self.predict_proba(x, a=0)[:, 1]
        return t_a1 - t_a0

    def predict(self, x, a=0): 
        _, preds = torch.max(self.predict_proba(x, a=a), 1)
        return preds

    def predict_proba(self, x, a=0): 
        a_out, out0, out1, eps = self(x)
        prob0 = F.softmax(out0, dim=-1)
        prob1 = F.softmax(out1, dim=-1)
        a_probs = F.softmax(a_out, dim=-1)[:, 1] + 0.01 / 1.02

        if self.hparams.get("tarreg", False):
            adj0 =  eps.squeeze(-1) / (1 - a_probs)
            adj1 = eps.squeeze(-1) / a_probs
            prob0 += torch.stack([adj0, -adj0], dim=1)    
            prob1 += torch.stack([-adj1, adj1], dim=1)
        if isinstance(a, np.ndarray) or isinstance(a, torch.Tensor):
            if not torch.is_tensor(a):
                a = torch.from_numpy(a).to(self.get_device()).bool()
            return torch.where((a == 0).view(-1, 1), prob0, prob1)
        elif a == 0:
            return prob0
        elif a == 1:
            return prob1
        else:
            raise ValueError(f"Keyword 'a' must be an integer (0, 1) or an array-like (np.ndarray, torch.Tensor). Found a={a}")

    def train_one_epoch(self, X_, y_, A=None, loss_weights=None, logger=None):
        assert A is not None # must pass in from SimpleMLP.fit()
        if not isinstance(A, torch.Tensor):
            A = torch.from_numpy(A).long().to(self.device)
        a_out, y0_out, y1_out, eps = self(X_) 
  
        loss_a = self.loss_fn(a_out, A)
        loss_y0 = self.loss_fn(y0_out[A == 0], y_[A == 0])
        loss_y1 = self.loss_fn(y1_out[A == 1], y_[A == 1]) 
        loss = (loss_y0 + loss_y1) + self.hparams['alpha'] * loss_a

        if self.hparams.get("tarreg", False):
            a_probs = F.softmax(a_out, dim=-1) + 0.01 / 1.02
            y_pred = A * F.softmax(y0_out, dim=-1)[:, 1] + (1-A) * F.softmax(y1_out, dim=-1)[:, 1]
            dy = (A / a_probs[:, 1]) - ((1 - A) / a_probs[:, 0])
            y_sus = y_pred + eps.squeeze(-1) * dy
            targeted_regularization = self.reg_loss_fn(y_sus, y_pred)
            loss += self.hparams['beta'] * targeted_regularization

            if logger is not None and self.global_step % 20 == 0:
                logger.add_scalar(f"tarreg/{self.parent}/train", targeted_regularization, self.global_step)

        if logger is not None and self.global_step % 20 == 0:
            logger.add_scalar(f"loss_prop/{self.parent}/train", loss_a, self.global_step)
            logger.add_scalar(f"loss_y0/{self.parent}/train", loss_y0, self.global_step)
            logger.add_scalar(f"loss_y1/{self.parent}/train", loss_y1, self.global_step)
            logger.add_scalar(f"loss/{self.parent}/train", loss, self.global_step)
        return loss 

class PUEstimator(SimpleMLP):
    """
        Method used by J. Bekker et. al., ECMLKPDD 2019. Strong baseline for
        our setting (i.e., accounts for instance-dependent noise), but PU
        assumptions are stronger than what we need for disparate censorship.

        https://arxiv.org/abs/1809.03207
    """

    def __init__(self, input_size, loss_fn=nn.CrossEntropyLoss(reduction='none'), **kwargs):
        super().__init__(input_size, **kwargs)
        self.input_size = input_size
        self.propensity_hparams = self.hparams | {"lr": self.hparams["propensity_lr"]}

        self.model = SimpleMLP(self.input_size, loss_fn=loss_fn, device=self.device, **self.hparams)  # TODO: pass in group model if applicable
        self.propensity_model = SimpleMLP(2 * self.input_size + 1, device=self.device, **self.propensity_hparams)
        del self.loss_fn
        # manually override super class loss_fn setting
        self.loss_fn = partial(weighted_loss_wrapped, loss_fn)

    def forward(self, X):
        return self.model.forward(X)

    def fit(self, X, y, A, X_val, y_val, A_val, true_y, true_Yval, T=None, T_val=None, logger=None):

        X_, A_, y_, T_, true_y_ = torch.FloatTensor(X), torch.FloatTensor(
            A).unsqueeze(-1), torch.LongTensor(y), torch.FloatTensor(T), torch.LongTensor(true_y)
        Xv_, Av_, yv_, true_yv_, Tv_ = torch.FloatTensor(X_val), torch.FloatTensor(
            A_val).unsqueeze(-1), torch.LongTensor(y_val), torch.LongTensor(true_Yval), torch.FloatTensor(T_val)
        X_, A_, y_,  yv_ = X_.to(self.device), A_.to(self.device), y_.to(self.device), yv_.to(self.device)
        Xv_, Av_ = Xv_.to(self.device), Av_.to(self.device)
        XA_ = torch.cat([X_, X_ * A_, A_], dim=1).to(self.device)
        XAv_ = torch.cat([Xv_, Xv_ * Av_, Av_], dim=1).to(self.device)
        aug_data = torch.cat([X_, X_], dim=0)
        em_its = self.hparams["em_its"]
        em_its = self.hparams["em_its"]
        n_epochs_no_improve = 0
        best_l_prop = float('inf')
        best_ll = float('inf')
        em_patience = self.hparams.get("em_patience", None)
        for i in range(em_its):
            # E-step: assign initial estimates for latent Y

            y_post = None
            with torch.no_grad():
                est_y_prior = self.model.predict_proba(X_)[:, 1]
                est_y_prior = torch.clamp(est_y_prior, min=self.hparams["y_prior_eps"], max=1-self.hparams["y_prior_eps"])
                est_yobs_prior = self.propensity_model.predict_proba(XA_)[:, 1]
                y_post = y_ + (1 - y_) * (est_y_prior * (1 - est_yobs_prior)
                                          ) / (1 - est_y_prior * est_yobs_prior)

                est_y_val_prior = self.model.predict_proba(Xv_)[:, 1]
                est_y_val_prior = torch.clamp(est_y_val_prior, min=self.hparams["y_prior_eps"], max=1-self.hparams["y_prior_eps"])
                est_yobs_val_prior = self.propensity_model.predict_proba(XAv_)[:, 1]
                y_post_val = yv_ + (1 - yv_) * (est_y_val_prior * (1 - est_yobs_val_prior)) / (1 - est_y_val_prior * est_yobs_val_prior)


            # M-step: maximize E[Y | X, Y_obs, A] -- i.e., use y_post to train propensity and classif. models
            # i.e. reset models + retrain on new labels

            # (X, A) -> Y_obs -- here, we use Y_obs instead of T to treat Y_obs as a {is_labeled} indicator. Room for improvement?
            prop_state_dict = self.propensity_model.state_dict()
            self.propensity_model = SimpleMLP(2 * self.input_size + 1, device=self.device, loss_fn=self.loss_fn, **self.propensity_hparams)
            # X -> Y

            model_state_dict = self.model.state_dict()
            self.model = SimpleMLP(self.input_size, device=self.device, loss_fn=self.loss_fn, **self.hparams)
            if self.hparams.get("m_step_warm_start", False):
                self.propensity_model.load_state_dict(prop_state_dict)
                self.model.load_state_dict(model_state_dict)

            prop_train, prop_loss = self.propensity_model.fit(XA_, y_, X_val=XAv_, y_val=yv_, loss_weights=y_post, val_loss_weights=y_post_val)
            if logger is not None:
                logger.add_scalar(f"propensity_loss/{self.parent}/train", prop_train, self.global_step)
                logger.add_scalar(f"propensity_loss/{self.parent}/val", prop_loss, self.global_step)
                prop_acc_tr = accuracy_score(self.propensity_model.predict(XA_).cpu(), y_.cpu())
                prop_acc_ts = accuracy_score(self.propensity_model.predict(XAv_).cpu(), yv_.cpu())
                logger.add_scalar(f"propensity_acc/{self.parent}/train", prop_acc_tr, self.global_step)
                logger.add_scalar(f"propensity_acc/{self.parent}/val", prop_acc_ts, self.global_step)

            aug_class_labels = torch.cat([torch.ones_like(
                y_post, dtype=torch.long), torch.zeros_like(y_post, dtype=torch.long)])
            # the "doubling up" of X and y -- equivalent to instance-dependent label-smoothing
            aug_weights = torch.cat([y_post, 1 - y_post], dim=0)
            aug_a = torch.cat([A_.squeeze(-1).long()] * 2, dim=0)
            aug_t = torch.cat([T_] * 2, dim=0)
            # aug_av = torch.cat([Av_.squeeze(-1).long()] * 2, dim=0)

            aug_val_weights = torch.cat([y_post_val, 1 - y_post_val], dim=0)

            smooth_loss, model_loss = self.model.fit(aug_data, aug_class_labels, X_val=Xv_, y_val=true_yv_, A=aug_a, A_val=Av_.squeeze(
                -1).long(), T=aug_t, T_val=Tv_, loss_weights=aug_weights, val_loss_weights=aug_val_weights)  # update weights 
            
            if logger is not None:
                logger.add_scalar(f"loss/{self.parent}/train", smooth_loss, self.global_step)
                logger.add_scalar(f"loss/{self.parent}/val", model_loss, self.global_step)

                with torch.no_grad():
                    train_acc = accuracy_score(self.predict(X_).cpu(), y_.cpu())
                    test_acc = accuracy_score(self.predict(X_val).cpu(), yv_.cpu())
                    true_acc = accuracy_score(self.predict(X_val).cpu(), true_yv_.cpu())
                
                logger.add_scalar(f"acc/{self.parent}/train", train_acc, self.global_step)
                logger.add_scalar(f"acc/{self.parent}/val", test_acc, self.global_step)
                logger.add_scalar(f"y_acc/{self.parent}/val", true_acc, self.global_step)

                pred_probs_train, pred_probs_val = self.predict_proba(X_), self.predict_proba(X_val)
                self.log_fairness_metrics_for_split(logger, pred_probs_train, true_y_, A, "train")
                self.log_fairness_metrics_for_split(logger, pred_probs_val, true_yv_, A_val, "val")

            # print losses for propensity model and y-model
            if min([prop_train, prop_loss, smooth_loss, model_loss]) < 1e-4:
                print(
                    f"EM it. ({i+1}/{em_its}) - l_tr_p: {prop_train:.2E} - l_ts_p: {prop_loss:.2E} - l_tr: {smooth_loss:.2E} - l_ts: {model_loss:.2E}")
            else:
                print(
                    f"EM it. ({i+1}/{em_its}) - l_tr_p: {prop_train:.3f} - l_ts_p: {prop_loss:.3f} - l_tr: {smooth_loss:.3f} - l_ts: {model_loss:.3f}")
            if em_patience is not None:
                if (model_loss >= best_ll) and (prop_loss >= best_l_prop):
                    n_epochs_no_improve += 1
                    print(
                        n_epochs_no_improve, "epochs with no improvement (max", str(em_patience) + ")")
                    if n_epochs_no_improve == em_patience:
                        break

                if model_loss < best_ll:
                    best_ll = model_loss
                    n_epochs_no_improve = 0
                if prop_loss < best_l_prop:
                    best_l_prop = prop_loss
                    n_epochs_no_improve = 0
            self.global_step += 1

        self.fitted_flag_ = True


class DisparateCensorshipEstimator(SimpleMLP):
    def __init__(self, input_size, **hparams):
        super().__init__(input_size, **hparams)
        self.input_size = input_size
        self.propensity_hparams = self.hparams.get("propensity_hparams", {})
        if "propensity_lr" in self.hparams: # for backward compat only
            self.propensity_hparams.update({
                "lr": self.hparams["propensity_lr"],
                "optimizer": self.hparams["optimizer"],
                "epochs": self.hparams["epochs"],
                "weight_decay": self.hparams["weight_decay"]
            })
        self.model = SimpleMLP(self.input_size, device=self.device, hidden_sizes=self.hidden_sizes, seed=self.seed, **self.hparams)
        self.uni_propensity_model = SimpleMLP(2 * self.input_size + 1, device=self.device, seed=self.seed, parent=self.parent + "_PropensityOnly", **self.propensity_hparams) # TODO: how to adjust hidden size from default? 

        if self.hparams.get("tmle", False):
            self.eps = nn.Linear(1, 1, bias=False, device=self.device)
            torch.nn.init.zeros_(self.eps.weight)


    def predict_proba(self, X, A=None, no_epsilon=False):
        with torch.no_grad():
            if not torch.is_tensor(X):
                # hacky, but otherwise doesn't play nice with the plotting code
                X = torch.from_numpy(X).to(self.get_device()).float()
            probs = F.softmax(self.model(X), dim=-1)
            if self.hparams.get("tmle", False) and not no_epsilon:
                if not torch.is_tensor(A):
                    A = torch.from_numpy(A).to(self.get_device()).float().unsqueeze(1)
                XA_ = torch.cat([X, X * A, A], dim=1)
                u_prop = F.softmax(self.uni_propensity_model(XA_) / self.hparams.get("softmax_temperature", 1.), dim=1)[:, 1].unsqueeze(1)
                new_prob = torch.special.expit(torch.special.logit(probs[:, 1]) + self.eps(1 / u_prop).squeeze(1)) # TODO: support TMLE -- add logit
                return torch.stack([1 - new_prob, new_prob], dim=1)
            else:
                return probs 

    def forward(self, X):
        return self.model.forward(X)

    def full_forward(self, X_, y_, y_post, t_score, A=None, A_val=None, T=None, reg=1., no_y_loss=False):

        hard_t = self.hparams.get("hard_t", False)

        out = self(X_)
        aux_weights = T if self.hparams.get("gate_consistency_regularization", False) else None
        total_loss, y_loss, consistency_loss = consistency_regularized_loss(out, y_, y_post, T if hard_t else t_score, reg=reg, no_y_loss=no_y_loss, aux_weights=aux_weights, hard_t_labels=hard_t)
        if self.constraints is not None:
            total_loss += self.constraints(out, y_)
        return total_loss, y_loss, consistency_loss

    def estep(self, X_, y_, T_, A=None, tmle=False, X_val=None, y_val=None, A_val=None, u_prop=None, u_prop_val=None, tmle_iterations=500, tmle_patience=5, causal_regularization=False):
        est_y_prior = self.predict_proba(X_, no_epsilon=True)[:, 1]
        y_post = T_ * y_ + (1 - T_) * est_y_prior
        return y_post

    def mstep(self, X_, y_, y_post, t_score, X_val, y_val, T, T_val, t_score_val, true_y, XA_, XAv_, reg=1., logger=None, initialize_tested_only=False):

        model_state_dict = self.model.state_dict()
        self.model = SimpleMLP(self.input_size, hidden_sizes=self.hidden_sizes, device=self.device, seed=self.seed, **self.hparams)
        if self.hparams.get("m_step_warm_start", False) and self.global_step != 0:
            self.model.load_state_dict(model_state_dict)
        
        if "initializer" in self.hparams:
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    getattr(torch.nn.init, self.hparams["initializer"])(m.weight)
            self.model.lins.apply(init_weights) # reinitialize the inner model

        if initialize_tested_only:
            self.model.load_state_dict(self.tested_model.state_dict())

        A = XA_[:, -1].squeeze(-1).long()
        A_val = XAv_[:, -1].squeeze(-1).long()

        def report_mstep_metrics(pbar, reg=1., no_y_loss=False, _logger=None):
            train_acc = accuracy_score(self.predict(X_).cpu(), y_.cpu())
            true_acc = accuracy_score(self.predict(X_val).cpu(), true_y.cpu())
            test_loss, test_y_loss, test_consistency_loss, test_acc = self.validate(
                X_val, y_val, T_val, t_score_val, A_val, reg=reg, no_y_loss=no_y_loss)
            pbar.set_postfix({"L_tr": total_loss.item(), "A_tr": train_acc,
                             "L_ts": test_loss.item(), "A_ts": test_acc, "A(y)_ts": true_acc})

            if _logger is not None:
                _logger.add_scalar(f"acc/{self.parent}/train", train_acc, self.global_step)
                _logger.add_scalar(f"acc/{self.parent}/val", test_acc, self.global_step)
                _logger.add_scalar(f"y_acc/{self.parent}/val", true_acc, self.global_step) # Do this for other models?
 
            return test_loss, test_y_loss, test_consistency_loss

        optimizer = getattr(optim, self.hparams["optimizer"])(
            self.model.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        if "scheduler" in self.hparams:
            scheduler_params = self.hparams["scheduler"]
            scheduler = getattr(optim.lr_scheduler, scheduler_params["name"])(
                optimizer, **scheduler_params["scheduler_kwargs"])
        else:
            scheduler = None
        pbar = tqdm(range(self.hparams["epochs"]))
        freeze_propensity = self.hparams.get("freeze_propensity_model", True)

        best_test_loss = float('inf')
        n_epochs_no_improve = 0

        for i in pbar:
            self.model.train()
            optimizer.zero_grad()
            if not freeze_propensity:
                self.uni_propensity_model.train()
                t_score = self.uni_propensity_model(XA_)
                with torch.no_grad():
                    self.uni_propensity_model.eval()
                    t_score_val = self.uni_propensity_model(XAv_)
            total_loss, y_loss, consistency_loss = self.full_forward(X_, y_, y_post, t_score, A=A, A_val=A_val, T=T, reg=reg)
          
            total_loss.backward() 

                

            optimizer.step()
            test_loss, test_y_loss, test_consistency_loss = report_mstep_metrics(pbar, reg=reg, _logger=None if i != len(pbar) - 1 else logger)
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_loss)
                else:
                    scheduler.step()

            if self.hparams.get("early_stopping", False):
                if test_loss < best_test_loss:
                    n_epochs_no_improve = 0
                    best_test_loss = test_loss
                    self.best_state_dict = self.state_dict()
                else:
                    n_epochs_no_improve += 1
                    if n_epochs_no_improve % 200 == 0:
                        print()
                        print(n_epochs_no_improve, f"epochs with no improvement (max", str(self.hparams["early_stop_patience"]) + ")")
                    if n_epochs_no_improve == self.hparams["early_stop_patience"]:
                        print()
                        print(self.hparams["early_stop_patience"], f"epochs with no improvement -- patience reached")
                        self.load_state_dict(self.best_state_dict)
                        break
        
        self.fitted_flag_ = True
        return total_loss, y_loss, consistency_loss, test_loss, test_y_loss, test_consistency_loss

    def validate(self, X_val, y_val, T_val, t_score_val, A_val, reg=1., no_y_loss=False):
        self.eval()

        with torch.no_grad():
            X_, y_, T_ = X_val.to(self.device), y_val.to(
                self.device), T_val.to(self.device)
            out = self(X_)
            y_post_val = self.estep(X_, y_, T_)

            aux_weights = T_val if self.hparams.get("gate_consistency_regularization", False) else None
            total_loss, y_loss, consistency_loss = consistency_regularized_loss(out, y_, y_post_val, t_score_val, reg=reg, no_y_loss=no_y_loss, aux_weights=aux_weights)

        test_acc = accuracy_score(self.predict(X_).cpu(), y_.cpu())
        return total_loss, y_loss, consistency_loss, test_acc

    def fit(self, X, y, A, T, X_val, y_val, A_val, T_val, true_Y, true_Yval, T_score=None, T_score_val=None, reg=1., logger=None):
        X_, A_, T_, y_, true_y_ = torch.FloatTensor(X), torch.FloatTensor(A).unsqueeze(-1), torch.LongTensor(T), torch.LongTensor(y), torch.LongTensor(true_Y)
        Xv_, Av_, Tv_, yv_, true_yv_ = torch.FloatTensor(X_val), torch.FloatTensor(A_val).unsqueeze(-1), torch.LongTensor(T_val), torch.LongTensor(y_val), torch.LongTensor(true_Yval)
        X_, A_, T_, y_ = X_.to(self.device), A_.to(self.device), T_.to(self.device), y_.to(self.device)
        Xv_, Av_, Tv_, yv_, true_yv_ = Xv_.to(self.device), Av_.to(self.device), Tv_.to(self.device), yv_.to(self.device), true_yv_.to(self.device)
        XA_ = torch.cat([X_, X_ * A_, A_], dim=1)
        XAv_ = torch.cat([Xv_, Xv_ * Av_, Av_], dim=1)

        # first, pretrain a universal testing probability prediction model
        if self.hparams.get("use_true_propensities", False):
            assert T_score is not None
            assert T_score_val is not None
            u_prop = torch.FloatTensor(T_score).to(self.device)
            u_prop_val = torch.FloatTensor(T_score_val).to(self.device)
        else:
            _ = self.uni_propensity_model.fit(XA_, T_, X_val=XAv_, y_val=Tv_)
            with torch.no_grad():
                u_prop = self.uni_propensity_model(XA_) / self.hparams.get("softmax_temperature", 1.)
                u_prop_val = self.uni_propensity_model(XAv_) / self.hparams.get("softmax_temperature", 1.)

            if self.hparams.get("save_extra_info", False):
                # save u_ptop
                self.extra_info["prop_estimates"] = u_prop

        initialize_tested_only = self.hparams.get("initialize_tested_only", False)
        if initialize_tested_only: 
            self.tested_model = SimpleMLP(self.input_size, device=self.device, hidden_sizes=self.hidden_sizes, seed=self.seed, **self.hparams)
            _ = self.tested_model.fit(X_[T_ == 1], y_[T_ == 1], X_val=Xv_, y_val=yv_)

        em_its = self.hparams["em_its"]
        n_epochs_no_improve = 0
        best_l_ts = float('inf')
        em_patience = self.hparams.get("em_patience", None)
        for i in range(em_its):
            # E-step: assign initial estimates for latent Y
            with torch.no_grad():
                if i == 0 and "fixed_initial_estep" in self.hparams:
                    y_post = torch.ones_like(y_) * self.hparams["fixed_initial_estep"]
                else:
                    y_post = self.estep(X_, y_, T_, 
                            tmle=self.hparams.get("tmle", False),
                            X_val=Xv_, y_val=yv_,
                            A=A_, A_val=Av_,
                            u_prop=u_prop, u_prop_val=u_prop_val,
                            causal_regularization=self.hparams.get("causal_regularization", False)
                            )

            # M-step: maximize E[Y | X, Y_obs, A]
            losses = self.mstep(X_, y_, y_post, u_prop, Xv_, yv_, T_, Tv_, u_prop_val, true_yv_, XA_, XAv_, reg=reg, logger=logger, initialize_tested_only=initialize_tested_only and (i == 0))
            l_tr, l_tr_y, l_tr_yy, l_ts, l_ts_y, l_ts_yy = losses

            if logger is not None:
                logger.add_scalar(f"loss/{self.parent}/train", l_tr, self.global_step)
                logger.add_scalar(f"y_loss/{self.parent}/train", l_tr_y, self.global_step)
                logger.add_scalar(f"con_loss/{self.parent}/train", l_tr_yy, self.global_step)

                logger.add_scalar(f"loss/{self.parent}/val", l_ts, self.global_step)
                logger.add_scalar(f"y_loss/{self.parent}/val", l_ts_y, self.global_step)
                logger.add_scalar(f"con_loss/{self.parent}/val", l_ts_yy, self.global_step)

                with torch.no_grad():
                    pred_probs_train, pred_probs_val = self.predict_proba(X_, A=A_), self.predict_proba(X_val, A=Av_)
                    self.log_fairness_metrics_for_split(logger, pred_probs_train, true_y_, A, "train")
                    self.log_fairness_metrics_for_split(logger, pred_probs_val, true_yv_, A_val, "val")

            print(
                f"EM it. ({i+1}/{em_its}) - l_tr {l_tr:.3f} ({l_tr_y:.3f} + {l_tr_yy:.3f}) - l_ts {l_ts:.3f} ({l_ts_y:.3f} + {l_ts_yy:.3f})")
            if em_patience is not None:
                if l_ts < best_l_ts:
                    best_l_ts = l_ts
                    n_epochs_no_improve = 0
                else:
                    n_epochs_no_improve += 1
                    print(
                        n_epochs_no_improve, "epochs with no improvement (max", str(em_patience) + ")")
                    if n_epochs_no_improve == em_patience:
                        break
            self.global_step += 1
        self.fitted_flag_ = True



class SELFEstimator(SimpleMLP):
    def __init__(self,
                 input_size, 
                 seed=0,
                 consistency_loss=nn.MSELoss(),
                 **hparams
                 ):
        super().__init__(input_size, **hparams)
        self.m_best = None
        self.filter_mask = None
        self.prediction_cache = None
        self.seed = seed
        self.consistency_loss = consistency_loss
        self.hparams = hparams # redeclare this to prevent SimpleMLP from overwriting

    def train_mean_teacher_ensemble(self, X, y, X_val, y_val, loss_weights=None, logger=None):
        m_i_s = SimpleMLP(X.shape[-1],
                #device=self.device,
                **self.hparams)
        m_i_t = copy.deepcopy(m_i_s)
        m_i_t.load_state_dict(m_i_s.state_dict())
        ema = ExponentialMovingAverage(
                m_i_s.parameters(),
                decay=self.hparams["mean_teacher_alpha"],
                use_num_updates=False
            )
        # we're supposed to inject different noise to the inputs to m_i_s, m_i_t, but how?
        pbar = tqdm(range(self.hparams["epochs"]))
        optimizer = getattr(optim, self.hparams["optimizer"])(
            m_i_s.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
            **self.hparams["optim_kwargs"],
        )

        for i in pbar:
            m_i_s.train()
            class_loss = m_i_s.train_one_epoch(X, y, loss_weights=loss_weights)
            m_i_s.eval()
            m_i_s_out = m_i_s(X + self.hparams["noise_var"] * torch.randn_like(X))
            m_i_t_out = m_i_t(X + self.hparams["noise_var"] * torch.randn_like(X))
            consistency_loss = self.consistency_loss(m_i_s_out, m_i_t_out)
            loss = class_loss + self.hparams["reg"] * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m_i_t.train()  # just in case?
            ema.update()
            ema.copy_to(m_i_t.parameters())
            m_i_t.eval()

            train_acc_s = accuracy_score(m_i_s.predict(X).cpu(), y.cpu())
            train_acc_t = accuracy_score(m_i_t.predict(X).cpu(), y.cpu())

            results_dict = {
                    "L_tr_s": f"{loss.item():.3f} ({class_loss:.3f},{consistency_loss:.3f})",
                    "A_tr_s": train_acc_s,
                    "A_tr_t": train_acc_t
                }

            test_loss_s, test_acc_s = m_i_s.validate(
                X_val, y_val, loss_weights=loss_weights)
            test_loss_t, test_acc_t = m_i_t.validate(
                X_val, y_val, loss_weights=loss_weights)
            results_dict |= {"L_ts_s": test_loss_s, "A_ts_s": test_acc_s,
                             "L_ts_t": test_loss_t, "A_ts_t": test_acc_t}

            pbar.set_postfix(results_dict)
            # TODO: do early stoppting check?

        if logger is not None and self.global_step % 20 == 0:
            logger.add_scalar(f"cls_loss/{self.parent}/train", class_loss, self.global_step)
            logger.add_scalar(f"mse_con_loss/{self.parent}/train", consistency_loss, self.global_step)
            logger.add_scalar(f"loss/{self.parent}/train", loss, self.global_step)

            logger.add_scalar(f"acc_s/{self.parent}/train", train_acc_s, self.global_step)
            logger.add_scalar(f"acc_t/{self.parent}/train", train_acc_t, self.global_step)

            logger.add_scalar(f"loss_s/{self.parent}/val", test_loss_s, self.global_step)
            logger.add_scalar(f"loss_s/{self.parent}/val", test_loss_t, self.global_step)
            logger.add_scalar(f"acc_s/{self.parent}/val", test_acc_s, self.global_step)
            logger.add_scalar(f"acc_t/{self.parent}/val", test_acc_t, self.global_step)
        # only log on lost epoch for ensemble training
        return m_i_t, loss.item(), test_loss_t

    def fit(self, X, y, A=None, T=None, X_val=None, y_val=None, A_val=None, T_val=None, loss_weights=None, no_validate=False, logger=None):
        if not no_validate:
            assert X_val is not None
            assert y_val is not None
        X, y = torch.FloatTensor(X).to(
            self.device), torch.LongTensor(y).to(self.device)
        np.random.seed(self.seed)
        m_i, train_loss, test_loss = self.train_mean_teacher_ensemble(
            X, y,  X_val, y_val, loss_weights, logger=logger)
        m_best = copy.deepcopy(m_i)
        m_best.load_state_dict(m_i.state_dict())
        self.load_state_dict(m_i.state_dict())

        self.filter_mask = torch.ones_like(y).bool().to(self.device)
        self.prediction_cache = torch.zeros(y.size(0), 2).to(self.device)

        preds_i = m_i.predict(X_val).cpu()
        preds_best = m_best.predict(X_val).cpu()
        if len(np.unique(y_val)) == 1:
            warnings.warn("y_val has only one class; AUC is undefined, so SELF step is not well-defined. Returning initial ensemble")
            auc_i = 0.
            auc_best = 0.
            self.fitted_flag_ = True
            return train_loss, test_loss
        else:
            auc_i = roc_auc_score(y_val, preds_i)
            # use the final metric for evaluating model
            auc_best = roc_auc_score(y_val, preds_best)

        n_epochs_no_improve = 0
        for i in range(self.hparams["max_iters"]):
            print(f"\nSELF iter. ({i}/{self.hparams['max_iters']}): l_tr {train_loss:.3f} - l_ts {test_loss:.3f} - auc_i {auc_i:.3f} - auc_best {auc_best:.3f}")
            m_best = copy.deepcopy(m_i)
            m_best.load_state_dict(m_i.state_dict())
            best_probs = m_best.predict_proba(X)
            self.prediction_cache = self.hparams["alpha"] * self.prediction_cache + (1 - self.hparams["alpha"]) * best_probs

            n_remaining = self.filter_mask.sum()
            self.filter_mask[(y != torch.max(self.prediction_cache, 1)[1])] = False
            if self.hparams.get("no_filter_positives", False): # Auto-correct filtered-out positives
                n_after_filter = self.filter_mask.sum()
                self.filter_mask[y == 1] = True
                print(f"Restored {self.filter_mask.sum() - n_after_filter} examples")
                if logger is not None and self.global_step % 20 == 0:
                    logger.add_scalar(f"n_restored_after_filter/{self.parent}/train", self.filter_mask.sum() - n_after_filter, self.global_step)

            print(f"Filtered {n_remaining - self.filter_mask.sum()} examples")
            if logger is not None and self.global_step % 20 == 0:
                logger.add_scalar(f"n_filtered/{self.parent}/train", n_remaining - self.filter_mask.sum(), self.global_step)
            
            m_i, train_loss, test_loss = self.train_mean_teacher_ensemble(
                X[self.filter_mask],
                y[self.filter_mask],
                X_val, y_val,
                loss_weights, logger=logger
            )
            # Logged in inner loop

            self.filter_mask = torch.ones_like(y).bool()
            preds_i = m_i.predict(X_val).cpu()
            preds_best = m_best.predict(X_val).cpu()
            auc_i = roc_auc_score(y_val, preds_i)
            auc_best = roc_auc_score(y_val, preds_best)

            if logger is not None and self.global_step % 20 == 0:
                pred_probs_train = m_i.predict_proba(X)
                pred_probs_val = m_i.predict_proba(X_val)
                self.log_fairness_metrics_for_split(logger, pred_probs_train, y, A, "train")
                self.log_fairness_metrics_for_split(logger, pred_probs_val, y_val, A_val, "val")
                logger.add_scalar(f"auc_new/{self.parent}", auc_i, self.global_step)
                logger.add_scalar(f"auc_best/{self.parent}", auc_best, self.global_step)
                logger.add_scalar(f"auc_comp_diff/{self.parent}", auc_i - auc_best, self.global_step)

            if auc_i > auc_best:
                n_epochs_no_improve = 0
                self.load_state_dict(m_i.state_dict())
            else:
                n_epochs_no_improve += 1
                print(f"Failed to improve for {n_epochs_no_improve} epochs")
            if n_epochs_no_improve == self.hparams["self_patience"]:
                break
            self.global_step += 1

        print(
            f"\nSELF iter. ({i}/{self.hparams['max_iters']}): l_tr {train_loss:.3f} - l_ts {test_loss:.3f} - auc_i {auc_i:.3f} - auc_best {auc_best:.3f}")
        self.fitted_flag_ = True
        return train_loss, test_loss


class DivideMixEstimator(SimpleMLP):
    """
        Heavily based on the original DivideMix repo for CIFAR: https://github.com/LiJunnan1992/DivideMix/blob/master/Train_cifar.py

        See Lee, Socher, and Hoi (2021).
    """

    def __init__(self, input_size, *args, **kwargs):
        super().__init__(input_size, *args, **kwargs)
        self.model1 = SimpleMLP(input_size, *args, **kwargs)
        self.model2 = SimpleMLP(input_size, *args, seed=43, **kwargs)
        self.gmm_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.gmm_loss_ema = None 

        self.mm = GaussianMixture(n_components=2, warm_start=self.hparams["gmm_warm_start"])

        import os
        os.environ['OPENBLAS_NUM_THREADS'] = '1' 

    def codivide(self, X_, y_, T, logger=None):
        gmm_clean_threshold = self.hparams["p_threshold"]
        probs1 = self.gmm_for_model("model1", X_, y_, logger=logger)
        probs2 = self.gmm_for_model("model2", X_, y_, logger=logger)
        clean1_mask = (probs1 > gmm_clean_threshold)  # thresholded P(clean)
        clean2_mask = (probs2 > gmm_clean_threshold)
        
        if self.hparams.get("no_filter_positives", False):
            clean1_mask |= (y_ == 1).cpu().numpy()
            clean2_mask |= (y_ == 1).cpu().numpy()
        return probs1, probs2, clean1_mask, clean2_mask

    def coguess(self, models, X, temp):
        # co-guesses labeles in a split
        outs = reduce(op.add, [F.softmax(m(X), dim=1)
                      for m in models]) / len(models)
        outs = outs ** (1 / temp)
        final_outs = outs / outs.sum(dim=1, keepdim=True)
        return final_outs.detach()

    def cotrain(self, i, training_model, fixed_model, optim, mask, probs, X_, y_, true_y=None, logger=None):
        """
            Involves:
                * Coguessing on unlabeled sample
                * Corefining on labeled sample
                * Optional mixmatch
        """
        training_model.train()
        fixed_model.eval()
        X_labeled, X_unlabeled = X_[mask], X_[~mask]
        y_labeled, y_unlabeled = y_[mask], y_[~mask]
        targets = torch.full((y_.size(0), 2), float('nan')).to(self.device)
        optim.zero_grad()

        model_id = self.get_model_id(training_model)
        with torch.no_grad():
            # skip augmentation -- d/n apply for tabular
            # Co-guess: P(Y) = average of model probabilities (w/ softmax temp. renormalization)
            targets_u = self.coguess([training_model, fixed_model], X_unlabeled, self.hparams["T"])
            targets_l = self.coguess([training_model, fixed_model], X_labeled, self.hparams["T"])
            targets[mask], targets[~mask] = targets_l, targets_u
            if true_y is not None:
                preds = torch.square(true_y - targets[np.arange(len(true_y)), true_y])
                if logger is not None and self.global_step % 20 == 0:
                    logger.add_scalar(f"gmm_mse/{self.parent}/{model_id}/total", preds.mean().item(), self.global_step)
                    logger.add_scalar(f"gmm_mse/{self.parent}/{model_id}/clean", preds[mask].mean().item(), self.global_step)
                    logger.add_scalar(f"gmm_mse/{self.parent}/{model_id}/unclean", preds[~mask].mean().item(), self.global_step)
            assert not torch.any(targets.isnan())

        if self.hparams["mixmatch"]:
            # Mixmatch -- unsure it'll help in tabular setting
            b_ = np.random.beta(self.hparams["mixmatch_alpha"], self.hparams["mixmatch_alpha"])
            b_ = max(b_, 1 - b_)

            idx = torch.randperm(X_.size(0))
            mix_X = b_ * X_ + (1 - b_) * X_[idx]
            mix_Y = b_ * targets + (1 - b_) * targets[idx]
            mix_out = training_model(mix_X)
            out, targets = mix_out, mix_Y
        else:
            out = training_model(X_)
        lx, lu = semi_loss(out, targets, mask)

        # regularize predictions -- prevent class collapse
        prior = torch.ones(2).to(self.device) * 0.5  # [1/2, 1/2]
        mean_pred = torch.softmax(out, dim=1).mean(dim=0)
        lc = torch.sum(prior * torch.log(prior / mean_pred))

        # if there is a rampup
        reg_u = linear_warmup_multiplier(i, self.hparams.get("warmup", 0)) * self.hparams["lambda_u"]
        final_loss = lx + reg_u * lu + self.hparams["lambda_c"] * lc
        final_loss.backward()
        optim.step()

        if logger is not None and self.global_step % 20 == 0:
            logger.add_scalar(f"mixmatch_loss/{self.parent}/{model_id}/train", final_loss.item(), self.global_step)
            logger.add_scalar(f"ce_labeled/{self.parent}/{model_id}/train", lx.item(), self.global_step)
            logger.add_scalar(f"reg_u/{self.parent}/{model_id}/train", reg_u.item(), self.global_step)
            logger.add_scalar(f"mse_unlabeled/{self.parent}/{model_id}/train", lu.item(), self.global_step)
            logger.add_scalar(f"class_entropy/{self.parent}/{model_id}/train", lc.item(), self.global_step)

        return final_loss

    def get_model_id(self, model):
        if model is self.model1:
            return "1"
        elif model is self.model2:
            return "2"
        else:
            raise ValueError("Model object is not one of the DivideMix models")

    def warmup_epoch(self, model, optim, X_, y_, logger=None):
        model.train()
        optim.zero_grad()
        out = model(X_)

        # we already know that noise is asymmetric, so we just apply the negentropy regularization automatically
        class_loss = self.loss_fn(out, y_) 
        neg_loss = negentropy(out)
        loss = class_loss + neg_loss
        loss.backward()
        optim.step()

        if logger is not None and self.global_step % 20 == 0:
            model_id = self.get_model_id(model)
            logger.add_scalar(f"warmup_ce_loss/{self.parent}/{model_id}/train", class_loss, self.global_step)
            logger.add_scalar(f"warmup_negentropy/{self.parent}/{model_id}/train", neg_loss, self.global_step)
            logger.add_scalar(f"warmup_loss/{self.parent}/{model_id}/train", loss, self.global_step)
        return loss

    def fit(self, X, y, y_true=None, A=None, T=None, X_val=None, y_val=None, A_val=None, T_val=None, loss_weights=None, no_validate=False, logger=None):
        if not no_validate:
            assert X_val is not None
            assert y_val is not None
        optimizer1 = getattr(optim, self.hparams["optimizer"])(
            self.model1.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        optimizer2 = getattr(optim, self.hparams["optimizer"])(
            self.model2.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        pbar = tqdm(range(self.hparams["epochs"]), disable=no_validate)
        X_, y_, T_ = torch.FloatTensor(X).to(self.device), torch.LongTensor(y).to(self.device), torch.LongTensor(T).to(self.device)
        X_val, y_val = torch.FloatTensor(X_val).to(self.device), torch.LongTensor(y_val).to(self.device)
        y_true = torch.from_numpy(y_true).to(self.device)
        best_gmm_acc1 = 0.
        best_gmm_acc2 = 0.
        gmm1_break, gmm2_break = False, False
        gmm_patience = self.hparams.get("gmm_early_stop", 9999)
        n_epochs_no_improve_gmm1, n_epochs_no_improve_gmm2 = 0., 0.
        for i in pbar:
            results_dict = {}
            if i < self.hparams.get("warmup", 0):
                pbar.set_description("PHASE - WARMUP")
                loss1 = self.warmup_epoch(self.model1, optimizer1, X_, y_, logger=logger)
                loss2 = self.warmup_epoch(self.model2, optimizer2, X_, y_, logger=logger)
                # logged in inner loop 
            else:
                pbar.set_description("PHASE - DIVIDEMIX")
                # codivide using GMMs
                probs1, probs2, clean1_mask, clean2_mask = self.codivide(X_, y_, T_, logger=logger)
                if y_true is not None and self.global_step % 20 == 0:
                    is_clean = (y_true == y_).long().cpu()
                    acc1 = accuracy_score(is_clean, clean1_mask)
                    acc2 = accuracy_score(is_clean, clean2_mask)
                    if logger is not None:
                        logger.add_scalar(f"gmm_acc/{self.parent}/1/train", acc1, self.global_step)
                        logger.add_scalar(f"gmm_acc/{self.parent}/2/train", acc2, self.global_step)
                    results_dict |= {"A_gmm": f"{acc1}/{acc2}"}
                
                if acc1 > best_gmm_acc1:
                    best_gmm_acc1 = acc1
                    n_epochs_no_improve_gmm1 = 0.
                else:
                    n_epochs_no_improve_gmm1 += 1

                if n_epochs_no_improve_gmm1 == gmm_patience:
                    print("GMM #1 convergence has worsened -- stopping")
                    gmm1_break = True

                if acc2 > best_gmm_acc2:
                    best_gmm_acc2 = acc2
                    n_epochs_no_improve_gmm2 = 0.
                else:
                    n_epochs_no_improve_gmm2 += 1.

                if n_epochs_no_improve_gmm2 == gmm_patience:
                    print("GMM #2 convergence has worsened -- stopping")
                    gmm2_break = True
                if gmm1_break or gmm2_break:
                    break

                # train model 1 on split 2
                loss1 = self.cotrain(
                    i,
                    self.model1,
                    self.model2,
                    optimizer1,
                    clean2_mask,
                    probs2,
                    X_, y_,
                    true_y=y_true,
                    logger=logger
                )
                loss2 = self.cotrain(
                    i,
                    self.model2,
                    self.model1,
                    optimizer2,
                    clean1_mask,
                    probs1,
                    X_, y_,
                    true_y=y_true,
                    logger=logger
                )
                # logged in inner loop 
            with torch.no_grad():
                loss = loss1 + loss2
                acc = accuracy_score(self.predict(X_).cpu(), y_.cpu())
                results_dict |= {
                    "L_tr": f"{loss1:.3f}/{loss2:.3f}",
                    "A_tr": acc
                }
                if logger is not None and self.global_step % 20 == 0:
                    logger.add_scalar(f"loss/{self.parent}/train", loss, self.global_step)
                    logger.add_scalar(f"acc/{self.parent}/train", acc, self.global_step) 
                if no_validate:
                    test_acc = None, None
                else:
                    test_acc = accuracy_score(self.predict(X_val).cpu(), y_val.cpu())
                    if logger is not None and self.global_step % 20 == 0:
                        logger.add_scalar(f"acc/{self.parent}/val", test_acc, self.global_step)
                    results_dict |= {"A_ts": test_acc}
                if logger is not None and self.global_step % 20 == 0: 
                    pred_probs_train, pred_probs_val = self.predict_proba(X_), self.predict_proba(X_val)
                    self.log_fairness_metrics_for_split(logger, pred_probs_train, y_, A, "train")
                    self.log_fairness_metrics_for_split(logger, pred_probs_val, y_val, A_val, "val")

                pbar.set_postfix(results_dict)
            self.global_step += 1
        self.fitted_flag_ = True
        return loss.item() 

    def gmm_for_model(self, model_str, X_, y_, logger=None):
        model = getattr(self, model_str)
        with torch.no_grad():
            out = self(X_)
            losses = self.gmm_loss_fn(out, y_).cpu()
            losses = (losses - losses.min()) / (losses.max() - losses.min())

        if self.hparams["use_emwa_for_gmm"]:
            if self.gmm_loss_ema is None:
                self.gmm_loss_ema = losses
            else:
                self.gmm_loss_ema = self.hparams["gmm_emwa_weight"] * self.gmm_loss_ema + (1 - self.hparams["gmm_emwa_weight"]) * losses
            final_loss = self.gmm_loss_ema.reshape(-1, 1)
        else:
            final_loss = losses.reshape(-1, 1)
        self.mm.fit(final_loss)
        prob = self.mm.predict_proba(final_loss)
        prob = prob[:, self.mm.means_.argmin()]
        """mm_results = {"loss": final_loss, "prob": prob, "means": self.mm.means_, "weights": self.mm.weights_}
        if hasattr(self.mm, "covariances_"):
            mm_results |= {"cov": self.mm.covariances_}
        if hasattr(self.mm, "alphas_"):
            mm_results |= {"alphas": self.mm.alphas_, "betas": self.mm.betas_}
        self.mm_results.append(mm_results) #"cov": self.mm.covariances_})"""
        if logger is not None and self.global_step % 20 == 0:
            mean0, mean1 = self.mm.means_.ravel()
            var0, var1 = self.mm.covariances_.ravel()
            w0, w1 = self.mm.weights_.ravel()
            #logger.add_histogram(f"gmm_loss_histogram/{self.parent}/{model_str[-1]}", final_loss, self.global_step)
            logger.add_scalar(f"gmm_mean0/{self.parent}/{model_str[-1]}", mean0, self.global_step)
            logger.add_scalar(f"gmm_mean1/{self.parent}/{model_str[-1]}", mean1, self.global_step)
            logger.add_scalar(f"gmm_var0/{self.parent}/{model_str[-1]}", var0, self.global_step)
            logger.add_scalar(f"gmm_var1/{self.parent}/{model_str[-1]}", var1, self.global_step)
            logger.add_scalar(f"gmm_weight0/{self.parent}/{model_str[-1]}", w0, self.global_step)
            logger.add_scalar(f"gmm_weight1/{self.parent}/{model_str[-1]}", w1, self.global_step)
        return prob

    def forward(self, X, separate=False):
        if separate:
            return self.model1(X), self.model2(X)
        return self.model1(X) + self.model2(X)

    def predict_proba(self, X, A=None):
        with torch.no_grad():
            if not torch.is_tensor(X):
                # hacky, but otherwise doesn't play nice with the plotting code
                X = torch.from_numpy(X).to(
                    next(self.parameters()).device).float()
            out1, out2 = self(X, separate=True)
            return (F.softmax(out1, dim=-1) + F.softmax(out2, dim=-1)) / 2
