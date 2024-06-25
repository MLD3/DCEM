import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm

from losses import causal_regularization_loss
from utils import  _enforce_scalar_bounds, _query_tensor_dataset

from typing import List, Optional

class DCEM(nn.Module):
    def __init__(
        self,
        propensity_model,
        outcome_model,
        propensity_epochs: Optional[int] = 200,
        propensity_lr: Optional[float] = 1e-3,
        propensity_weight_decay: Optional[float] = 0.,
        propensity_loss_fn: Optional[nn.Module] = nn.CrossEntropyLoss(),
        outcome_epochs: Optional[int] = 100,
        outcome_loss_fn: Optional[nn.Module] = nn.CrossEntropyLoss(reduction="none"),
        lr: Optional[float] = 1e-3,
        weight_decay: Optional[float] = 0.,
        max_em_iterations: Optional[int] = 20,
        em_early_stop_patience: Optional[int] = 3,
        val_size: Optional[float] = 0.3,
        batch_size: Optional[int] = 128,
        seed: Optional[int] = 42,
    ):
        """Initializes the DCEM estimator.

        Args:
            propensity_model (nn.Module): the base module for the propensity model
            outcome_model (nn.Module): the base module for the final outcome model
            propensity_epochs (Optional[int], optional): Number of epochs to train the propensity model. Defaults to 200.
            propensity_lr (Optional[float], optional): Propensity model learning rate. Defaults to 1e-3.
            propensity_weight_decay (Optional[float], optional): Propensity model weight decay. Defaults to 0..
            propensity_loss_fn (Optional[nn.Module], optional): Loss function for training the propensity model. Defaults to nn.CrossEntropyLoss().
            outcome_epochs (Optional[int], optional): Number of epochs to train the outcome model (per M-step). Defaults to 100.
            outcome_loss_fn (Optional[nn.Module], optional): Loss function for training the outcome model. Defaults to nn.CrossEntropyLoss(reduction="none").
            lr (Optional[float], optional): Learning rate for training the outcome model. Defaults to 1e-3.
            weight_decay (Optional[float], optional): Weight decay for the outcome model. Defaults to 0..
            max_em_iterations (Optional[int], optional): Maximum number of EM iterations. Defaults to 20.
            em_early_stop_patience (Optional[int], optional): Number of EM iterations without validation loss improvement before stopping EM. Defaults to 3.
            val_size (Optional[float], optional): Proportion of data to be reserved for validation. Defaults to 0.3.
            batch_size (Optional[int], optional): Batch size for training. Defaults to 128.
            seed (Optional[int], optional): Random seed, used mostly for train-val splitting. Defaults to 42.
        """
        super().__init__()
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        self.propensity_epochs = propensity_epochs
        self.propensity_lr = propensity_lr
        self.propensity_weight_decay = propensity_weight_decay
        self.propensity_loss_fn = propensity_loss_fn
        self.outcome_epochs = outcome_epochs
        self.outcome_loss_fn = outcome_loss_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_iterations = max_em_iterations
        self.patience = em_early_stop_patience
        self.val_size = val_size
        self.batch_size = batch_size
        self.seed = seed

        _enforce_scalar_bounds(self.val_size, "val_size", _min=0., _max=1.,)
        _enforce_scalar_bounds(self.batch_size, "batch size", _min=0)

        self.outcome_loss_fn.reduction = "none" # very important -- need to reweight the loss

        if not isinstance(self.outcome_model, nn.Module):
            t = type(self.outcome_model)
            raise ValueError(f"The outcome model must be a PyTorch module, but got a model of type {t}")
        if hasattr(self.propensity_model, 'predict_proba'): # duck-typing check for sklearn estimator
            self.propensity_pred_fn = getattr(self.propensity_model, 'predict_proba')
        elif isinstance(self.propensity_model, nn.Module): # otherwise, we assume that `forward` yields logits/probabilities
            self.propensity_pred_fn = getattr(self.propensity_model, 'forward')
        else:
            raise ValueError(f"The propensity model must either implement a scikit-learn style `predict_proba` function, or be a PyTorch module.")

    def fit_propensity_model(self, train_loader: DataLoader, val_loader: DataLoader):
        """This function fits the propensity model once at the start of the algorithm. 

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for the train split.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation split.
        """
        propensity_optim = torch.optim.Adam(
            self.propensity_model.parameters(),
            lr=self.propensity_lr,
            weight_decay=self.propensity_weight_decay
        )
        with tqdm(range(self.propensity_epochs), desc="Fitting propensity model") as pbar:
            for _ in pbar:
                prop_loss = 0.
                for _, batch in enumerate(train_loader):
                    X, A, T, *_ = batch
                    XA = torch.cat([X, X * A.view(-1, 1), A.view(-1, 1)], dim=1)

                    propensity_optim.zero_grad()
                    t_pred = self.propensity_model(XA)
                    batch_prop_loss = self.propensity_loss_fn(t_pred, T.squeeze(-1))
                    batch_prop_loss.backward()
                    propensity_optim.step()

                    prop_loss += batch_prop_loss.item()
                with torch.no_grad():
                    prop_scores = self.get_propensity_score(val_loader, return_logits=False)
                    T = _query_tensor_dataset(val_loader.dataset, "T")
                    pbar.set_postfix(
                        l_tr=prop_loss,
                        l_ts=F.binary_cross_entropy(prop_scores, T.float()).item(),
                        auc_ts=roc_auc_score(T.cpu().numpy(), prop_scores.cpu().numpy())
                    )

    def get_propensity_score(self, dataloader: DataLoader, return_logits: Optional[bool] = True):
        """Returns the propensity score.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the relevant split.
            return_logits (Optional[bool], optional): If false, returns probabilities; otherwise, returns logits. Defaults to True.

        Returns:
            tensor: Propensity scores (probabilities or logits).
        """
        prop_scores = []
        all_t = []
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                X, A, T, *_ = batch
                XA = torch.cat([X, X * A.view(-1, 1), A.view(-1, 1)], dim=1)
                t_logits = self.propensity_model(XA)
                if return_logits:
                    prop_score = t_logits
                else:
                    prop_score = F.softmax(t_logits, dim=-1)[:, 1]
                prop_scores.append(prop_score)
                all_t.append(T)
            return torch.cat(prop_scores, dim=0) 
            
    def get_outcome_estimates(
        self,
        dataloader: DataLoader,
        estep: Optional[bool] = False, 
        return_loss: Optional[bool] = False, # additionally returns the causal loss. 
    ):
        """This function is used whenever we need to use estimates of P(Y|X) or related quantities (e.g., the E-step estimates).

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the relevant split.
            estep (Optional[bool], optional): Boolean flag for whether we want to return E-step estimates. Defaults to False.
            return_loss (Optional[bool], optional): Boolean flag for whether we want to also return the M-step loss. Defaults to False. Only available when `estep=False`.

        Returns:
        One of the following:
            * tensor: E-step estimates, if `estep=True`
            * tensor: estimates of P(Y|X), if `estep=False`
            * Tuple[tensor, float]: estimates of P(Y|X) and the M-step loss associated with said estimates, if `estep=False` and `return_loss=True`.
        """
        outcome_estimates = []
        e_step_estimates = []
        with torch.no_grad():
            loss = 0.
            for _, batch in enumerate(dataloader):
                if len(batch) > 6: # dcem_train -- includes propensity score and potentially the E-step estimate
                    X, _, T, Y_obs, _, prop_score, y_post = batch       
                elif len(batch) > 5:
                    X, _, T, Y_obs, _, prop_score = batch # prior to the first E-step, these will not be initialized      
                else:
                    X, _, T, Y_obs, _ = batch
                y_out = self.outcome_model(X)
                outcome_estimates.append(y_out)
                if estep:
                    y_probs = F.softmax(y_out, dim=-1)[:, 1]
                    e_step_outputs = torch.where(T.squeeze(-1) == 1, Y_obs.squeeze(-1), y_probs)
                    e_step_estimates.append(e_step_outputs)
                if return_loss:
                    c_loss = causal_regularization_loss(self.outcome_loss_fn, Y_obs.squeeze(-1), y_post, y_out, prop_score)
                    loss += c_loss.item()
            if estep:
                e_step_estimates = torch.cat(e_step_estimates)
            outcome_estimates = torch.cat(outcome_estimates)
        if estep:
            return e_step_estimates
        else:
            if return_loss:
                return y_out, loss
            else:
                return y_out


    def m_step(self, train_loader: DataLoader, val_loader: DataLoader):
        """This function abstracts away the M-step.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for train split.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation split.

        Returns:
            Tuple[float, float]: training, validation loss (in a tuple)
        """

        optim = torch.optim.Adam(
            self.outcome_model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
    
        with tqdm(range(self.outcome_epochs), desc="Fitting outcome model") as pbar:
            for _ in pbar:
                loss = 0.
                for _, batch in enumerate(train_loader):
                    X, _, _, _, Y_obs, prop_score, y_post = batch
                    
                    optim.zero_grad()
                    y_pred = self.outcome_model(X)
                    batch_loss = causal_regularization_loss(self.outcome_loss_fn, Y_obs.squeeze(-1), y_post, y_pred, prop_score)
                    batch_loss.backward()
                    optim.step()
                    loss += batch_loss.item()
                with torch.no_grad():
                    y_preds, val_loss = self.get_outcome_estimates(val_loader, return_loss=True)
                    y_probs = F.softmax(y_preds, dim=-1)[:, 1]
                    Y = _query_tensor_dataset(val_loader.dataset, "Y") 
                    pbar.set_postfix(
                        l_tr=loss,
                        auc_ts=None if torch.isnan(Y).all() else roc_auc_score(Y.cpu().numpy(), y_probs.cpu().numpy()),
                    )
        return loss, val_loss
    
    def train_val_from_tensors(self, tensors: List[torch.Tensor], seed: Optional[int] = None):
        """This function reinitializes dataset splits given a list of tensors. This is useful, since we need to maintain
        the same train-val split while adding the propensity score and E-step estimates to the dataset, while allowing for 
        E-step estimates to change every epoch.

        Args:
            tensors (List[torch.Tensor]): List of tensors (e.g., [X, A, T, ...])
            seed (int, optional): A random seed, if overriding self.seed. Defaults to None.

        Returns:
            _type_: _description_
        """
        seed = self.seed if seed is None else seed
        rng = torch.Generator().manual_seed(seed) 
        dataset = TensorDataset(*tensors)
        train_data, val_data = random_split(dataset, [1 - self.val_size, self.val_size], generator=rng)
        return train_data, val_data

    def fit(
        self,
        X,
        A,
        T,
        Y_obs,
        Y: Optional[torch.Tensor] = None, # only used for factual evaluation on synthetic data
        iterations: Optional[int] = None,
        val_size: Optional[float] = None,
    ) -> None:
        """This function fits the outcome model for DCEM.

        Args:
            X (array-like): Covariates.
            A (array-like): "Protected attribute," or any variable that does not cause the outcome.
            T (array-like): Labeling decision.
            Y_obs (_type_): Observed labels.
            Y (Optional[torch.Tensor], optional): True labels. Only collectable in synthetic data. Defaults to None.
            val_size (Optional[float], optional): Proportion of data to use for validation (random split), if overriding self.val_size. Defaults to None.
        """
        
        iterations = self.max_iterations if iterations is None else iterations
        val_size = self.val_size if val_size is None else val_size

        if Y is None:
            Y = torch.ones_like(Y_obs) # sentinel value (invalid)
            Y.fill_(np.nan)
        train_data, val_data = self.train_val_from_tensors([X, A, T, Y_obs, Y])
        rng = torch.Generator().manual_seed(self.seed)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, generator=rng)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False) 
        self.fit_propensity_model(train_loader, val_loader)

        # here we want to create a new version of the training set augmented w/ the propensity score
        prop_score = self.get_propensity_score(train_loader)
        prop_score_val = self.get_propensity_score(val_loader) # not strictly needed, but useful for model selection
        all_prop_scores = torch.cat([prop_score, prop_score_val], dim=0)
        final_train_data, final_val_data = self.train_val_from_tensors([X, A, T, Y_obs, Y, all_prop_scores])
 
        dcem_train_loader = DataLoader(final_train_data, batch_size=self.batch_size, shuffle=True, generator=rng)
        dcem_val_loader = DataLoader(final_val_data, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float('inf')
        for i in range(iterations):
            if verbose >= 0:
                print(f"EM Epoch {i+1}/{iterations}")

            # e-step
            e_step_estimates = self.get_outcome_estimates(dcem_train_loader, estep=True, return_loss=False)
            e_step_val = self.get_outcome_estimates(dcem_val_loader, estep=True, return_loss=False)
            
            # m-step
            all_estep_estimates = torch.cat([e_step_estimates, e_step_val], dim=0)
            final_train_data, final_val_data = self.train_val_from_tensors([X, A, T, Y_obs, Y, all_prop_scores, all_estep_estimates])
            dcem_train_loader = DataLoader(final_train_data, batch_size=self.batch_size, shuffle=True, generator=rng)
            dcem_val_loader = DataLoader(final_val_data, batch_size=self.batch_size, shuffle=False)
             
            # evaluation
            loss, val_loss = self.m_step(dcem_train_loader, dcem_val_loader)
            Y_ts = _query_tensor_dataset(dcem_val_loader.dataset, "Y")
            Y_tr = _query_tensor_dataset(dcem_train_loader.dataset, "Y")
            log_str = f"loss {loss:.3f} | val loss {val_loss:.3f}" 
            if not torch.isnan(Y_tr).all(): # if we have factual data, we can do a little extra evaluation
                unshuffled_train_loader = DataLoader(final_train_data, batch_size=self.batch_size, shuffle=False) 
                train_preds = self.get_outcome_estimates(unshuffled_train_loader, estep=False) 
                val_preds = self.get_outcome_estimates(dcem_val_loader, estep=False) 
                train_probs = F.softmax(train_preds, dim=-1)[:, 1]
                val_probs = F.softmax(val_preds, dim=-1)[:, 1]
                auroc = roc_auc_score(Y_tr.cpu().numpy(), train_probs.cpu().numpy())
                val_auroc = roc_auc_score(Y_ts.cpu().numpy(), val_probs.cpu().numpy())
                log_str += f" | roc {auroc:.3f} | val roc {val_auroc:.3f}"
            print(log_str)

            # early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(epochs_no_improve, f"epochs without improvement (max: {self.patience})")

            if epochs_no_improve == self.patience:
                print("Early stopping triggered")
                break

    def predict(self, X) -> torch.Tensor:
        """Convenience function, to replicate scikit-learn-like behavior. Non-batched.

        Args:
            X (torch.Tensor): Covariates

        Returns:
            Tensor: A tensor with predicted P(Y|X) [binary], based on a 0.5 threshold.
        """
        y_probs = self.predict_proba(X)
        y_preds = (y_probs[:, 1] > 0.5)
        return y_preds.long()

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Convenience function, to replicate scikit-learn-like behavior. Non-batched.

        Args:
            X (torch.Tensor): Covariates

        Returns:
            Tensor: A tensor with estimates of P(Y|X)
        """
        logits = self.outcome_model(X)
        y_probs = F.softmax(logits, dim=-1)
        return y_probs
