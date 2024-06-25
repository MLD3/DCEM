import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
import torch
import torch.nn as nn

DECISION_BOUNDARY_RESOLUTION = 100

def safe_scatter(ax, sample, marker, color, label, alpha):
    if len(sample) != 0:
        ax.scatter(*zip(*sample), marker=marker, color=color, label=label, alpha=alpha)

def preview_sim_data(X, A, Y, Y_obs, n_points=500, decision_fn=None, test_fn0=None, testfn_1=None, model_name=None, figsize=(8, 4), alpha=0.5, xmin=-0.45, xmax=0.8, ymin=-0.45, ymax=0.9, fig_bbox_to_anchor_x=1.2, fig_bbox_to_anchor_y=0.8, plotting_seed=42):
    assert X.shape[1] == 2
    if plotting_seed is not None:
        np.random.seed(plotting_seed)
    indices = np.random.choice(np.arange(X.shape[0]), size=min(n_points, X.shape[0]), replace=False)
    X_sample = X[indices]
    A_sample = A[indices]
    if decision_fn is not None:
        Y = decision_fn(X)
    Y_sample = Y[indices]
    if test_fn0 is not None and test_fn1 is not None:
        Y_obs = np.concatenate([test_fn0(X), test_fn1(X)])
    Y_obs_sample = Y_obs[indices]
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    titles = ["Y (true labels)", r"$\tilde{Y}$ (observed labels)"]
    label_set = [Y_sample, Y_obs_sample]
    for i, (a, subtitle, labels) in enumerate(zip(ax, titles, label_set)):
        a.set_xlim((xmin, xmax))
        a.set_ylim((ymin, ymax))
        a.set_xlabel("$x_1$")
        a.set_ylabel("$x_2$")
        a.set_title(subtitle)

        safe_scatter(a, X_sample[(A_sample == 0) & (labels == 0)], marker='o', color='tab:blue', label='$a=0$ (-)' if i == 0 else None, alpha=alpha)
        safe_scatter(a, X_sample[(A_sample == 0) & (labels == 1)], marker='+', color='tab:blue', label='$a=0$ (+)' if i == 0 else None, alpha=alpha)
        safe_scatter(a, X_sample[(A_sample == 1) & (labels == 0)], marker='o', color='tab:orange', label='$a=1$ (-)' if i == 0 else None, alpha=alpha)
        safe_scatter(a, X_sample[(A_sample == 1) & (labels == 1)], marker='+', color='tab:orange', label='$a=1$ (+)' if i == 0 else None, alpha=alpha)


    lgd = fig.legend(bbox_to_anchor=(fig_bbox_to_anchor_x, fig_bbox_to_anchor_y), ncol=1, title="Legend")
    fig.tight_layout()
    return fig, ax

def plot_model_predictions(X_val, clf, title=None, fig=None, ax=None, alpha=0.5, figsize=(4, 4), xmin=-0.45, xmax=0.8, ymin=-0.45, ymax=0.9, from_estimator=True, z=None, grid=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    if not isinstance(ax, np.ndarray): ax = [ax]
    if isinstance(clf, nn.Module):
        X_val = torch.from_numpy(X_val).float().to("cpu")
        clf = clf.to("cpu")
    for ax_ in ax:
        if from_estimator:
            disp = DecisionBoundaryDisplay.from_estimator(
                 clf, X_val, response_method="predict_proba",
                 xlabel="$x_1$", ylabel="$x_2$",
                 alpha=alpha, ax=ax_, **kwargs,
            )
        else:
            """
                Adapted from scikit-learn code. 
            """
            if z is None or grid is None:
                raise ValueError("Must pass in z and grid (model response) explicitly when creating a decision boundary plot without `from_estimator`.") 
            if torch.is_tensor(z):
                z = z.to("cpu")
            xx0, xx1 = grid
            disp = DecisionBoundaryDisplay(
                xx0=xx0,    
                xx1=xx1,
                response=z.reshape(xx0.shape),
                xlabel="$x_1$",
                ylabel="$x_2$",
            ).plot(ax=ax_, plot_method="contourf", alpha=alpha, **kwargs)
        ax_.set_xlim((xmin, xmax))
        ax_.set_ylim((ymin, ymax))
    if fig is not None:
        fig.subplots_adjust(top=0.8)
        fig.suptitle("Sample of simulated data and decision boundary" + f", {title}" if title is not None else "", y=0.99)
    return fig, ax

def plot_boundaries(sim, plot_points=100, fig=None, ax=None):
    if not isinstance(ax, np.ndarray): ax = [ax]
    for ax_ in ax:
        xmin, xmax = ax_.get_xlim()
        ymin, ymax = ax_.get_ylim()
        x = np.linspace(xmin, xmax, plot_points)
        y = np.linspace(ymin, ymax, plot_points)
        xx, yy = np.meshgrid(x, y) # shape (plot_points, plot_points)
        XY = np.stack([xx, yy], axis=-1).reshape((-1, 2))


        test_0 = sim.test_decision_fn0
        test_1 = sim.test_decision_fn1
        label_fn = sim.label_decision_fn0
        assert sim.label_decision_fn0 == sim.label_decision_fn1
        th0 = sim.test_th0
        th1 = sim.test_th1
        y_bd = sim.label_th0
        assert sim.label_th0 == sim.label_th1
        
        # to plot testing thresholds -- need to solve eqn test_0(x) = test_threshold_group0
        test_0_scores = test_0(XY).reshape((plot_points, plot_points))
        test_1_scores = test_1(XY).reshape((plot_points, plot_points))
        label_scores = label_fn(XY).reshape((plot_points, plot_points))
        ax_.contour(xx, yy, test_0_scores, levels=[th0], colors="blue", linestyles="dashed")
        ax_.contour(xx, yy, test_1_scores, levels=[th1], colors="orange", linestyles="dashed")
        ax_.contour(xx, yy, label_scores, levels=[y_bd], colors="black")

    return fig, ax

