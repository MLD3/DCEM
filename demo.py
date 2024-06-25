import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn

from dcem import DCEM

from typing import Optional

class ToyNN(nn.Module): # for demonstration only. This architecture was not tuned and may not be optimal in practice.
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.lins = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, X):
        return self.lins(X)

if __name__ == '__main__':
    # the runtime of this demo is approximately 5-6 min.
    data = pd.read_csv("./data/demo_data.csv", index_col=(0, 1))
    train_df = data.xs("train", level=0)
    test_df = data.xs("test", level=0)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # these column names are from a toy dataset in the `./data/` directory.
    X_tr = torch.from_numpy(train_df.loc[:, ["x1", "x2"]].values).float().to(device)
    A_tr = torch.from_numpy(train_df.loc[:, ["a"]].values).long().to(device)
    T_tr = torch.from_numpy(train_df.loc[:, ["t"]].values).long().to(device) 
    Y_tr = torch.from_numpy(train_df.loc[:, ["y"]].values).long().to(device)
    Y_obs_tr = torch.from_numpy(train_df.loc[:, ["y_obs"]].values).long().to(device) 
    
    propensity_model = ToyNN(5) 
    outcome_model = ToyNN(2)
    propensity_model.to(device)
    outcome_model.to(device)

    dcem_estimator = DCEM(propensity_model, outcome_model, batch_size=20000) # our toy dataset is small enough to allow this.
    dcem_estimator.fit(X_tr, A_tr, T_tr, Y_obs_tr, Y=Y_tr)

    X_ts = torch.from_numpy(test_df.loc[:, ["x1", "x2"]].values).float().to(device)
    Y_ts = torch.from_numpy(test_df.loc[:, ["y"]].values).long().to(device)
    with torch.no_grad():
        preds = dcem_estimator.predict_proba(X_ts)[:, 1]
        print("AUC:", roc_auc_score(Y_ts.cpu().numpy(), preds.cpu().numpy()))
    torch.save(dcem_estimator.state_dict(), "demo_dcem_model.pkl")

