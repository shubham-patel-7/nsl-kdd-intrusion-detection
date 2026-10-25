"""Model zoo: Logistic Regression, Random Forest, XGBoost (classic ML) and a PyTorch MLP (DL).

Every model exposes the sklearn-style `.fit(...)` / `.predict_proba(X)` so evaluation code is uniform.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "MLP (PyTorch)"]
STOCHASTIC = {"Random Forest", "XGBoost", "MLP (PyTorch)"}  # averaged over several seeds


class MLPNet(nn.Module):
    def __init__(self, n_in: int, n_classes: int, hidden=(128, 64), dropout: float = 0.3):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchMLP:
    """Small feed-forward network trained with AdamW + early stopping on validation loss."""

    def __init__(self, n_in: int, n_classes: int, seed: int = 0, epochs: int = 40,
                 batch: int = 512, lr: float = 1e-3, patience: int = 5):
        self.n_in, self.n_classes, self.seed = n_in, n_classes, seed
        self.epochs, self.batch, self.lr, self.patience = epochs, batch, lr, patience
        torch.manual_seed(seed)
        self.net = MLPNet(n_in, n_classes)
        self.history: dict = {"train_loss": [], "val_loss": []}

    def fit(self, X, y, X_val, y_val, class_weight=None):
        torch.manual_seed(self.seed)
        w = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        xt, yt = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int64))
        xv, yv = torch.from_numpy(X_val.astype(np.float32)), torch.from_numpy(y_val.astype(np.int64))
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, bad = float("inf"), None, 0
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(xt), generator=g)
            total = 0.0
            for i in range(0, len(xt), self.batch):
                idx = perm[i : i + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            self.net.eval()
            with torch.no_grad():
                val = loss_fn(self.net(xv), yv).item()
            self.history["train_loss"].append(total / len(xt))
            self.history["val_loss"].append(val)
            if val < best - 1e-4:
                best, best_state, bad = val, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        self.net.eval()
        out = []
        X = X.astype(np.float32)
        for i in range(0, len(X), 8192):
            out.append(torch.softmax(self.net(torch.from_numpy(X[i : i + 8192])), dim=1).numpy())
        return np.concatenate(out)

    def save(self, path: Path) -> None:
        torch.save({"state": self.net.state_dict(), "n_in": self.n_in, "n_classes": self.n_classes}, path)

    @classmethod
    def load(cls, path: Path) -> "TorchMLP":
        blob = torch.load(path, map_location="cpu")
        m = cls(blob["n_in"], blob["n_classes"])
        m.net.load_state_dict(blob["state"])
        m.net.eval()
        return m


def fit_model(name: str, X, y, X_val, y_val, n_classes: int, seed: int, weights: np.ndarray):
    """Train one model. `weights` are per-class weights (square-root inverse frequency)."""
    sample_w = weights[y]
    if name == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        return m.fit(X, y, sample_weight=sample_w)
    if name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier

        m = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=seed)
        return m.fit(X, y, sample_weight=sample_w)
    if name == "XGBoost":
        from xgboost import XGBClassifier

        m = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", n_jobs=-1, random_state=seed, early_stopping_rounds=20,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
        )
        return m.fit(X, y, sample_weight=sample_w, eval_set=[(X_val, y_val)], verbose=False)
    if name == "MLP (PyTorch)":
        return TorchMLP(X.shape[1], n_classes, seed=seed).fit(X, y, X_val, y_val, class_weight=weights)
    raise ValueError(name)
