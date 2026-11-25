"""Real small model architectures to benchmark: logistic regression, random
forest, and a small PyTorch MLP."""
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class SmallMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x):
        return self.net(x)


def _train_mlp(X_train, y_train, X_test, n_features, epochs=60):
    model = SmallMLP(n_features)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(xt)
        loss = loss_fn(out, yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1).numpy()
    return preds


MODEL_NAMES = ["logistic_regression", "random_forest", "small_mlp"]


def run_architecture(name, X, y, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    start = time.perf_counter()
    if name == "logistic_regression":
        clf = LogisticRegression().fit(X_train, y_train)
        preds = clf.predict(X_test)
    elif name == "random_forest":
        clf = RandomForestClassifier(n_estimators=50, random_state=seed).fit(X_train, y_train)
        preds = clf.predict(X_test)
    elif name == "small_mlp":
        preds = _train_mlp(X_train, y_train, X_test, n_features=X.shape[1])
    else:
        raise ValueError(name)
    latency_ms = (time.perf_counter() - start) * 1000
    acc = accuracy_score(y_test, preds)
    return {"accuracy": float(acc), "latency_ms": float(latency_ms)}
