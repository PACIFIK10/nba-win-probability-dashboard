"""
model.py
--------
The "MODEL" layer of the tech stack: pytorch -> neural network.

A small feedforward net that takes the 4 features from the video
(score differential, time remaining, possession, fouls -- fouls expanded
into home/away foul counts + bonus flags for a bit more signal) and
outputs a live win probability for the home team.
"""
import torch
import torch.nn as nn

FEATURE_COLUMNS = [
    "score_diff",
    "seconds_remaining",
    "possession_home",
    "home_fouls",
    "away_fouls",
    "home_bonus",
    "away_bonus",
]


class WinProbabilityNet(nn.Module):
    def __init__(self, n_features: int = len(FEATURE_COLUMNS), hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x).squeeze(-1)
        return logits  # raw logits; apply sigmoid outside (BCEWithLogitsLoss)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))
