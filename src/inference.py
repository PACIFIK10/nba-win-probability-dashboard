"""
inference.py
-------------
Small helper that loads the trained WinProbabilityNet + normalization
stats once, and exposes a predict(features_dict) -> probability function
used by both the Flask HTTP endpoint and the WebSocket streamer.
"""
import json
import os

import numpy as np
import torch

from model import FEATURE_COLUMNS, WinProbabilityNet

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class WinProbabilityModel:
    def __init__(self, model_path: str = None, stats_path: str = None):
        model_path = model_path or os.path.join(MODELS_DIR, "win_probability_net.pt")
        stats_path = stats_path or os.path.join(MODELS_DIR, "norm_stats.json")

        if not os.path.exists(model_path) or not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run `python src/train.py` first "
                f"(after `python src/features.py`)."
            )

        with open(stats_path) as f:
            meta = json.load(f)
        self.feature_columns = meta["feature_columns"]
        self.stats = meta["stats"]

        self.model = WinProbabilityNet(n_features=len(self.feature_columns),
                                        hidden=meta.get("hidden", 32))
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def _normalize(self, features: dict) -> np.ndarray:
        vec = []
        for col in self.feature_columns:
            mean = self.stats[col]["mean"]
            std = self.stats[col]["std"] or 1.0
            vec.append((features.get(col, 0) - mean) / std)
        return np.array(vec, dtype=np.float32)

    def predict(self, features: dict) -> float:
        x = torch.from_numpy(self._normalize(features)).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(x)).item()
        return prob


_singleton = None


def get_model() -> WinProbabilityModel:
    global _singleton
    if _singleton is None:
        _singleton = WinProbabilityModel()
    return _singleton
