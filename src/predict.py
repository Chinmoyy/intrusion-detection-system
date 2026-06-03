"""
Network Intrusion Detection - Prediction Module
Load trained model and classify network traffic records.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Union

MODEL_DIR = "models"

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]


class IntrusionDetector:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.model     = None
        self.scaler    = None
        self.encoders  = None
        self.meta      = None
        self._load()

    def _load(self):
        meta_path = os.path.join(self.model_dir, "metrics.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                "No trained model found. Run: python src/train.py"
            )
        with open(meta_path) as f:
            self.meta = json.load(f)

        best = self.meta["best_model"].lower().replace(" ", "_")
        self.model    = joblib.load(os.path.join(self.model_dir, f"{best}.pkl"))
        self.scaler   = joblib.load(os.path.join(self.model_dir, "scaler.pkl"))
        self.encoders = joblib.load(os.path.join(self.model_dir, "encoders.pkl"))
        print(f"[✓] Loaded model: {self.meta['best_model']}")

    def predict(self, record: Union[dict, pd.DataFrame]) -> dict:
        """
        Classify one or more network records.
        record can be a single dict or a DataFrame.
        Returns list of prediction dicts.
        """
        if isinstance(record, dict):
            df = pd.DataFrame([record])
        else:
            df = record.copy()

        cat_cols = ["protocol_type", "service", "flag"]
        num_cols = [c for c in COLUMNS if c not in cat_cols]

        for col in cat_cols:
            le = self.encoders[col]
            df[col] = df[col].map(
                lambda v: v if v in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

        df[num_cols] = self.scaler.transform(df[num_cols])
        X = df[COLUMNS].values

        preds  = self.model.predict(X)
        probas = self.model.predict_proba(X)[:, 1]

        results = []
        for pred, prob in zip(preds, probas):
            results.append({
                "is_attack":   bool(pred),
                "label":       "🚨 ATTACK" if pred else "✅ Normal",
                "confidence":  round(float(prob if pred else 1 - prob), 4),
                "attack_prob": round(float(prob), 4),
            })
        return results

    def get_metrics(self) -> dict:
        return self.meta.get("model_results", {})

    def get_best_model_name(self) -> str:
        return self.meta.get("best_model", "Unknown")


# ──────────────────────────────────────────────
# CLI usage example
# ──────────────────────────────────────────────
if __name__ == "__main__":
    detector = IntrusionDetector()

    # Example normal traffic record
    sample_normal = {
        "duration": 0, "protocol_type": "tcp", "service": "http",
        "flag": "SF", "src_bytes": 215, "dst_bytes": 45076,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
        "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 1, "srv_count": 1, "serror_rate": 0.0,
        "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0,
        "srv_diff_host_rate": 0.0, "dst_host_count": 255,
        "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.0,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0,
    }

    result = detector.predict(sample_normal)
    print(f"\nPrediction: {result[0]['label']}")
    print(f"Confidence: {result[0]['confidence']:.1%}")
    print(f"Attack probability: {result[0]['attack_prob']:.1%}")
