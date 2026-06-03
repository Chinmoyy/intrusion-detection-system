"""
Network Intrusion Detection System - Training Pipeline
Uses NSL-KDD dataset to train and evaluate ML classifiers
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Column names for NSL-KDD dataset
# ──────────────────────────────────────────────
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
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

ATTACK_CATEGORIES = {
    "normal": "Normal",
    "neptune": "DoS", "back": "DoS", "land": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS",
    "apache2": "DoS", "processtable": "DoS", "udpstorm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L", "sendmail": "R2L",
    "named": "R2L", "snmpattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "worm": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "httptunnel": "U2R", "ps": "U2R",
    "sqlattack": "U2R", "xterm": "U2R",
}


def load_data(train_path: str, test_path: str = None):
    """Load and label NSL-KDD data."""
    print("[+] Loading dataset...")
    train_df = pd.read_csv(train_path, header=None, names=COLUMNS)

    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path, header=None, names=COLUMNS)
    else:
        train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Map labels to attack categories
    train_df["attack_category"] = train_df["label"].str.strip(".").map(
        lambda x: ATTACK_CATEGORIES.get(x.lower(), "Unknown")
    )
    test_df["attack_category"] = test_df["label"].str.strip(".").map(
        lambda x: ATTACK_CATEGORIES.get(x.lower(), "Unknown")
    )

    # Binary label: 0 = normal, 1 = attack
    train_df["is_attack"] = (train_df["attack_category"] != "Normal").astype(int)
    test_df["is_attack"] = (test_df["attack_category"] != "Normal").astype(int)

    print(f"    Train samples : {len(train_df):,}")
    print(f"    Test  samples : {len(test_df):,}")
    print(f"    Attack rate   : {train_df['is_attack'].mean():.1%}")
    return train_df, test_df


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Encode categoricals, scale numerics."""
    print("[+] Preprocessing...")
    feature_cols = [c for c in COLUMNS if c not in ["label", "difficulty"]]
    cat_cols = ["protocol_type", "service", "flag"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        all_vals = pd.concat([train_df[col], test_df[col]])
        le.fit(all_vals)
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        encoders[col] = le

    scaler = StandardScaler()
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])

    X_train = train_df[feature_cols].values
    y_train = train_df["is_attack"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["is_attack"].values
    y_cat   = test_df["attack_category"].values

    return X_train, y_train, X_test, y_test, y_cat, scaler, encoders, feature_cols


def train_models(X_train, y_train):
    """Train multiple classifiers and return all."""
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=20, n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
    }

    trained = {}
    for name, model in models.items():
        print(f"[+] Training {name}...")
        model.fit(X_train, y_train)
        cv = cross_val_score(model, X_train, y_train, cv=3, scoring="f1", n_jobs=-1)
        print(f"    CV F1: {cv.mean():.4f} ± {cv.std():.4f}")
        trained[name] = model

    return trained


def evaluate(models, X_test, y_test, y_cat):
    """Evaluate all models; return metrics dict."""
    results = {}
    best_f1, best_name = 0, ""

    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        except Exception:
            auc = 0.0
        cm   = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"  Accuracy : {acc:.4f}  |  F1: {f1:.4f}  |  AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))

        results[name] = {
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "roc_auc":  round(auc, 4),
            "confusion_matrix": cm,
            "report": report,
        }

        if f1 > best_f1:
            best_f1, best_name = f1, name

    # Category-level breakdown for best model
    best_pred = models[best_name].predict(X_test)
    cat_results = {}
    for cat in np.unique(y_cat):
        mask = y_cat == cat
        if mask.sum() == 0:
            continue
        cat_results[cat] = {
            "total": int(mask.sum()),
            "correct": int((best_pred[mask] == y_test[mask]).sum()),
            "accuracy": round((best_pred[mask] == y_test[mask]).mean(), 4),
        }

    return results, best_name, cat_results


def save_artifacts(models, scaler, encoders, feature_cols, results,
                   best_name, cat_results, out_dir="models"):
    """Persist all model artifacts and metrics."""
    os.makedirs(out_dir, exist_ok=True)
    for name, model in models.items():
        safe = name.lower().replace(" ", "_")
        joblib.dump(model, f"{out_dir}/{safe}.pkl")

    joblib.dump(scaler,   f"{out_dir}/scaler.pkl")
    joblib.dump(encoders, f"{out_dir}/encoders.pkl")

    meta = {
        "best_model": best_name,
        "feature_cols": feature_cols,
        "model_results": results,
        "attack_category_results": cat_results,
        "attack_categories": list(ATTACK_CATEGORIES.values()),
    }
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[✓] Artifacts saved to '{out_dir}/'")
    print(f"[✓] Best model: {best_name} (F1={results[best_name]['f1_score']})")


def main():
    train_path = "data/KDDTrain+.txt"
    test_path  = "data/KDDTest+.txt"

    if not os.path.exists(train_path):
        print("[!] Dataset not found. Run: python src/download_data.py")
        return

    train_df, test_df = load_data(train_path, test_path)
    X_train, y_train, X_test, y_test, y_cat, scaler, encoders, fcols = preprocess(train_df, test_df)
    models = train_models(X_train, y_train)
    results, best_name, cat_results = evaluate(models, X_test, y_test, y_cat)
    save_artifacts(models, scaler, encoders, fcols, results, best_name, cat_results)


if __name__ == "__main__":
    main()
