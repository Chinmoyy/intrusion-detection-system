# 🛡️ Network Intrusion Detection System (NIDS)

A machine learning system that classifies network traffic as **normal or malicious** using the NSL-KDD dataset. Trains multiple classifiers, evaluates them, and serves results through an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

| Component | Description |
|-----------|-------------|
| **Dataset** | NSL-KDD — 41 network features, 5 attack categories |
| **Models** | Random Forest, Gradient Boosting, Logistic Regression |
| **Task** | Binary classification (Normal vs Attack) + category labeling |
| **Dashboard** | Streamlit app with live classifier, confusion matrix, model comparison |

### Attack Categories Detected
- **DoS** — Denial of Service (Neptune, Smurf, Back…)
- **Probe** — Port scanning & network mapping (NMAP, IPSweep…)
- **R2L** — Remote to Local unauthorized access (FTP Write, Guess Password…)
- **U2R** — User to Root privilege escalation (Buffer Overflow, Rootkit…)

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/<your-username>/intrusion-detection-system.git
cd intrusion-detection-system
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python src/download_data.py
```

### 3. Train Models
```bash
python src/train.py
```
Training takes ~2–5 minutes. Artifacts saved to `models/`.

### 4. Launch Dashboard
```bash
streamlit run dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501)

---

## 📁 Project Structure

```
intrusion-detection-system/
├── src/
│   ├── train.py          # ML pipeline: load → preprocess → train → evaluate
│   ├── predict.py        # IntrusionDetector class for inference
│   └── download_data.py  # Download NSL-KDD dataset
├── dashboard/
│   └── app.py            # Streamlit dashboard
├── models/               # Saved .pkl models + metrics.json (generated)
├── data/                 # Raw dataset files (generated)
├── requirements.txt
└── README.md
```

---

## 📊 Model Performance (Typical Results)

| Model | Accuracy | F1 Score | ROC AUC |
|-------|----------|----------|---------|
| Random Forest | ~99.5% | ~0.997 | ~0.999 |
| Gradient Boosting | ~99.2% | ~0.994 | ~0.998 |
| Logistic Regression | ~97.1% | ~0.974 | ~0.991 |

> Results vary slightly based on train/test split. Random Forest typically performs best.

---

## 🧠 Key ML Concepts Used

- **Feature Engineering** — Label encoding for categorical features, Standard scaling for numerics
- **Cross-Validation** — 3-fold CV to detect overfitting
- **Ensemble Methods** — Random Forest (bagging), Gradient Boosting (boosting)
- **Evaluation** — Confusion matrix, F1 score, ROC-AUC
- **Imbalanced Classes** — Attack categories (R2L, U2R) are naturally rare

---

## 🔍 Live Classifier

The dashboard includes a **real-time traffic inspector** where you can input:
- Protocol type (TCP, UDP, ICMP)
- Service (HTTP, FTP, SSH, DNS, etc.)
- Connection flags, byte counts, login status

And get an instant **Normal / Attack** prediction with confidence score.

---

## 📚 Dataset Reference

**NSL-KDD** — Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009).
*A Detailed Analysis of the KDD CUP 99 Data Set.*
IEEE Symposium on Computational Intelligence for Security and Defense Applications.
[Dataset](https://www.unb.ca/cic/datasets/nsl.html)

---

## 📄 License

MIT License — free to use for educational and portfolio purposes.

---

*Built as part of a CIS portfolio at Baruch College, CUNY.*
# intrusion-detection-system
