# Credit Risk Scoring Engine
> End-to-end ML pipeline for real-time credit default prediction — data ingestion through REST API deployment.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-orange)](https://lightgbm.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-containerised-blue)](https://docker.com)
[![AUC](https://img.shields.io/badge/AUC-0.7777-brightgreen)]()

---

## Problem Statement
Banks require automated systems to assess credit default risk before loan disbursement. Manual review at scale is infeasible. This system ingests raw customer financial data and returns a calibrated default probability in real time, enabling risk-based decision making at inference speed.

---

## Dataset
**UCI Credit Card Default Dataset** — 30,000 Taiwanese cardholders, April–September 2005.
- 24 raw features: credit limit, demographics, 6-month repayment history, bill statements, payment amounts
- Target: binary default indicator for the following month
- Class distribution: 77.9% non-default / 22.1% default — handled via `class_weight='balanced'` and `scale_pos_weight`

---

## Test Screenshots

## API Test Results

![High Risk Test](/testing/test1.png)
**HIGH RISK — 0.8123** | 30k limit customer, 3 months consecutively late (PAY_0=3, PAY_2=3), 97% utilization, paying only 500/month against a 29,000 bill. Deteriorating payment history across all 6 months confirmed by model at 81.2% default probability.

![Low Risk Test](/testing/test2.png)
**LOW RISK — 0.1482** | 500k limit, PAY=-2 across all 6 months (no consumption / paid in full), paying exact bill amount every month, 1% utilization. Model correctly identifies zero delinquency signals — 14.8% default probability.

![High Risk Test 2](/testing/test3.png)
**HIGH RISK — 0.8043** | 50k limit at 94% utilization, actively 2 months late on most recent payments (PAY_0=2, PAY_2=2), debt growing from 35k to 47k over 6 months, payment ratio under 4%. Clear deterioration trajectory confirmed at 80.4% default probability.

![Low Risk Test 2](/testing/test4.png)
**LOW RISK — 0.1039** | 200k limit at 12% utilization, paying full bill amount each month, only two months of minimum payments (PAY_3=0, PAY_4=0) but never late. Stable debt with no delinquency — model correctly assigns 10.4% default probability.

## System Architecture
```
dataset.csv
    │
    ▼
Data Cleaning & Preprocessing
    │  • Collapsed undocumented EDUCATION codes (0,5,6) → category 4 (others)
    │  • Collapsed undocumented MARRIAGE code (0) → category 3 (others)
    │  • Retained PAY column value -2 (no consumption) as semantically distinct from -1 (paid duly)
    ▼
Feature Engineering
    │  • 5 domain-driven engineered features (see below)
    │  • Log transforms on right-skewed monetary columns
    ▼
Model Training & Selection
    │  • 4 models trained and compared: Decision Tree, Random Forest, XGBoost, LightGBM
    │  • LightGBM selected — best AUC, F1, lowest false negatives
    │  • Hyperparameter tuning via RandomizedSearchCV (20 iterations, 5-fold stratified CV)
    ▼
risk_model.pkl
    │
    ▼
FastAPI REST API  →  POST /score  →  { default_probability, risk_label }
    │
    ▼
Docker Container (production-ready deployment)
```

---

## Feature Engineering
Domain knowledge from credit risk literature drove all feature construction. Raw monetary columns were not transformed until after ratio features were computed to preserve interpretability.

| Feature | Formula | Rationale |
|---|---|---|
| `avg_utilization` | `mean(BILL_AMT1–6) / (LIMIT_BAL + 1)` | Measures financial overextension relative to credit ceiling |
| `missed_payments` | `count(PAY cols > 0)` | Counts months with active payment delays — strongest default predictor |
| `payment_ratio` | `sum(PAY_AMT1–6) / sum(BILL_AMT1–6)` | Fraction of total outstanding balance actually repaid |
| `util_trend` | `BILL_AMT1 - BILL_AMT6` | Direction of debt trajectory — positive = worsening |
| `weighted_util` | `Σ(wᵢ × BILL_AMTᵢ) / LIMIT_BAL`, weights=[0.35,0.25,0.15,0.10,0.08,0.07] | Recency-weighted utilization — penalises recent high balances more than historical ones |

**Log transforms applied to:** `LIMIT_BAL`, `PAY_AMT1–6` — all exhibit strong right skew (confirmed via boxplot analysis pre-transformation).

**Features intentionally retained as raw:** `BILL_AMT1–6`, `PAY_0–6` — individual monthly columns preserve trend, recency, and volatility signals that aggregates lose.

---

## Model Comparison
All models trained on identical 80/20 stratified split (`random_state=42`). Evaluated on held-out test set (6,000 records). Class imbalance handled at algorithm level — avoids synthetic sample leakage risk from SMOTE.

| Model | AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| **LightGBM** | **0.7758** | **0.613** | **0.464** | **0.528** |
| Decision Tree | 0.7620 | 0.638 | 0.422 | 0.508 |
| XGBoost | 0.7539 | 0.564 | 0.458 | 0.506 |
| Random Forest | 0.7509 | 0.341 | 0.644 | 0.446 |

**Evaluation metric rationale:** AUC-ROC used as primary metric — threshold-independent, measures ranking quality across the full probability spectrum. Accuracy excluded — a trivial classifier predicting no default achieves 77.9% accuracy on this dataset.

**Model selection rationale:** LightGBM leads on AUC (fairest cross-model comparison) and F1. Decision Tree achieves marginally higher Recall (0.638 vs 0.613) but at the cost of 224 additional false positives and 1.4% lower AUC — reckless flagging, not superior discrimination. Random Forest's Recall of 0.341 (misses 66% of defaulters) is disqualifying for credit risk deployment regardless of its Precision.

**After hyperparameter tuning (RandomizedSearchCV, 20 iterations, 5-fold CV):**
```
Tuned LightGBM Test AUC: 0.7777
              precision    recall  f1-score
  No Default       0.88      0.79      0.83
     Default       0.46      0.62      0.53
```

---

## Known Model Limitations
**Minimum payment trap:** Customers consistently paying minimum amounts (PAY=0) against maxed-out balances score as lower risk than they are. `missed_payments` does not fire for PAY=0 (not technically late), yet a 2% payment ratio against a 98% utilization card represents genuine financial stress.

**Post-delinquency recovery pattern:** Customers who were severely late (PAY≥3) 4–6 months ago but made a large lump-sum rescue payment and recently reduced card usage can fool the model. `util_trend` reads the balance drop as improvement; `weighted_util` partially mitigates this but does not fully capture the fragility of recent recovery.

*Production mitigation: credit bureau integration, longer time windows (24+ months), and transaction-level behavioural data would address both limitations.*

---

## REST API
Built with FastAPI. Feature engineering is replicated identically inside the endpoint — no preprocessing pipeline external to the API. Column alignment to training feature order enforced via `model.feature_names_in_` before inference.

### Run Locally
```bash
pip install -r requirements.txt
uvicorn api:app --reload
# Interactive docs at http://localhost:8000/docs
```

### Run with Docker
```bash
docker build -t risk-api .
docker run -p 8000:8000 risk-api
```

### Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/score` | Score a single customer |

### Sample Request
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 30000, "SEX": 1, "EDUCATION": 3, "MARRIAGE": 1, "AGE": 26,
    "PAY_0": 3, "PAY_2": 3, "PAY_3": 2, "PAY_4": 2, "PAY_5": 1, "PAY_6": 1,
    "BILL_AMT1": 29000, "BILL_AMT2": 28000, "BILL_AMT3": 27000,
    "BILL_AMT4": 26000, "BILL_AMT5": 25000, "BILL_AMT6": 24000,
    "PAY_AMT1": 500, "PAY_AMT2": 500, "PAY_AMT3": 500,
    "PAY_AMT4": 500, "PAY_AMT5": 500, "PAY_AMT6": 500
  }'
```

### Sample Response
```json
{
  "default_probability": 0.8123,
  "risk_label": "HIGH RISK"
}
```

### Live Test Results
| Customer Profile | Probability | Label |
|---|---|---|
| Maxed out, 3 months late, paying 500/month against 29k bill | 0.8123 | HIGH RISK |
| 500k limit, 1% utilization, paying full balance every month | 0.1482 | LOW RISK |
| 50k limit, 94% utilization, 2 months late, growing debt | 0.8043 | HIGH RISK |
| 200k limit, 12% utilization, paying full balance, stable | 0.1039 | LOW RISK |

---

## Repository Structure
```
credit-risk-engine/
├── risk_scoring.ipynb     # Full pipeline: EDA → feature engineering → training → evaluation
├── api.py                 # FastAPI application with feature engineering at inference
├── Dockerfile             # Container definition — python:3.10-slim base
├── requirements.txt       # Dependencies
├── risk_model.pkl         # Serialised tuned LightGBM (joblib)
└── dataset.csv            # UCI Credit Card Default Dataset
```

---

## Tech Stack
| Layer | Technology |
|---|---|
| Data processing | pandas, NumPy |
| Visualisation | matplotlib, seaborn |
| ML | scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Hyperparameter tuning | RandomizedSearchCV |
| API | FastAPI, Pydantic, uvicorn |
| Serialisation | joblib |
| Containerisation | Docker |
