import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI(title="Credit Risk Scoring API")
model=joblib.load('risk_model.pkl')
#user to enter parameters
class CustomerData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

@app.get("/")
def root():
    return {"message": "Credit Risk Scoring API is running."}

@app.post("/score")
def score_customer(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    #calculating engineered features for the entered user parameters
    bill_cols=['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
    pay_amt_cols=['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
    pay_his_cols=['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']

    df['avg_utilization']=df[bill_cols].mean(axis=1) / (df['LIMIT_BAL'] + 1)
    df['missed_payments']=(df[pay_his_cols] > 0).sum(axis=1)
    bill_total=df[bill_cols].clip(lower=0).sum(axis=1) + 1
    pay_total=df[pay_amt_cols].sum(axis=1)
    df['payment_ratio']=pay_total / bill_total
    df['util_trend']=df['BILL_AMT1']-df['BILL_AMT6']
    weights=[0.35, 0.25, 0.15, 0.10, 0.08, 0.07]
    df['weighted_util']=sum(
    w*df[col] for w, col in zip(weights, bill_cols))/(df['LIMIT_BAL']+1)
    for col in pay_amt_cols:
        df[col]=np.log1p(df[col])
    df['LIMIT_BAL']=np.log1p(df['LIMIT_BAL'])
    #---OUTPUT--
    prob=model.predict_proba(df)[0][1]
    risk_label="HIGH RISK" if prob>=0.5 else "LOW RISK"

    return {
        "default_probability": round(float(prob), 4),
        "risk_label":          risk_label
    }