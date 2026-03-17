import pandas as pd
import numpy as np

def addReturns(df):
    df["Return"] = df["Close"].pct_change()
    df["LogReturn"] = np.log1p(df["Return"])
    return df

def addLags(df, lagList=[1, 2, 3, 5]):
    for lag in lagList:
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
    return df

def addMovingAverages(df, windowList=[5, 10, 20, 50]):
    for w in windowList:
        df[f"Close_ma_{w}"] = df["Close"].rolling(w).mean()
    return df

def addVolatility(df, windowList=[5, 10, 20]):
    for w in windowList:
        df[f"Close_vol_{w}"] = df["Close"].rolling(w).std()
    return df

def preparePredictionData(df, nDays):
    df = df.copy()
    for i in range(1, nDays + 1):
        df[f"Close_plus_{i}"] = df["Close"].shift(-i)
    return df

def featurePipeline(df, nDays):
    df = df.copy()
    df = addReturns(df)
    df = addLags(df)
    df = addMovingAverages(df)
    df = addVolatility(df)
    df = preparePredictionData(df, nDays)
    df = df.loc[:, df.nunique() > 1].dropna().reset_index(drop=True)
    return df