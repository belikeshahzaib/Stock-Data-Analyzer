import pandas as pd
import numpy as np

def statisticalSummary(df):
    df = df.copy()
    requiredCols = ['Close', 'Volume', 'Percent_Change', 'MA7', 'MA30']
    for col in requiredCols:
        if col not in df.columns:
            df[col] = np.nan

    priceStats = df['Close'].describe()
    returnsData = df['Percent_Change'].dropna()
    returnStats = returnsData.describe()
    volatilityVal = returnsData.std()
    var95Val = np.percentile(returnsData, 5)
    sharpeRatioVal = (returnsData.mean() / returnsData.std() if returnsData.std() != 0 else np.nan)
    volumeStats = df['Volume'].describe()
    corrMatrix = df[requiredCols].corr()

    summary = {
        "priceStats": priceStats,
        "returnStats": returnStats,
        "volatility": volatilityVal,
        "var95": var95Val,
        "sharpeRatio": sharpeRatioVal,
        "volumeStats": volumeStats,
        "correlation": corrMatrix
    }
    return summary, df