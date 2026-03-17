import yfinance as yf
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

def downloadStock(ticker, startDate, endDate, savePath):
    folder = os.path.dirname(savePath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    df = yf.download(ticker, start=startDate, end=endDate)
    df.reset_index(inplace=True)
    df.to_csv(savePath, index=False)

    print(f"Saved raw data for {ticker} to: {os.path.abspath(savePath)}")
    return df

def loadCsv(path):
    df = pd.read_csv(path)

    if 'Date' not in df.columns:
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    numericCols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in numericCols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.dropna(subset=['Close'])

    return df