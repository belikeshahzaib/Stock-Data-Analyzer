import pandas as pd

def cleanBasic(df):
    if 'Close' not in df.columns:
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        else:
            raise KeyError("No 'Close' or 'Adj Close' column found")
    df = df.drop_duplicates().dropna(subset=['Close'])
    return df

def fixDateAndSort(df):
    if 'Date' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={'index': 'Date'})
        else:
            df = df.reset_index()
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    return df

def fixNumeric(df):
    numericCols = ['Open','High','Low','Close','Adj Close','Volume']
    for col in numericCols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def addBasicFeatures(df):
    df['Price_Change'] = df['Close'].diff()   
    df['Percent_Change'] = df['Close'].pct_change()
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['MA120'] = df['Close'].rolling(window=120).mean()
    return df

def cleanPipeline(df):
    df = cleanBasic(df)
    df = fixDateAndSort(df)
    df = fixNumeric(df)
    df = addBasicFeatures(df)
    df = df.round(5).dropna().reset_index(drop=True)
    return df