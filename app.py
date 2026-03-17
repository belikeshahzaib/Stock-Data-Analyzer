import streamlit as st
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pandas.tseries.offsets import BDay
from datetime import date, timedelta
import plotly.express as px
import os
from clean_data import cleanPipeline
from feature_engineering import featurePipeline
import eda
import model_training
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Stock Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.header("Stock Prediction Dashboard")
trainTicker = st.sidebar.selectbox("Train Stock", ["GOOGL", "AAPL", "MSFT", "TSLA", "NVDA"])
testTicker  = st.sidebar.selectbox("Test Stock", ["AAPL", "MSFT", "TSLA", "GOOGL", "NVDA"])

startDate   = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=2*365))
endDate     = st.sidebar.date_input("End Date", value=date.today())
predictionHorizon = st.sidebar.number_input("Prediction Horizon (days)", min_value=1, max_value=10, value=5)
st.sidebar.markdown("---")

@st.cache_data
def loadStock(ticker, start, end):
    df = yf.download(ticker, start=start, end=end).reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

st.title("Stock Data Analyzer")
st.info(f"Downloading {trainTicker} (train) and {testTicker} (test)...")
trainDf = loadStock(trainTicker, startDate, endDate)
testDf  = loadStock(testTicker, startDate, endDate)
st.success("Data downloaded ✅")

trainClean = cleanPipeline(trainDf)
testClean  = cleanPipeline(testDf)
st.success("Data cleaned ✅")

st.header("Exploratory Data Analysis (EDA)")
pltStyle = {
    "figure.facecolor": "#0B1E3F", 
    "axes.facecolor": "#0B1E3F",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "gray",
    "grid.linestyle": "--",
    "text.color": "white",
    "legend.facecolor": "#0B1E3F",
    "legend.edgecolor": "white"
}
plt.rcParams.update(pltStyle)

if st.checkbox("Show statistical summary"):
    summaryData, testClean = eda.statisticalSummary(testClean)
    st.subheader("Market Pulse")
    lastClose = testClean['Close'].iloc[-1]
    meanClose = summaryData["priceStats"]["mean"]
    stdClose  = summaryData["priceStats"]["std"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Close", f"${round(meanClose,2)}", delta=f"${round(lastClose - meanClose,2)}")
    c2.metric("Price Std Dev", f"${round(stdClose,2)}")
    c3.metric("Sharpe Ratio", f"{summaryData['sharpeRatio']:.3f}")
    c4.metric("Volatility", f"{summaryData['volatility']*100:.2f}%")

    st.markdown("---")
    st.subheader("Risk Radar")
    worstLoss = summaryData['returnStats']['min']
    bestGain  = summaryData['returnStats']['max']
    var95     = summaryData['var95']

    r1, r2, r3 = st.columns(3)
    r1.metric("Worst Daily Loss", f"{worstLoss*100:.2f}%")
    r2.metric("Best Daily Gain", f"{bestGain*100:.2f}%")
    r3.metric("VaR (95%)", f"{var95*100:.2f}%")

    st.markdown("---")
    st.subheader("Return Distribution")
    fig, ax = plt.subplots(figsize=(9,4))
    returnsData = testClean["Percent_Change"].dropna()
    sns.histplot(returnsData[returnsData>=0], bins=60, kde=True, color="green", ax=ax)
    sns.histplot(returnsData[returnsData<0], bins=60, kde=True, color="red", ax=ax)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Market Relationships")
    corrDf = summaryData["correlation"]
    def colorCorr(val):
        if val > 0: return "color: green; font-weight: bold"
        elif val < 0: return "color: red; font-weight: bold"
        return ""
    st.dataframe(corrDf.style.applymap(colorCorr).format("{:.2f}"))

if st.checkbox("Show visuals"):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(testClean['Date'], testClean['Close'], color='cyan')
    ax.set_title("Closing Price Trend")
    st.pyplot(fig)

trainFeatures = featurePipeline(trainClean, nDays=predictionHorizon)
testFeatures  = featurePipeline(testClean, nDays=predictionHorizon)
st.success("Feature engineering complete ✅")

st.info(f"Training model on {trainTicker} and predicting for {testTicker}...")
predictionsDf = model_training.trainModels(
    trainFeatures,
    testFeatures,
    nFuture=predictionHorizon,
    saveModels=False,
    showFeatureImportance=False
)
st.success("Prediction complete ✅")

lastDate = testClean['Date'].iloc[-1]
predCols = [f"Close_plus_{i}_pred" for i in range(1, predictionHorizon+1)]
predValues = predictionsDf[predCols].iloc[-1].values
futureDates = [lastDate + BDay(i) for i in range(1, predictionHorizon+1)]

predDf = pd.DataFrame({'Date': futureDates, 'Close': predValues})
histDf = testClean[['Date', 'Close']].copy()
histDf['Type'] = 'Historical'
predDfCopy = predDf.copy()
predDfCopy['Type'] = 'Predicted'

plotDf = pd.concat([histDf, predDfCopy], ignore_index=True)
fig = px.line(plotDf, x='Date', y='Close', color='Type', markers=True, title=f"{testTicker} Historical vs Predicted")
st.plotly_chart(fig, use_container_width=True)

st.subheader(f"Next {predictionHorizon} Days Predicted Close Prices")
predDfDisplay = predDf.copy()
predDfDisplay['Date'] = predDfDisplay['Date'].dt.strftime('%Y-%m-%d')
st.table(predDfDisplay.set_index('Date'))