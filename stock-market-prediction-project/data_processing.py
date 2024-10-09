import yfinance as yf
import os
import pandas as pd

DATA_PATH = "nvda_data.json"

def download_data():
    """
    Download the historical stock data for NVIDIA (NVDA).
    If the data file exists locally, it reads from the file.
    If not, it downloads and saves it for future use.
    """
    if os.path.exists(DATA_PATH):
        # Read from file if we've already downloaded the data.
        with open(DATA_PATH) as f:
            nvda_hist = pd.read_json(DATA_PATH)
    else:
        nvda = yf.Ticker("NVDA")
        nvda_hist = nvda.history(period="max")
        # Save file to json in case we need it later.
        nvda_hist.to_json(DATA_PATH)
    
    return nvda_hist

def preprocess_data(stock_data):
    """
    Prepares the stock data by creating a target column
    and shifting stock prices for prediction.
    """
    data = stock_data[["Close"]]
    data = data.rename(columns={'Close': 'Actual_Close'})
    
    # Create Target: 1 if stock price goes up, 0 if it goes down
    data["Target"] = stock_data.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]
    
    # Shift stock prices forward by one day
    stock_data_prev = stock_data.shift(1)
    predictors = ["Close", "Volume", "Open", "High", "Low"]
    
    # Join the shifted data with the original data
    data = data.join(stock_data_prev[predictors]).iloc[1:]
    return data