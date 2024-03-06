import requests
import pandas as pd
# import lovely_tensors as lt
# import numpy as np
# import matplotlib.pyplot as plt
# import plotly.express as px
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_percentage_error
# import tensorflow as tf
# from keras import Model
# from keras.layers import Input, Dense, Dropout
# from keras.layers import LSTM


def get_api_key(api_key_issuer: str) -> str:
    api_keys_df = pd.read_csv("/etc/R/api_keys.csv")
    filtered_df = api_keys_df[api_keys_df['issuer'] == api_key_issuer]
    key = filtered_df['keys'].values[0] if not filtered_df.empty else None
    return key


api_key = get_api_key("alphavantage_co")

# Check if API key is available before constructing the URL
if api_key:
    symbol = 'GOLD'
    interval = '5min'
    outputsize = 'full'
    dataperiod = '1000'

    url = (f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}'
           f'&interval={interval}'
           f'&outputsize={outputsize}'
           f'&dataperiod={dataperiod}'
           f'&apikey={api_key}')
    r = requests.get(url)
    data = r.json()

    if 'Time Series (5min)' in data:
        time_series_data = data['Time Series (5min)']
        df = pd.DataFrame(time_series_data).transpose()
        df.index = pd.to_datetime(df.index)
        csv_filename = 'alphavantage_data.csv'
        df.to_csv(csv_filename)

        # Print head of CSV
        print(f'Head of {csv_filename}:')
        print(df.head())
        print(df.tail())

        # Print number of rows and columns
        n_rows, n_cols = df.shape
        print(f'Number of rows: {n_rows}, Number of columns: {n_cols}')
    else:
        print('No data available for the provided symbol and interval.')
