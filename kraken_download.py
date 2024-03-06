import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# Function to log errors to a file
def log_error(error_message):
    with open('error_log.txt', 'a') as log_file:
        log_file.write(f'{datetime.now()} - {error_message}\n')

try:
    # Calculate the timestamp for 2 seconds ago
    since_time = int((datetime.now() - timedelta(minutes=1)).timestamp())

    # Make the API request with the calculated timestamp
    resp = requests.get(
        'https://api.kraken.com/0/public/OHLC',
        params={'pair': 'PAXGUSD', 'interval': 1, 'since': since_time}
    )
    resp.raise_for_status()  # Raise an exception for bad responses (4xx or 5xx)

    # Convert the JSON response to a DataFrame with specified column names
    columns = ['timestamp', 'price', 'open', 'high', 'low', 'close', 'vol', 'trade_n']
    df = pd.DataFrame(resp.json()['result']['PAXGUSD'], columns=columns)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Convert columns to specified data types
    float_columns = ['price', 'open', 'high', 'low', 'close', 'vol']
    ufloat_columns = ['trade_n']

    df[float_columns] = df[float_columns].astype('float32')
    df[ufloat_columns] = df[ufloat_columns].astype('float16')

    # Check if the CSV file exists
    csv_file_path = 'PAXGUSD.csv'
    if not os.path.exists(csv_file_path):
        # If the file doesn't exist, create it with column names
        df.to_csv(csv_file_path, index=False)
    else:
        # If the file exists, read the existing data and append new data
        existing_data = pd.read_csv(csv_file_path)
        updated_data = pd.concat([df, existing_data], ignore_index=True)
        updated_data.to_csv(csv_file_path, index=False)

except requests.exceptions.RequestException as e:
    # Handle network or API request errors
    error_message = f'Request error: {str(e)}'
    print(error_message)
    log_error(error_message)

except Exception as e:
    # Handle other unexpected errors
    error_message = f'Unexpected error: {str(e)}'
    print(error_message)
    log_error(error_message)
