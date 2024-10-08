import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries as ts
from darts.models import NHiTSModel
from darts.metrics import mape
from loguru import logger
import torch
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
import traceback
import sys
from datetime import date


def fetch_crypto_data(coin_id, days=365, vs_currency='usd'):
    cg = CoinGeckoAPI()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"Fetching data for {coin_id} from {start_date} to {end_date}")

    data = cg.get_coin_market_chart_range_by_id(
        coin_id, 
        vs_currency, 
        start_date.timestamp(), 
        end_date.timestamp()
    )

    print(f"Retrieved {len(data['prices'])} data points.")

    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Print the raw data with proper timestamp
    print(df)

    return df

# Assuming `fetch_crypto_data` is defined in a script named `data_console.py`

# Call the function with arguments
data = fetch_crypto_data("bitcoin", days=1)  # Replace "bitcoin" with desired coin ID

# Print the returned DataFrame (containing raw price data)
print(data)