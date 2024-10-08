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

   
    data = cg.get_coin_market_chart_range_by_id(
        coin_id, 
        vs_currency, 
        start_date.timestamp(), 
        end_date.timestamp()
    )

    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['log_price'] = np.log(df['price'])

    df_resampled = df['log_price'].resample('1min').interpolate(method='time').ffill()

    return pd.DataFrame(df_resampled)