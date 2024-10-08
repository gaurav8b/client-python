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

torch.set_float32_matmul_precision('medium')

def sharpe(ser):
    try:
        return ser.mean() / ser.std()
    except:
        return -1

def custom_sign(series, tolerance=1e-6):
    return np.where(np.abs(series) < tolerance, 0, np.sign(series))

class MyModeler:
    def __init__(self, data, modelfile, *, window=100, horizon=30, resample_rate='1min', n_epochs=5, modeltype=NHiTSModel, **modelparams):
        self.data = data
        self.model_file = modelfile
        self.window = window
        self.horizon = horizon
        self.n_epochs = n_epochs
        self.fcast = {}
        self.fcastdf = {}
        self.mape = {}
        self.datadict = {}
        self.netrets = {}
        self.grossrets = {}
        self.resample_rate = resample_rate
        self.modeltype = modeltype
        self.modelparams = modelparams
        self.init_model()

    def init_model(self):
        try:
            self.model = self.modeltype.load(path=self.model_file)
        except FileNotFoundError:
            self.model = self.modeltype(
                input_chunk_length=self.window,
                output_chunk_length=2 * self.horizon,
                random_state=42,
                n_epochs=self.n_epochs,
                **self.modelparams
            )

    def day_fit(self, *, data=None, theday=None):
        if data is None:
            data = self.data
        if theday is not None:
            subdata = data[data.index.normalize() == pd.to_datetime(theday)]
        else:
            subdata = data
        targetts = ts.from_dataframe(subdata, freq=self.resample_rate)
        self.model.fit(targetts, verbose=True)
        self.model.save(self.model_file)

    def day_predict(self, *, data=None, theday=None):
        if data is None:
            data = self.data
        if theday is not None:
            subdata = data[data.index.normalize() == pd.to_datetime(theday)]
        else:
            subdata = data
        
        if subdata.empty:
            logger.warning(f"No data available for the specified day: {theday}")
            return None, None

        targetts = ts.from_dataframe(subdata, freq=self.resample_rate)
        
        if len(targetts) <= self.window:
            logger.warning(f"Insufficient data for prediction. Required: >{self.window}, Available: {len(targetts)}")
            return None, None

        try:
            historical_fcast = self.model.historical_forecasts(
                targetts,
                retrain=False,
                verbose=True,
                start=self.window,
                forecast_horizon=self.horizon
            )
            fcast_mape = mape(historical_fcast, targetts)
            return historical_fcast, fcast_mape
        except ValueError as e:
            logger.error(f"Error during prediction: {str(e)}")
            return None, None

    @logger.catch
    def fit_span(self, beg_day=None, end_day=None):
        subdata = self.data.loc[beg_day:end_day]
        for d, df in subdata.groupby(subdata.index.date):
            logger.info(d)
            self.datadict[d] = df
            self.day_fit(data=df)

    @logger.catch
    def pred_span(self, beg_day=None, end_day=None, do_pl=False, **kwargs):
        subdata = self.data.loc[beg_day:end_day]
        for d, df in subdata.groupby(subdata.index.date):
            logger.info(d)
            self.datadict[d] = df
            f, m = self.day_predict(data=df)
            self.fcast[d] = f
            fdf = f.pd_dataframe()
            self.fcastdf[d] = fdf
            self.mape[d] = m
            if do_pl:
                self.do_pl(d, **kwargs)

    def do_pl(self, d, tolerance, tcosts):
        forecast_df = self.fcastdf[d]
        actual_df = self.datadict[d]
        custom_df = forecast_df.diff(self.horizon).apply(lambda x: custom_sign(x, tolerance))
        investment_returns = actual_df.diff(self.horizon) * custom_df / self.horizon
        investment_returns.dropna(inplace=True)
        net_returns = investment_returns - (tcosts / self.horizon) * (investment_returns != 0.0)
        total_netto = net_returns.sum() / self.horizon
        total_brutto = investment_returns.sum() / self.horizon
        self.netrets[d] = total_netto
        self.grossrets[d] = total_brutto

def fetch_crypto_data(coin_id, days=365, vs_currency='usd'):
    cg = CoinGeckoAPI()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = cg.get_coin_market_chart_range_by_id(
        id=coin_id,
        vs_currency=vs_currency,
        from_timestamp=start_date.timestamp(),
        to_timestamp=end_date.timestamp()
    )
    
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['log_price'] = np.log(df['price'])
    
    df_resampled = df['log_price'].resample('1min').interpolate(method='time')
    
    return pd.DataFrame(df_resampled)

# Example usage
if __name__ == "__main__":
    coin_id = "bitcoin"
    days = 365  # Fetch data for the last 365 days

    crypto_df = fetch_crypto_data(coin_id, days)

    # Calculate the start and end dates for training and testing
    end_date = crypto_df.index[-1]
    train_end_date = end_date - timedelta(days=30)  # Use the last 30 days for testing
    train_start_date = crypto_df.index[0]

    mymod = MyModeler(crypto_df, modelfile='btc_NHiTS.pth', window=200, horizon=60, 
                      resample_rate='1min', modeltype=NHiTSModel)

    # Fit the model for a specific day
    specific_day = train_start_date
    mymod.day_fit(theday=specific_day)

    # Make predictions for a specific day
    specific_day = train_end_date + timedelta(minutes=1)
    historical_fcast, fcast_mape = mymod.day_predict(theday=specific_day)

    if historical_fcast is not None and fcast_mape is not None:
        # Print the MAPE for the prediction
        print(f"Forecast MAPE: {fcast_mape}")

        # Calculate P&L for the specific day
        mymod.do_pl(specific_day, tolerance=0.001, tcosts=0.00005)

        # Analyze results
        if mymod.netrets:
            netdf = pd.DataFrame(mymod.netrets).T.sort_index()
            print(netdf.describe())

            # Plot cumulative returns
            plt.figure(figsize=(12, 6))
            netdf['log_price'].cumsum().plot()
            plt.title('Cumulative Returns')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Log Returns')
            plt.show()

            # Calculate Sharpe ratio
            print(f"Sharpe Ratio: {sharpe(netdf['log_price'])}")
        else:
            print("No P&L data available. Make sure do_pl() has been called.")
    else:
        print("Prediction failed. Please check the logs for more information.")

    # Print some information about the data
    print(f"Data range: {crypto_df.index[0]} to {crypto_df.index[-1]}")
    print(f"Total data points: {len(crypto_df)}")
    print(f"Data points for prediction day: {len(crypto_df[crypto_df.index.normalize() == pd.to_datetime(specific_day)])}")