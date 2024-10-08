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

import torch
from darts.models.forecasting.torch_forecasting_model import PLForecastingModule

# Set torch precision
torch.set_float32_matmul_precision('medium')

# Register PLForecastingModule as a safe class for torch loading
#torch.serialization.register_safe_class(PLForecastingModule)

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
            # Try loading the model without weights_only first
            self.model = self.modeltype.load(path=self.model_file)
            logger.info(f"Model loaded from {self.model_file}")
        except TypeError as e:
            logger.error(f"TypeError occurred: {e}")
            logger.error(traceback.format_exc())
            # If TypeError occurs, try loading without weights_only
            try:
                self.model = self.modeltype.load(path=self.model_file)
                logger.info(f"Model loaded from {self.model_file} without weights_only parameter")
            except FileNotFoundError:
                logger.warning(f"Model file {self.model_file} not found. Initializing a new model.")
                self.model = self.modeltype(
                    input_chunk_length=self.window,
                    output_chunk_length=2 * self.horizon,
                    random_state=42,
                    n_epochs=self.n_epochs,
                    **self.modelparams
                )
        except FileNotFoundError:
            logger.warning(f"Model file {self.model_file} not found. Initializing a new model.")
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
        if subdata.empty:
            logger.warning(f"No data available for the specified day: {theday}")
            return
        targetts = ts.from_dataframe(subdata, freq=self.resample_rate)
        self.model.fit(targetts, verbose=True)
        self.model.save(self.model_file)
        logger.info(f"Model fitted and saved to {self.model_file}")

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
            logger.info(f"Prediction completed for {theday} with MAPE: {fcast_mape}")

            # Store forecast for the day, ensuring consistent datetime format
            if historical_fcast is not None:
                day_key = pd.to_datetime(theday)  # Ensure theday is in datetime format
                self.fcastdf[day_key] = historical_fcast.pd_dataframe()
                logger.info(f"Forecast data stored for {day_key}")

            return historical_fcast, fcast_mape
        except ValueError as e:
            logger.error(f"Error during prediction: {str(e)}")
            return None, None

    @logger.catch
    def fit_span(self, beg_day=None, end_day=None):
        subdata = self.data.loc[beg_day:end_day]
        for d, df in subdata.groupby(subdata.index.date):
            logger.info(f"Fitting model for date: {d}")
            self.datadict[d] = df
            self.day_fit(data=df)

    @logger.catch
    def pred_span(self, beg_day=None, end_day=None, do_pl=False, **kwargs):
        subdata = self.data.loc[beg_day:end_day]
        for d, df in subdata.groupby(subdata.index.date):
            logger.info(f"Predicting for date: {d}")
            self.datadict[d] = df
            f, m = self.day_predict(data=df)
            self.fcast[d] = f
            if f is not None:
                fdf = f.pd_dataframe()
                self.fcastdf[pd.to_datetime(d)] = fdf  # Ensure consistent datetime format
                self.mape[d] = m
                if do_pl:
                    self.do_pl(pd.to_datetime(d), **kwargs)

    def do_pl(self, d, tolerance, tcosts):
        # Check if forecast data exists for the day
        if d not in self.fcastdf:
            logger.error(f"No forecast data available for {d}. Skipping P&L calculation.")
            return

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

        logger.info(f"P&L calculated for {d}: Netto={total_netto}, Brutto={total_brutto}")

def fetch_crypto_data(coin_id, days=365, vs_currency='usd'):
    cg = CoinGeckoAPI()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Fix argument order: Positional arguments before keyword arguments
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

    # Resample to 1-minute intervals with interpolation and forward-fill
    df_resampled = df['log_price'].resample('1min').interpolate(method='time').ffill()

    return pd.DataFrame(df_resampled)


# Example usage
if __name__ == "__main__":
    coin_id = "bitcoin"
    days = 365  # Fetch data for the last 365 days

    logger.info(f"Fetching {coin_id} data for the past {days} days.")
    crypto_df = fetch_crypto_data(coin_id, days)

    # Calculate the start and end dates for training and testing
    end_date = crypto_df.index[-1]
    train_end_date = end_date - timedelta(days=30)  # Use the last 30 days for testing
    train_start_date = crypto_df.index[0]

    logger.info(f"Data range: {train_start_date} to {end_date}")
    logger.info(f"Total data points: {len(crypto_df)}")

    mymod = MyModeler(
        crypto_df,
        modelfile='btc_NHiTS.pth',
        window=200,
        horizon=60,
        resample_rate='1min',
        modeltype=NHiTSModel
    )

    # Fit the model for the training span
    logger.info("Fitting model on training data.")
    train_start_date = datetime(2024, 10, 1)
    train_end_date = datetime(2024, 10, 3)
    mymod.fit_span(beg_day=train_start_date, end_day=train_end_date - timedelta(minutes=1))

    # Select the last available day for prediction
    specific_day = train_end_date.date()
    specific_day = datetime(2024, 10, 4)
    logger.info(f"Selected prediction day: {specific_day}")

    # Make predictions for the specific day
    historical_fcast, fcast_mape = mymod.day_predict(theday=specific_day)

    if historical_fcast is not None and fcast_mape is not None:
        # Print the MAPE for the prediction
        print(f"Forecast MAPE for {specific_day}: {fcast_mape}")

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
    specific_day_data = crypto_df[crypto_df.index.normalize() == pd.to_datetime(specific_day)]
    print(f"Data range: {crypto_df.index[0]} to {crypto_df.index[-1]}")
    print(f"Total data points: {len(crypto_df)}")
    print(f"Data points for prediction day ({specific_day}): {len(specific_day_data)}")
