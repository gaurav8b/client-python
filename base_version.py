import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries as ts
from darts.utils.missing_values import fill_missing_values as fill_missing
from dateutil.relativedelta import relativedelta
from darts.models import ExponentialSmoothing, ARIMA, AutoARIMA, Prophet, Theta, FFT, NBEATSModel, NHiTSModel, TSMixerModel
import torch
torch.set_float32_matmul_precision('medium')
from darts.metrics import mape
import optuna
from tqdm import tqdm

# Import logging module
import logging

# Define a logger object
logger = logging.getLogger(__name__)

# Set up logging format and level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sharpe(ser):
    try:
        return ser.mean()/ser.std()
    except:
        return -1

def custom_sign(series, tolerance=1e-6):
    return np.where(np.abs(series) < tolerance, 0, np.sign(series))

class MyModeler:
    def __init__(self, data, modelfile, *, initialize_model = True, window = 100, horizon = 30, resample_rate = '1s', n_epochs = 5,  modeltype = NHiTSModel,  **modelparams):
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
        self.resample_rate= resample_rate
        self.modeltype = modeltype
        self.modelparams = modelparams

        self.init_model()

    def init_model(self):
        try:
            # Attempt to load existing model
            model_path = self.model_file
            self.model = self.modeltype.load(path=model_path)
        except FileNotFoundError:
            # If model does not exist, initialize a new one
            self.model = self.modeltype(
                input_chunk_length=self.window,
                output_chunk_length=2 * self.horizon,
                random_state=42,
                n_epochs=self.n_epochs,
                num_workers=15,  # Add num_workers if needed
                **self.modelparams,
            )

    def day_fit(self, *, data = None, theday = None):
        if data is None:
            data = self.data
        if theday is not None:
            subdata = data[data.index.normalize() == pd.to_datetime(theday)]
        else:
            subdata = data
        targetts  = ts.from_dataframe(subdata, freq = self.resample_rate)

        self.model.fit(targetts, verbose=True)
        self.model.save(self.model_file)

    def day_predict(self, *, data = None, theday = None):
        if data is None:
            data = self.data
        if theday is not None:
            subdata = data[data.index.normalize() == pd.to_datetime(theday)]
        else:
            subdata = data
        targetts  = ts.from_dataframe(subdata, freq = self.resample_rate)
        historical_fcast = self.model.historical_forecasts(
             targetts,
             retrain=False, verbose=True,
            )
        fcast_mape = mape(historical_fcast, targetts)
        return historical_fcast, fcast_mape

    def fit_span(self, beg_day = None, end_day = None):
        try:
            if beg_day is None:
                if end_day is None:
                    subdata = self.data
                else:
                    subdata = self.data.loc[:end_day]
            else:
                if end_day is None:
                    subdata = self.data.loc[beg_day:]
                else:
                    subdata = self.data.loc[beg_day:end_day]

            for d, df in subdata.groupby(subdata.index.date):
                logger.info(d)
                self.datadict[d] = subdata
                self.day_fit(data = df)
        except Exception as e:
            logger.error(f"Error in fit_span: {str(e)}")
            raise

    def pred_span(self, beg_day = None, end_day = None, do_pl = False, **kwargs):
        try:
            if beg_day is None:
                if end_day is None:
                    subdata = self.data
                else:
                    subdata = self.data.loc[:end_day]
            else:
                if end_day is None:
                    subdata = self.data.loc[beg_day:]
                else:
                    subdata = self.data.loc[beg_day:end_day]

            for d, df in subdata.groupby(subdata.index.date):
                logger.info(d)
                self.datadict[d] = subdata
                f, m = self.day_predict(data = df)
                self.fcast[d] = f
                fdf = f.univariate_component(0).pd_dataframe()
                self.fcastdf[d] = fdf
                self.mape[d] = m
                if do_pl:
                    self.do_pl(d, **kwargs)
        except Exception as e:
            logger.error(f"Error in pred_span: {str(e)}")
            raise

    def do_pl(self, d, tolerance, tcosts):
        forecast_df = self.fcastdf[d]
        actual_df = self.datadict[d]
        # Assuming `actual_df` and `forecast_df` are your DataFrame variables
        actual_df.index = actual_df.index.tz_localize(None)  # Remove timezone, making it timezone-naive
        forecast_df.index = forecast_df.index.tz_localize(None)  # Remove timezone, making it timezone-naive

        custom_df = forecast_df.diff(self.horizon).apply(lambda x: custom_sign(x, tolerance))
        investment_returns = actual_df.diff(self.horizon) * custom_df/self.horizon
        investment_returns.dropna(inplace=True)
        net_returns = investment_returns - (tcosts / self.horizon) * (investment_returns != 0.0)
        total_netto = net_returns.sum()/self.horizon
        total_brutto = investment_returns.sum()/self.horizon

        self.netrets[d] = total_netto
        self.grossrets[d] = total_brutto

# Data loading and model creation
data = pd.read_csv('data.csv')

# Set the 'timestamp' column as the index and convert to datetime with timezone
data.set_index('timestamp', inplace=True)
data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S%z')

mymod = MyModeler(data, modelfile = 'qqqmod_NHiTS_11.pth', window = 600, horizon=200, modeltype = NHiTSModel, save_checkpoints = True, force_reset = True)

mymod.fit_span(beg_day = '2023-12-01',end_day = '2023-12-31')

mymod.pred_span(do_pl=True, beg_day = '2024-01-01', end_day='2024-01-10',tolerance = 0.0002, tcosts = 0)