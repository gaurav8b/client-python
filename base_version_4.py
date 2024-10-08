import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries as ts
from darts.utils.missing_values import fill_missing_values as fill_missing
from dateutil.relativedelta import relativedelta
from darts.models import NHiTSModel
import torch
torch.set_float32_matmul_precision('medium')
from darts.metrics import mape
import logging
import warnings
from pytorch_lightning.callbacks import Callback

# Suppress warnings
warnings.filterwarnings("ignore")

# Define a logger object
logger = logging.getLogger(__name__)

# Set up logging format and level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sharpe(ser):
    return ser.mean() / ser.std() if ser.std() != 0 else 0

def custom_sign(series, tolerance=1e-6):
    return np.where(np.abs(series) < tolerance, 0, np.sign(series))

class PrintEpochLoss(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        logger.info(f"Epoch {trainer.current_epoch}: train_loss={trainer.callback_metrics['train_loss']:.6f}")

class MyModeler:
    def __init__(self, data, modelfile, *, window=100, horizon=30, resample_rate='1s', n_epochs=5, modeltype=NHiTSModel, **modelparams):
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
            self.model = self.modeltype.load(self.model_file)
            logger.info(f"Loaded existing model from {self.model_file}")
        except FileNotFoundError:
            logger.info(f"No existing model found. Initializing new model.")
            self.model = self.modeltype(
                input_chunk_length=self.window,
                output_chunk_length=2 * self.horizon,
                n_epochs=self.n_epochs,
                random_state=42,
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
        
        print_epoch_loss = PrintEpochLoss()
        self.model.fit(targetts, verbose=True, pl_trainer_kwargs={"callbacks": [print_epoch_loss]})
        self.model.save(self.model_file)

    def day_predict(self, *, data=None, theday=None):
        if data is None:
            data = self.data
        if theday is not None:
            subdata = data[data.index.normalize() == pd.to_datetime(theday)]
        else:
            subdata = data
        targetts = ts.from_dataframe(subdata, freq=self.resample_rate)
        historical_fcast = self.model.historical_forecasts(
            targetts,
            retrain=False,
            verbose=True,
            num_samples=100  # Increase for more stable MAPE
        )
        fcast_mape = mape(historical_fcast, targetts)
        return historical_fcast, fcast_mape

    def fit_span(self, beg_day=None, end_day=None):
        try:
            subdata = self.data.loc[beg_day:end_day]
            for d, df in subdata.groupby(subdata.index.date):
                logger.info(f"Fitting for date: {d}")
                self.datadict[d] = df
                self.day_fit(data=df)
        except Exception as e:
            logger.error(f"Error in fit_span: {str(e)}")
            raise

    def pred_span(self, beg_day=None, end_day=None, do_pl=False, **kwargs):
        try:
            subdata = self.data.loc[beg_day:end_day]
            for d, df in subdata.groupby(subdata.index.date):
                logger.info(f"Predicting for date: {d}")
                self.datadict[d] = df
                f, m = self.day_predict(data=df)
                self.fcast[d] = f
                fdf = f.pd_dataframe()
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
        forecast_df.index = forecast_df.index.tz_localize(None)

        custom_df = forecast_df.diff(self.horizon).apply(lambda x: custom_sign(x, tolerance))
        investment_returns = actual_df.diff(self.horizon) * custom_df / self.horizon
        investment_returns.dropna(inplace=True)
        net_returns = investment_returns - (tcosts / self.horizon) * (investment_returns != 0.0)
        total_netto = net_returns.sum() / self.horizon
        total_brutto = investment_returns.sum() / self.horizon

        self.netrets[d] = total_netto
        self.grossrets[d] = total_brutto

# Data loading and model creation
data = pd.read_csv('data.csv', parse_dates=['timestamp'], index_col='timestamp')

# Ensure the index is timezone-naive
data.index = data.index.tz_localize(None)

mymod = MyModeler(data, modelfile='qqqmod_NHiTS_11.pth', window=600, horizon=200, modeltype=NHiTSModel)

# Use more workers in DataLoader
torch.set_num_threads(4)  # Adjust based on your system's capabilities

mymod.fit_span(beg_day='2023-12-01', end_day='2023-12-05')

mymod.pred_span(do_pl=True, beg_day='2024-01-01', end_day='2024-01-10', tolerance=0.0002, tcosts=0)

# Print results
for date, mape_value in mymod.mape.items():
    print(f"Date: {date}, MAPE: {mape_value}")

for date, net_return in mymod.netrets.items():
    print(f"Date: {date}, Net Return: {net_return}")