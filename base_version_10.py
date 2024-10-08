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
import os

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
        self.output_dir = "model_output"
        os.makedirs(self.output_dir, exist_ok=True)

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
        
        self.model.fit(targetts, verbose=True)
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
            
            # Save results after processing all dates
            self.save_results()
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

        # Ensure we're storing scalar values
        self.netrets[d] = float(total_netto)
        self.grossrets[d] = float(total_brutto)

    def save_results(self):
        """
        Save all important data to CSV files.
        """
        logger.info("Saving results to CSV files...")

        # Save forecasts
        forecasts_df = pd.concat([df.assign(date=date) for date, df in self.fcastdf.items()])
        forecasts_df.to_csv(os.path.join(self.output_dir, "forecasts.csv"), index=True)

        # Save MAPE scores
        mape_df = pd.DataFrame.from_dict(self.mape, orient='index', columns=['MAPE'])
        mape_df.index.name = 'date'
        mape_df.to_csv(os.path.join(self.output_dir, "mape_scores.csv"))

        # Save net and gross returns
        net_returns = list(self.netrets.values())
        gross_returns = list(self.grossrets.values())

        # Ensure net_returns and gross_returns contain only scalar values
        net_returns = [float(x) if isinstance(x, (int, float)) else np.nan for x in net_returns]
        gross_returns = [float(x) if isinstance(x, (int, float)) else np.nan for x in gross_returns]

        returns_df = pd.DataFrame({
            'date': self.netrets.keys(),
            'net_returns': net_returns,
            'gross_returns': gross_returns
        })
        returns_df.set_index('date', inplace=True)
        returns_df.to_csv(os.path.join(self.output_dir, "returns.csv"))

        # Save daily performance metrics
        daily_performance = []
        for date in self.datadict.keys():
            actual = self.datadict[date]
            forecast = self.fcastdf.get(date)
            if forecast is not None:
                logger.info(f"Columns in 'actual' for date {date}: {actual.columns}")

                # Check if 'close' column exists (case-insensitive)
                close_column = next((col for col in actual.columns if col.lower() == 'close'), None)
                
                if close_column:
                    daily_sharpe = sharpe(actual[close_column])
                else:
                    logger.warning(f"'close' column is missing in the DataFrame for date {date}. Setting Sharpe ratio to NaN.")
                    daily_sharpe = np.nan

                daily_mape = self.mape.get(date, np.nan)
                daily_net_return = self.netrets.get(date, np.nan)
                daily_gross_return = self.grossrets.get(date, np.nan)
                
                daily_performance.append({
                    'date': date,
                    'sharpe_ratio': daily_sharpe,
                    'mape': daily_mape,
                    'net_return': daily_net_return,
                    'gross_return': daily_gross_return
                })
        
        daily_performance_df = pd.DataFrame(daily_performance)
        daily_performance_df.set_index('date', inplace=True)
        daily_performance_df.to_csv(os.path.join(self.output_dir, "daily_performance.csv"))

        # Save overall model performance metrics
        overall_performance = {
            'mean_mape': np.mean(list(self.mape.values())),
            'median_mape': np.median(list(self.mape.values())),
            'total_net_return': sum(net_returns),
            'total_gross_return': sum(gross_returns),
            'sharpe_ratio': sharpe(pd.Series(net_returns)),
            'win_rate': np.mean([r > 0 for r in net_returns]),
            'num_trades': len(net_returns)
        }
        pd.DataFrame([overall_performance]).to_csv(os.path.join(self.output_dir, "overall_performance.csv"), index=False)

        logger.info("Results saved successfully.")

# Data loading and model creation
data = pd.read_csv('data.csv', parse_dates=['timestamp'], index_col='timestamp')

# Ensure the index is timezone-naive
data.index = data.index.tz_localize(None)

mymod = MyModeler(data, modelfile='qqqmod_NHiTS_11.pth', window=600, horizon=200, modeltype=NHiTSModel)

# Use more workers in DataLoader
torch.set_num_threads(4)  # Adjust based on your system's capabilities

# Fit the model
mymod.fit_span(beg_day='2023-12-01', end_day='2023-12-02')

# Make predictions and calculate profit/loss
mymod.pred_span(do_pl=True, beg_day='2024-01-01', end_day='2024-01-02', tolerance=0.0002, tcosts=0)

# Results will be automatically saved by the save_results method called within pred_span

# Print summary of results
print("\nSummary of Results:")
print("-" * 20)
with open(os.path.join(mymod.output_dir, "overall_performance.csv"), 'r') as f:
    print(f.read())

print("\nDetailed results have been saved in the 'model_output' directory.")