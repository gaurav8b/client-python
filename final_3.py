import logging
from darts import TimeSeries
from darts.models import NHiTSModel
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
from darts.metrics import mape
import pandas as pd

logging.basicConfig(level=logging.INFO)

class MyModeler:
    def __init__(self, model_file, modeltype, input_dim, output_dim, **model_kwargs):
        self.model_file = model_file
        self.modeltype = modeltype
        self.model = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_kwargs = model_kwargs
        self.init_model()

    def init_model(self):
        """Initialize the model and load the pre-trained weights for fine-tuning."""
        # Initialize the model without weights
        self.model = self.modeltype(
            input_chunk_length=self.input_dim,
            output_chunk_length=self.output_dim,
            **self.model_kwargs
        )
        
        # Load pre-trained weights
        self.model.load_weights(self.model_file)
        logging.info(f"Model weights loaded from {self.model_file}")

    def train_model(self, train_series, val_series, num_epochs=10, batch_size=32):
        """Train the model on the given data."""
        self.model.fit(
            series=train_series,
            val_series=val_series,
            epochs=num_epochs,
            batch_size=batch_size
        )
        logging.info("Model training completed.")

    def evaluate_model(self, test_series):
        """Evaluate the model on the test data."""
        pred_series = self.model.predict(len(test_series))
        error = mape(test_series, pred_series)
        logging.info(f"Model MAPE on test data: {error:.2f}%")
        return error

    def save_model(self, path):
        """Save the fine-tuned model weights."""
        self.model.save(path)
        logging.info(f"Fine-tuned model saved at {path}")


# Data fetching and preprocessing
def fetch_data():
    """Fetch and preprocess the time series data."""
    # Sample data for demonstration purposes
    # Replace with real Bitcoin data fetching logic
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="T")
    prices = pd.Series(1000 + 0.1 * (pd.Series(range(1000)) - 500).apply(lambda x: x**2), index=dates)

    df = pd.DataFrame({"log_price": prices})
    
    # Resample to 1-minute intervals and interpolate missing values
    df_resampled = df['log_price'].resample('1min').interpolate(method='time').ffill()
    
    # Convert to TimeSeries for Darts
    ts = TimeSeries.from_series(df_resampled)
    
    return ts

def main():
    # Fetch and split the data
    series = fetch_data()
    train_series, val_series, test_series = series.split_before([0.7, 0.85])
    
    # Normalize the data
    transformer = Scaler(MinMaxScaler())
    train_series = transformer.fit_transform(train_series)
    val_series = transformer.transform(val_series)
    test_series = transformer.transform(test_series)
    
    # Initialize the modeler
    model_file = 'path/to/model_weights.pth'  # Replace with actual path
    mymod = MyModeler(
        model_file=model_file,
        modeltype=NHiTSModel,
        input_dim=30,  # Example input_chunk_length
        output_dim=1,  # Example output_chunk_length
        model_kwargs={"n_layers": 3, "n_blocks": 1}  # Model customization
    )

    # Train the model
    mymod.train_model(train_series, val_series, num_epochs=5, batch_size=32)

    # Evaluate the model on test data
    mymod.evaluate_model(test_series)

    # Save the fine-tuned model
    mymod.save_model('path/to/fine_tuned_model.pth')

if __name__ == "__main__":
    main()
