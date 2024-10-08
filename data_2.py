import requests
import pandas as pd

# Replace with your Kaiko API key
api_key = "YOUR_API_KEY"

# Define the endpoint and parameters
endpoint = "https://api.kaiko.com/v1/data/trades"
params = {
    "asset_id": "BTCUSD",
    "start_date": "2024-09-01",
    "end_date": "2024-09-30",
    "frequency": "1s"
}

# Make the API request
headers = {"X-API-Token": api_key}
response = requests.get(endpoint, params=params, headers=headers)

# Parse the response
data = response.json()
df = pd.DataFrame(data["data"])

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# Print the DataFrame
print(df)