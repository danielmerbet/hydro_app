import hydrogr
from pathlib import Path
import pandas as pd
from numpy import sqrt, mean
import datetime
from hydrogr import InputDataHandler, ModelGr4h
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flask import Flask, render_template
from apscheduler.schedulers.background import BackgroundScheduler
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import pandas as pd
import os

app = Flask(__name__)

def fetch_data():
      # Step 1: Fetch data from the API
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 41.97,
        "longitude": 2.38,
        "hourly": "temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m,cloud_cover,surface_pressure,shortwave_radiation,et0_fao_evapotranspiration,evapotranspiration",
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Step 2: Extract and prepare data
    timestamps = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
    temperature = data["hourly"]["temperature_2m"]
    precipitation = data["hourly"]["precipitation"]
    humidity = data["hourly"]["relative_humidity_2m"]
    wind_speed = data["hourly"]["wind_speed_10m"]
    cloud_cover = data["hourly"]["cloud_cover"]
    surface_pressure = data["hourly"]["surface_pressure"]
    shortwave_radiation = data["hourly"]["shortwave_radiation"]
    potential_evaporation = data["hourly"]["et0_fao_evapotranspiration"]
    evaporation = data["hourly"]["evapotranspiration"]
    
    # Convert to pandas DataFrame
    df = pd.DataFrame({
        "Time": timestamps,
        "Temperature (°C)": temperature,
        "Precipitation (mm)": precipitation,
        "Relative Humidity (%)": humidity,
        "Wind Speed (km/h)": wind_speed,
        "Solar Radiation (W/m²)": shortwave_radiation,
        "Cloud Cover (%)": cloud_cover,
        "Surface Pressure (hPa)": surface_pressure,
        "Potential Evaporation (mm)": potential_evaporation,
        "Evaporation (mm)": evaporation
    })
    df.set_index("Time", inplace=True)
    
    return df

df = fetch_data()

df.columns =['temperature', 'precipitation', 'humidity',
       'wind', 'radiation', 'cloud',
       'pressure', 'pevaporation',
       'evapotranspiration']

inputs = InputDataHandler(ModelGr4h, df)

# Set the model :
#x1=227.75304279035723, x2=0.04490349268502891, x3=68.43833032945605, x4=8.838229659085371
parameters = {
        "X1": 227.7530,
        "X2": 0.0449,
        "X3": 68.4383,
        "X4": 8.8382
    }
model = ModelGr4h(parameters)
model.set_parameters(parameters)  # Re-define the parameters for demonstration purpose.

# Initial state :
initial_states = {
    "production_store": 0.5,
    "routing_store": 0.5,
    "uh1": None,
    "uh2": None
}
model.set_states(initial_states)

outputs = model.run(inputs.data)
outputs
print(outputs.head())
