
#globals().clear()

#from pathlib import Path
import pandas as pd
from numpy import sqrt, mean
import datetime
#from hydrogr import InputDataHandler, ModelGr4h
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import os
from numpy import sqrt, mean
import numpy as np
#import spotpy
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots

#get_obs_hydro_data.py must be run first


homedir ="/home/dmercado/Documents/intoDBP/hydro_app/"
os.chdir(homedir)

obs_data = pd.read_csv("/home/dmercado/Documents/intoDBP/hydro_app/hydro_cal/aca_data_"+str(datetime.today().date())+".csv", index_col=0, parse_dates=True)
start_date = obs_data.index[0]; start_date

end_date = obs_data.index[-1]; end_date #las day is not complete
stream_data = obs_data.resample('h').mean()
#end_date = stream_data.index[-1]

lat_1 = 41.97
lon_1 = 2.38
lat_2 = 42.19
lon_2 = 2.19
area = 1502660000.000 #m²
#if h: historical, if f: forecast
id = "f_" +  str(datetime.today().year) + "_" + str(datetime.today().month) + "_" + str(datetime.today().day)

#fetch historical data, for approx. the last 3 months (limited by ACA data)
def fetch_historical(start_date, end_date, lat, lon):
      # Step 1: Fetch data from the API
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.date().strftime('%Y-%m-%d'),    # Start date
        "end_date": end_date.date().strftime('%Y-%m-%d'),      # End date
        "hourly": "precipitation",
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Step 2: Extract and prepare data
    timestamps = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
    precipitation = data["hourly"]["precipitation"]
    
    # Convert to pandas DataFrame
    df = pd.DataFrame({
        "Time": timestamps,
        "Precipitation (mm)": precipitation
    })
    df.set_index("Time", inplace=True)
    
    return df

clim_data = fetch_historical(start_date, end_date, lat_1, lon_1)

clim_data.columns =['precipitation']

# Calculate accumulated ET for the month
accumulated_et = pd.read_csv("hydro_cal/Evaporation_ERA5_1940-2024.csv")
# Convert the index of precipitation_et_df to datetime if it's not already
clim_data.index = pd.to_datetime(clim_data.index)
# Extract the month from the Time index
clim_data['month'] = clim_data.index.month
# Map the monthly mean values from the monthly_mean DataFrame to the 'et' column
accumulated_et_dict = dict(zip(accumulated_et['month'], accumulated_et['e']))
clim_data['et'] = clim_data['month'].map(accumulated_et_dict)
clim_data = clim_data.drop(columns=['month'])

#check
clim_data.shape[0] == stream_data.shape[0]
# add discharge data
clim_data['discharge'] = stream_data

#fetch forecast data
def fetch_data():
      # Step 1: Fetch data from the API
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation",
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Step 2: Extract and prepare data
    timestamps = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
    precipitation = data["hourly"]["precipitation"]
    
    # Convert to pandas DataFrame
    df = pd.DataFrame({
        "Time": timestamps,
        "Precipitation (mm)": precipitation
    })
    df.set_index("Time", inplace=True)
    
    # Filter the DataFrame to show only data from the current hour onwards
    #current_time = datetime.now()
    #df = df[df.index >= current_time]
    
    return df

fore_data = fetch_data()
fore_data.columns =['precipitation']
# Convert the index of precipitation_et_df to datetime if it's not already
fore_data.index = pd.to_datetime(fore_data.index)
# Extract the month from the Time index
fore_data['month'] = fore_data.index.month
fore_data['et'] = fore_data['month'].map(accumulated_et_dict)
fore_data = fore_data.drop(columns=['month'])
fore_data['discharge'] = np.nan

#merge, historical climate data and forecast data
all_data = pd.concat([clim_data, fore_data], axis=0)


all_data.index.names = ['datetime']
all_data.to_csv("HydroLSTM/data/"+ id + "_data.csv")

