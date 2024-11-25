
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
#import spotpy
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots

homedir ="/home/dmercado/Documents/intoDBP/hydro_app/"
os.chdir(homedir)

obs_data = pd.read_csv("/home/dmercado/Documents/intoDBP/hydro_app/hydro_cal/aca_data_2024-11-25.csv", index_col=0, parse_dates=True)
# historical precipitation data starts here
#start_date = '2021-03-23 01:00:00'
start_date = obs_data.index[0]; start_date
#set start day to account for full months 
start_date = '2024-09-01 00:00:00'
start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
obs_data = obs_data.loc[start_date:]
#start_date = obs_data.index[0]
end_date = obs_data.index[-1]; end_date #las day is not complete
#obs_data = obs_data.loc[:end_date]
#reproduce values to have every hour (asusme is the same very 5 min)
stream_data = obs_data.resample('h').mean()
#end_date = stream_data.index[-1]

lat = 41.97
lon = 2.38
area = 1502660000.000 #m²
#if h: historical, if f: forecast
id = "h_3months_" +  str(end_date.year) + "_" + str(end_date.month) + "_" + str(end_date.day)

def fetch_historical():
      # Step 1: Fetch data from the API
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.date().strftime('%Y-%m-%d'),    # Start date
        "end_date": end_date.date().strftime('%Y-%m-%d'),      # End date
        "hourly": "precipitation,evapotranspiration",
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Step 2: Extract and prepare data
    timestamps = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
    precipitation = data["hourly"]["precipitation"]
    et = data["hourly"]["evapotranspiration"]
    
    # Convert to pandas DataFrame
    df = pd.DataFrame({
        "Time": timestamps,
        "Precipitation (mm)": precipitation,
        "Evaporation (mm)": et
    })
    df.set_index("Time", inplace=True)
    
    return df

clim_data = fetch_historical()

clim_data.columns =['precipitation', 'et']

# Calculate accumulated ET for the month
accumulated_et = clim_data['et'].resample('ME').sum()
accumulated_et.index = accumulated_et.index.to_period('M')
clim_data['accumulated_et'] = clim_data.index.to_period('M').map(accumulated_et)
# Drop the original 'et' column
clim_data = clim_data.drop(columns=['et'])
#set final names
clim_data.columns =['precipitation', 'et']

#check
clim_data.shape[0] == stream_data.shape[0]

# add discharge data
clim_data['discharge'] = stream_data

#save
clim_data.index.names = ['datetime']
clim_data.to_csv("HydroLSTM/data/"+ id + "_data.csv")

