#globals().clear()

import hydrogr
from pathlib import Path
import pandas as pd
from numpy import sqrt, mean
import datetime
from hydrogr import InputDataHandler, ModelGr4h
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import os
from numpy import sqrt, mean
import spotpy
import plotly.graph_objects as go
from plotly.subplots import make_subplots


homedir ="/home/dmercado/Documents/intoDBP/hydro_app/"
os.chdir(homedir)
obs_data = pd.read_csv("hydro_cal/aca_data_2017-2024.csv", index_col=0, parse_dates=True)

# historical precipitation data starts here
start_date = '2021-03-23 01:00:00'
obs_data = obs_data.loc[start_date:]
start_date = obs_data.index[0]
end_date = obs_data.index[-2] #las day is not complete
obs_data = obs_data.loc[:end_date]
#reproduce values to have every hour (asusme is the same very 5 min)
stream_data = obs_data.resample('h').mean()

test = obs_data.resample('d').mean()

lat_1 = 41.97
lon_1 = 2.38
lat_2 = 42.19
lon_2 = 2.19
area = 1502660000.000 #m²
id = "2"

def fetch_historical(lat, lon):
      # Step 1: Fetch data from the API
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.date().strftime('%Y-%m-%d'),    # Start date
        "end_date": end_date.date().strftime('%Y-%m-%d'),      # End date
        "hourly": "precipitation", #,evapotranspiration
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Step 2: Extract and prepare data
    timestamps = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
    precipitation = data["hourly"]["precipitation"]
    #et = data["hourly"]["evapotranspiration"]
    
    # Convert to pandas DataFrame
    df = pd.DataFrame({
        "Time": timestamps,
        "Precipitation (mm)": precipitation#,
        #"Evaporation (mm)": et
    })
    df.set_index("Time", inplace=True)
    
    return df

clim_data_1 = fetch_historical(lat_1, lon_1)
clim_data_1 = clim_data_1.iloc[1:] #first row is NAN for precipitation
clim_data_1.columns =['precipitation']
clim_data_2 = fetch_historical(lat_2, lon_2)
clim_data_2 = clim_data_2.iloc[1:] #first row is NAN for precipitation
clim_data_2.columns =['precipitation']

clim_data = (clim_data_1["precipitation"] + clim_data_2["precipitation"])/2
clim_data = clim_data.to_frame()

time = clim_data.index
plt.figure(figsize=(10, 6))
plt.plot(time, clim_data["precipitation"].values, label='mean Precipitation', color='blue')
plt.plot(time, clim_data_1["precipitation"].values, label='DF1 Precipitation', color='green')
plt.plot(time, clim_data_2["precipitation"].values, label='DF2 Precipitation', color='orange')
plt.show()

clim_data.columns =['precipitation']


#clim_data = clim_data.iloc[1:]

# Extract accumulated ET for the month
accumulated_et = pd.read_csv("hydro_cal/Evaporation_ERA5_1940-2024.csv")
# Convert the index of precipitation_et_df to datetime if it's not already
clim_data.index = pd.to_datetime(clim_data.index)
# Extract the month from the Time index
clim_data.loc[:,'month'] = clim_data.index.month
# Map the monthly mean values from the monthly_mean DataFrame to the 'et' column
accumulated_et_dict = dict(zip(accumulated_et['month'], accumulated_et['e']))
clim_data.loc[:,'et'] = clim_data['month'].map(accumulated_et_dict)
clim_data = clim_data.drop(columns=['month'])

# Calculate accumulated ET for the month
#accumulated_et = clim_data['et'].resample('ME').sum()
# Add accumulated ET as a new column
#accumulated_et.index = accumulated_et.index.to_period('M')
#clim_data['accumulated_et'] = clim_data.index.to_period('M').map(accumulated_et)
# Drop the original 'et' column
#clim_data = clim_data.drop(columns=['et'])

#set final names
clim_data.columns =['precipitation', 'et']

if clim_data.shape[0] == stream_data.shape[0]:
    print("The two DataFrames have the same nuber of rows:", clim_data.shape[0])
else:
    print("The two DataFrames do not have the same shape.")
    print("Shape of df1:", df_historical.shape)
    print("Shape of df2:", obs_data.shape)
    
# add discharge data
clim_data['discharge'] = stream_data

#save
clim_data.index.names = ['datetime']
clim_data.to_csv("HydroLSTM/data/"+ id + "_data.csv")
