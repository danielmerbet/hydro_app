
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



obs_data = pd.read_csv("/home/dmercado/Documents/intoDBP/hydro_app/hydro_cal/aca_data_2017-2024.csv", index_col=0, parse_dates=True)
# historical precipitation data starts here
start_date = '2021-03-23 01:00:00'
obs_data = obs_data.loc[start_date:]
start_date = obs_data.index[0]
end_date = obs_data.index[-2] #las day is not complete
obs_data = obs_data.loc[:end_date]
#reproduce values to have every hour (asusme is the same very 5 min)
stream_data = obs_data.resample('h').mean()
end_date = stream_data.index[-1]

lat = 41.97
lon = 2.38
area = 1502660000.000 #m²

def fetch_historical():
      # Step 1: Fetch data from the API
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.date().strftime('%Y-%m-%d'),    # Start date
        "end_date": end_date.date().strftime('%Y-%m-%d'),      # End date
        #"hourly": "temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m,cloud_cover,surface_pressure,shortwave_radiation,et0_fao_evapotranspiration,evapotranspiration",
        "hourly": "temperature_2m,precipitation,evapotranspiration",
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Step 2: Extract and prepare data
    timestamps = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
    temperature = data["hourly"]["temperature_2m"]
    precipitation = data["hourly"]["precipitation"]
    #humidity = data["hourly"]["relative_humidity_2m"]
    #wind_speed = data["hourly"]["wind_speed_10m"]
    #cloud_cover = data["hourly"]["cloud_cover"]
    #surface_pressure = data["hourly"]["surface_pressure"]
    #shortwave_radiation = data["hourly"]["shortwave_radiation"]
    #potential_evaporation = data["hourly"]["et0_fao_evapotranspiration"]
    evaporation = data["hourly"]["evapotranspiration"]
    
    # Convert to pandas DataFrame
    df = pd.DataFrame({
        "Time": timestamps,
        "Temperature (°C)": temperature,
        "Precipitation (mm)": precipitation,
        #"Relative Humidity (%)": humidity,
        #"Wind Speed (km/h)": wind_speed,
        #"Solar Radiation (W/m²)": shortwave_radiation,
        #"Cloud Cover (%)": cloud_cover,
        #"Surface Pressure (hPa)": surface_pressure,
        #"Potential Evaporation (mm)": potential_evaporation,
        "Evaporation (mm)": evaporation
    })
    df.set_index("Time", inplace=True)
    
    return df

clim_data = fetch_historical()

#clim_stream_data.columns =['temperature', 'precipitation', 'humidity',
#       'wind', 'radiation', 'cloud',
#       'pressure', 'pevaporation',
#       'evapotranspiration']

clim_data.columns =['temperature', 'precipitation', 'evapotranspiration']
#first row is NAN for precipitation
clim_data = clim_data.iloc[1:]

clim_stream_data = clim_data
clim_stream_data['flow'] = stream_data
clim_stream_data["flow"].isnull().values.any()
clim_stream_data['flow'].isna().sum()
clim_stream_data["flow"] = clim_stream_data["flow"].ffill()
clim_stream_data['flow_mm'] = (clim_stream_data["flow"]*(60*60)*1000)/area
clim_stream_data["flow"].isnull().values.any()

#Interface to calibrate with RMSE
class SpotpySetup(object):
    """
    Interface to use the model with spotpy
    """

    def __init__(self, data):
        self.data = data
        self.model_inputs = InputDataHandler(ModelGr4h, self.data)
        self.params = [spotpy.parameter.Uniform('x1', 0.01, 300.0),
                       spotpy.parameter.Uniform('x2', 0.0, 5.0),
                       spotpy.parameter.Uniform('x3', 0.01, 300.0),
                       spotpy.parameter.Uniform('x4', 0.5, 10.0),
                       ]

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        simulations = self._run(x1=vector[0], x2=vector[1], x3=vector[2], x4=vector[3])
        return simulations

    def evaluation(self):
        return self.data['flow_mm'].values

    def objectivefunction(self, simulation, evaluation):
        res = - sqrt(mean((simulation - evaluation) ** 2.0))
        return res
    
    def _run(self, x1, x2, x3, x4):
        parameters = {"X1": x1, "X2": x2, "X3": x3, "X4": x4}
        model = ModelGr4h(parameters)
        outputs = model.run(self.model_inputs.data)
        return outputs['flow'].values

#Calibration
#mask = (clim_stream_data['date'] >= start_date) & (clim_stream_data['date'] <= end_date)
#calibration_data = clim_stream_data.loc[mask]

calibration_data = clim_stream_data
calibration_data = pd.concat([clim_stream_data] * 10, ignore_index=True)
date_range = pd.date_range(end=end_date, periods=len(calibration_data), freq='h')
#date_range = calibration_data.index
calibration_data['date'] = date_range
calibration_data.set_index('date', inplace=True)
calibration_data['date'] = date_range
#check
calibration_data.isnull().values.any()

spotpy_setup = SpotpySetup(calibration_data)

#sampler = spotpy.algorithms.mc(spotpy_setup, dbformat='ram')
sampler = spotpy.algorithms.mcmc(spotpy_setup, dbformat='ram')
#sampler = spotpy.algorithms.mle(spotpy_setup, dbformat='ram')
#sampler = spotpy.algorithms.lhs(spotpy_setup, dbformat='ram')
#sampler = spotpy.algorithms.sceua(spotpy_setup, dbformat='ram')
#sampler = spotpy.algorithms.demcz(spotpy_setup, dbformat='ram')
#sampler = spotpy.algorithms.sa(spotpy_setup, dbformat='ram')
#sampler = spotpy.algorithms.rope(spotpy_setup, dbformat='ram')

sampler.sample(500)
results=sampler.getdata() 
best_parameters = spotpy.analyser.get_best_parameterset(results)
print(best_parameters)

#Validation
parameters = list(best_parameters[0])
parameters = {"X1": parameters[0], "X2": parameters[1], "X3": parameters[2], "X4": parameters[3]}

validation_data = calibration_data

model = ModelGr4h(parameters)
outputs = model.run(validation_data)

rmse = sqrt(mean((outputs['flow'] - validation_data['flow_mm'].values) ** 2.0))
print(rmse)
correlation = outputs['flow'].corr(validation_data['flow_mm'])
print(f"Correlation: {correlation}")

r_squared = correlation**2
print(f"R²: {r_squared}")

fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.3, 0.7],
    shared_xaxes=True
)

fig.add_trace(
    go.Scatter(x=clim_stream_data.index, y=clim_stream_data['precipitation'], name="Precipitation"),
    row=1, col=1 
)
fig.update_yaxes(autorange="reversed", row=1, col=1)
fig.add_trace(
    go.Scatter(x=clim_stream_data.index, y=clim_stream_data['flow'], name="Observed flow"),
    row=2, col=1 
)
fig.add_trace(
    go.Scatter(x=outputs.index, y=(outputs['flow']*area)/(1000*60*60), name="Calculated flow"),
    row=2, col=1 
)
fig.update_yaxes(range=[0, 30], row=2, col=1)

fig.update_layout(template='plotly_dark', title_text='GR4H')
fig.show()


# Remove the first year used to warm up the model :
#filtered_input = validation_data[validation_data.index >= datetime.datetime(1990, 1, 1, 0, 0)]
#filtered_output = outputs[outputs.index >= datetime.datetime(1990, 1, 1, 0, 0)]

#rmse = sqrt(mean((filtered_output['flow'] - filtered_input['flow_mm'].values) ** 2.0))


