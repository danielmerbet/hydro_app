import requests
import pandas as pd
from datetime import datetime, timedelta

# Set the base URL and initial parameters
url = "http://aca-web.gencat.cat/sdim2/apirest/data/AFORAMENT-EST/CALC001144"
#start_date = datetime(2024, 11, 1)
#end_date = datetime(2024, 11, 11)
start_date = datetime.now().date() - timedelta(days=89)
end_date = datetime.now().date() - timedelta(days=1)
all_data = pd.DataFrame()

# Loop through the dates
#while start_date <= end_date:
for day_start in pd.date_range(start_date, end_date):

    # TIMES FROM 00:00:00 to 12:00:00
    params = {
        "limit": 200,
        "from": day_start.strftime("%d/%m/%YT%H:%M:%S"),
        "to": (day_start + timedelta(minutes=60 * 12)- timedelta(minutes=1)).strftime(
            "%d/%m/%YT%H:%M:%S"
        ),  
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        observations = data.get("observations", [])
        # Convert to DataFrame
        df = pd.DataFrame(observations)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d/%m/%YT%H:%M:%S")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Resample data to hourly and calculate mean
        df.set_index("timestamp", inplace=True)
        df = df["value"]

    else:
        print(f"Error fetching data: {response.status_code}")

    # TIMES FROM 12:00:00 to 24:00:00
    params = {
        "limit": 200,
        "from": (day_start + timedelta(minutes=60 * 12)).strftime("%d/%m/%YT%H:%M:%S"),
        "to": (day_start + timedelta(minutes=60 * 24) - timedelta(minutes=1)).strftime("%d/%m/%YT%H:%M:%S"),
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        observations = data.get("observations", [])
        # Convert to DataFrame
        df_2 = pd.DataFrame(observations)
        df_2["timestamp"] = pd.to_datetime(df_2["timestamp"], format="%d/%m/%YT%H:%M:%S")
        df_2["value"] = pd.to_numeric(df_2["value"], errors="coerce")

        # Resample data to hourly and calculate mean
        df_2.set_index("timestamp", inplace=True)
        df_2 = df_2["value"]
        #hourly_df = df.resample("h").mean()
        #all_data = pd.concat([all_data, hourly_df])  # Append data to all_data list
    else:
        print(f"Error fetching data: {response.status_code}")
    
    df12 = pd.concat([df, df_2]) 
    hourly_df = df12.resample("h").mean()
    all_data = pd.concat([all_data, hourly_df])  # Append data to all_data list
    

all_data.to_csv("/home/dmercado/Documents/intoDBP/hydro_app/hydro_cal/aca_data_"+str(datetime.today().date())+".csv") 

