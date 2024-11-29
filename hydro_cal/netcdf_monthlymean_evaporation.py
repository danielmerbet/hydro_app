import xarray as xr
import pandas as pd

# Load the NetCDF file
netcdf_file = "/home/dmercado/Documents/intoDBP/hydro_app/hydro_cal/Evaporation_ERA5_1940-2024.nc"  # Replace with your file path
output_csv = "/home/dmercado/Documents/intoDBP/hydro_app/hydro_cal/Evaporation_ERA5_1940-2024.csv"    # Replace with desired output file path

# Open the NetCDF file
ds = xr.open_dataset(netcdf_file)

# Convert the data to a Pandas DataFrame
df = ds.to_dataframe().reset_index()

# Group by 'date' and calculate mean for 'e'
df_mean = df.groupby('date')['e'].mean().reset_index()
# negative to positive, convention from ERA5
df_mean["e"] = df_mean["e"]*(-1)*1000*30; df_mean

# Convert 'date' to datetime format
df_mean['date'] = pd.to_datetime(df_mean['date'], format='%Y%m%d')

# Extract the month and group by it
df_mean['month'] = df_mean['date'].dt.month
monthly_mean = df_mean.groupby('month')['e'].mean().reset_index()

# Save DataFrame to a CSV file
monthly_mean.to_csv(output_csv, index=False)

print(f"Data from {netcdf_file} has been converted to {output_csv}")
