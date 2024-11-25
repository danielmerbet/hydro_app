import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Define helper functions
def series_to_supervised(data, n_in=1, n_out=1, dropnan=False):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # Forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return pd.DataFrame(agg.astype('float32'))

# Load and preprocess the new dataset
def prepare_new_data(file_path, hours_of_history, hours_to_predict):
    data = pd.read_csv(file_path)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)

    # Scale data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['precipitation', 'et', 'discharge']])
    data[['precipitation', 'et', 'discharge']] = data_scaled

    # Create supervised data
    data_sequence = series_to_supervised(data_scaled, hours_of_history, hours_to_predict)
    data_sequence.dropna(inplace=True)

    x_rainfall = data_sequence.iloc[:, 0::3].values.reshape(-1, hours_of_history + hours_to_predict, 1)
    discharge = data_sequence.iloc[:, 2::3].values.reshape(-1, hours_of_history + hours_to_predict, 1)
    x_discharge = discharge[:, :hours_of_history, :]
    x_et = data_sequence.iloc[:, 3 * hours_of_history + 1].values.reshape(-1, 1)  # Current ET value

    y_test = discharge[:, hours_of_history:, :]
    return [x_et, x_discharge, x_rainfall], y_test, scaler, data.index[hours_of_history + hours_to_predict - 1:]
  
# define custome loss function (you can use the simple 'mse' as well)
def nseloss(y_true, y_pred):
  return K.sum((y_pred-y_true)**2)/K.sum((y_true-K.mean(y_true))**2)

# Load the model
h5_path ='/home/dmercado/Documents/intoDBP/hydro_app/HydroLSTM/1_model1_model.keras'
model = load_model(h5_path, custom_objects={'nseloss': nseloss})
#name_id = "h_3months_2024_11_24_data"
name_id = "1_data"
# Parameters
hours_of_history = 72
hours_to_predict = 24

# Test new data
file_path = "/home/dmercado/Documents/intoDBP/hydro_app/HydroLSTM/data/"+name_id+".csv"  # Update with your test data path
x_test, y_test_scaled, scaler, dates = prepare_new_data(file_path, hours_of_history, hours_to_predict)

# Make predictions
y_pred_scaled = model.predict(x_test)

# Rescale predictions to the original range
q_max = np.max(scaler.data_max_[2])  # Discharge max
q_min = np.min(scaler.data_min_[2])  # Discharge min
y_pred = y_pred_scaled * (q_max - q_min) + q_min
y_test = y_test_scaled * (q_max - q_min) + q_min

# Save predictions
#pd.DataFrame(y_pred).to_csv("predictions.csv", index=False)
print("Predictions saved to 'predictions.csv'")

# Statistical functions
def nse(y_true, y_pred):
    return 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def kge(y_true, y_pred):
    kge_r = np.corrcoef(y_true, y_pred)[1][0]
    kge_a = np.std(y_pred) / np.std(y_true)
    kge_b = np.mean(y_pred) / np.mean(y_true)
    return 1 - np.sqrt((kge_r - 1) ** 2 + (kge_a - 1) ** 2 + (kge_b - 1) ** 2)
  
#plot prediction
#select which data to plot first:
y_test_p =y_test[:,0,0]
y_pred_p = y_pred[:,0]

#now plot
plt.figure(figsize=(15, 6))
plt.plot(
    dates[:len(y_test_p)],  # Match length of the predictions
    y_test_p,         # Extract mean discharge
    color='blue',
    alpha=0.9,
    label='Observed'
)
plt.plot(
    dates[:len(y_pred_p)],  # Match length of the predictions
    y_pred_p,         # Extract mean discharge
    color='red',
    alpha=0.9,
    label='Predicted'
)

# Calculate statistics
nse_value = nse(y_test_p, y_pred_p)
kge_value = kge(y_test_p, y_pred_p)
r_value = np.corrcoef(y_test_p, y_pred_p)[1][0]
r2_value = r2_score(y_test_p, y_pred_p)

print(f"NSE: {nse_value:.3f}, KGE: {kge_value:.3f}, r: {r_value:.3f}, R²: {r2_value:.3f}")

plt.xlabel("Time Step")
plt.ylabel("Discharge")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Add statistics to the plot
plt.figtext(0.15, 0.70, f"NSE: {nse_value:.3f}", fontsize=10)
plt.figtext(0.15, 0.65, f"KGE: {kge_value:.3f}", fontsize=10)
plt.figtext(0.15, 0.6, f"r: {r_value:.3f}", fontsize=10)
plt.figtext(0.15, 0.55, f"R²: {r2_value:.3f}", fontsize=10)

plt.savefig("/home/dmercado/Documents/intoDBP/hydro_app/HydroLSTM/data/"+name_id+".pdf", dpi=100)


#plot all 24 prediction for each value
#plot figure
plt.figure(figsize=(15, 6))
# Plot observed data
for i in range(y_test.shape[1]):  # Iterate through 24 prediction horizons
    plt.plot(
        dates[:len(y_test)],  # Match length of the predictions
        y_test[:, i,0],      # Extract the i-th prediction horizon
        color='blue',
        alpha=0.1,            # High transparency for individual predictions
        label='Observed' if i == 0 else ""  # Add legend for the first line only
    )
for i in range(y_pred.shape[1]):  # Iterate through 24 prediction horizons
    plt.plot(
        dates[:len(y_pred)],  # Match length of the predictions
        y_pred[:, i],      # Extract the i-th prediction horizon
        color='orange',
        alpha=0.1,            # High transparency for individual predictions
        label='Predicted' if i == 0 else ""  # Add legend for the first line only
    )

# Calculate ensemble mean
y_pred_mean = y_pred.mean(axis=1)  # Mean across the 24 prediction horizons

# Plot the ensemble mean
#plt.plot(
#    dates[:len(y_pred_mean)],  # Match length of the predictions
#    y_pred_mean,         # Extract mean discharge
#    color='red',
#    alpha=0.9,
#    label='Ensemble'
#)

plt.savefig("/home/dmercado/Documents/intoDBP/hydro_app/HydroLSTM/data/"+name_id+"_all.pdf", dpi=100)

#delete all from here?

plt.figure(figsize=(10, 10))
plt.plot(dates, y_test.flatten(), label='Observed', color='blue', alpha=0.7)
plt.plot(dates, y_pred.flatten(), label='Predicted', color='red', alpha=0.7)
plt.title("Observed vs Predicted Discharge")
plt.xlabel("Time Step")
plt.ylabel("Discharge")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Add statistics to the plot
plt.figtext(0.15, 0., f"NSE: {nse_value:.3f}", fontsize=10)
plt.figtext(0.15, 0.65, f"KGE: {kge_value:.3f}", fontsize=10)
plt.figtext(0.15, 0.6, f"r: {r_value:.3f}", fontsize=10)
plt.figtext(0.15, 0.55, f"R²: {r2_value:.3f}", fontsize=10)

plt.savefig("/home/dmercado/Documents/intoDBP/hydro_app/HydroLSTM/data/"+name_id+".pdf", dpi=100)

plt.show()
