# Contained here:

1. app_hydro.py: Run using ModelGr4H
2. get_obs_hydro_data.py: download and save data every 5 minutes of the last 3 month for discharge from the ACA database
3. hydro_cal.py: to calibrate ModelGr4H using observed historical data every 5 minutes
4. load_model1_EDLSTM.py: to load model already trained with model1_EDLSTM.py
5. model1_EDLSTM.py: to run LSTM hydrologic model, modified from here: [HydroLSTM](https://github.com/uihilab/HydroLSTM/)
6. prepare_HydroLSTM.py: prepare data to apply LSTM machine learnig method, for the historical period, years of data
7. prepare_HydroLSTM_3months.py: prepare data to apply LSTM machine learnig method, for the last 3 months
