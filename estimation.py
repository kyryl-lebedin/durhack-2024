import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import joblib
import os
import random

def predict(name):
    sector = 'Industrials'

    # Base directory where models are saved
    # models_base_dir = 'stocks/models/'
    # output_dir = os.path.join(models_base_dir, sector)

    # # Load the saved LSTM model for the 'Industrials' sector
    # model_filename = os.path.join(output_dir, 'stock_lstm_model.h5')
    # # loaded_model = load_model(model_filename)

    # Load the saved scalers
    # scaler_X_filename = os.path.join(output_dir, 'scaler_X.save')
    # scaler_y_filename = os.path.join(output_dir, 'scaler_y.save')

    # scaler_X_loaded = joblib.load(scaler_X_filename)
    # scaler_y_loaded = joblib.load(scaler_y_filename)


    # Load your new data (replace with your actual data file)
    # df_X = pd.read_csv('stocks/data/Industrials.csv', header=0)

    # # Ensure your DataFrame has enough rows to create a sequence of length 10
    # if len(df_X) < 10:
    #     raise ValueError("Not enough data to create a sequence of length 10.")

    # Extract features from columns 2-11 and 13-17 (indices 1-10 and 12-16)
    # feature_indices = list(range(1, 11)) + list(range(12, 17))  # Columns 2-11 and 13-17
    # X_new = df_X.iloc[:, feature_indices]

    # # Scale the new input data using the loaded scaler
    # # X_new_scaled = scaler_X_loaded.transform(X_new)


    # seq_length = 10  # Same as during training
    predictions = random.uniform(30, 50)

    # Create sequences from the scaled input data
    # X_new_seq = []
    # for i in range(len(X_new_scaled) - seq_length + 1):
    #     X_new_seq.append(X_new_scaled[i:i + seq_length, :])

    # X_new_seq = np.array(X_new_seq)  # Shape: (n_samples, seq_length, n_features)


    # Make predictions on all sequences
    # predictions_scaled = loaded_model.predict(X_new_seq)

    # # Inverse transform the predictions to get actual stock values
    # predictions = scaler_y_loaded.inverse_transform(predictions_scaled)
    return predictions
    # Now, 'predictions' contains the predicted stock values


print(predict('ur ass'))
