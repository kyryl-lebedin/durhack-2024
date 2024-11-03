import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import joblib  # For saving the scaler
import os

# List of sectors
sectors = [
    'Basic Materials', 'Communication Services', 'Consumer Cyclical',
    'Consumer Defensive', 'Financial Services', 'Industrials', 
    'Real Estate', 'Technology', 'Utilities', 'Healthcare', 'Energy'
]

# Base directory for saving models
models_base_dir = 'stocks/models/'
os.makedirs(models_base_dir, exist_ok=True)

for sector in sectors:
    print(f"Processing sector: {sector}")
    
    # Create output directory for the sector
    output_dir = os.path.join(models_base_dir, sector)
    os.makedirs(output_dir, exist_ok=True)
    
    # -----------------------------
    # Part 1: Load and Prepare Data
    # -----------------------------
    
    # Load your data into a DataFrame called df
    file = f'stocks/data/{sector}.csv'
    df = pd.read_csv(file, header=0)
    
    # Prepare the data
    # Assuming columns 2-11 are daily stock values over 10 years
    # Columns 13-17 are risk factors
    X = df.iloc[:, list(range(1, 11)) + list(range(12, 17))]  # Columns 2-11 and 13-17
    y = df.iloc[:, 11]  # 12th column is the target variable (stock value next year)
    
    # Scale X and y separately
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    
    # Save the scalers for future use
    scaler_X_filename = os.path.join(output_dir, 'scaler_X.save')
    scaler_y_filename = os.path.join(output_dir, 'scaler_y.save')
    joblib.dump(scaler_X, scaler_X_filename)
    joblib.dump(scaler_y, scaler_y_filename)
    
    # Create sequences
    def create_sequences(X_data, y_data, seq_length):
        X_seq = []
        y_seq = []
        for i in range(len(X_data) - seq_length):
            X_seq.append(X_data[i:(i + seq_length), :])
            y_seq.append(y_data[i + seq_length])
        return np.array(X_seq), np.array(y_seq)
    
    seq_length = 10
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
    
    # Split into training and testing sets
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    
    # ---------------------------------------
    # Part 2: Build, Train, and Save the Model
    # ---------------------------------------
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Implement early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0  # Set to 1 to see training progress
    )
    
    # Save the trained model
    model_filename = os.path.join(output_dir, 'stock_lstm_model.h5')
    model.save(model_filename)
    
    # ---------------------------
    # Part 3: Evaluate the Model
    # ---------------------------
    
    # Make predictions
    predictions_scaled = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    predictions = scaler_y.inverse_transform(predictions_scaled)
    y_test_actual = scaler_y.inverse_transform(y_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test_actual, predictions)
    print(f'Sector: {sector}, Mean Squared Error: {mse}')
    
    # Save predictions and actual values for future reference
    predictions_filename = os.path.join(output_dir, 'predictions.npy')
    y_test_actual_filename = os.path.join(output_dir, 'y_test_actual.npy')
    np.save(predictions_filename, predictions)
    np.save(y_test_actual_filename, y_test_actual)
    
    # -------------------------------------------
    # Part 4: Load Model and Use on New Input Data
    # -------------------------------------------
    
    # Example function to load the model and make a prediction for a specific sector
    def predict_future_stock_value(new_input_data, sector):
        """
        new_input_data: A DataFrame or 2D array containing the new input features.
                        It should have the same number of features as the training data.
                        For sequence prediction, provide data for the past 'seq_length' time steps.
        sector: The sector for which to load the model and scalers.
        """
        # Define the output directory for the sector
        output_dir = os.path.join(models_base_dir, sector)
        
        # Load the saved model
        model_filename = os.path.join(output_dir, 'stock_lstm_model.h5')
        loaded_model = load_model(model_filename)
        
        # Load the saved scalers
        scaler_X_filename = os.path.join(output_dir, 'scaler_X.save')
        scaler_y_filename = os.path.join(output_dir, 'scaler_y.save')
        scaler_X_loaded = joblib.load(scaler_X_filename)
        scaler_y_loaded = joblib.load(scaler_y_filename)
        
        # Scale the new input data
        new_input_scaled = scaler_X_loaded.transform(new_input_data)
        
        # Reshape the input data to match the LSTM input shape
        new_input_sequence = np.array([new_input_scaled])
        
        # Make prediction
        predicted_scaled = loaded_model.predict(new_input_sequence)
        
        # Inverse transform the prediction
        predicted_value = scaler_y_loaded.inverse_transform(predicted_scaled)
        
        return predicted_value[0][0]  # Return the predicted stock value
    
    # Example usage (uncomment and modify with actual data)
    # new_input_data = df.iloc[-seq_length:, list(range(1, 11)) + list(range(12, 17))].values
    # predicted_stock_value = predict_future_stock_value(new_input_data, sector)
    # print(f'Sector: {sector}, Predicted Stock Value for Next Year: {predicted_stock_value}')
