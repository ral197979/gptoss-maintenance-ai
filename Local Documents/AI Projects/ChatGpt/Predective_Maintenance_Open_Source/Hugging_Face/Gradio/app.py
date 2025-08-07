import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# -----------------------------
# Random Forest Model Training
# -----------------------------
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# -----------------------------
# LSTM Model Training
# -----------------------------
def train_lstm(X_train, y_train):
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, verbose=0)
    return model

# -----------------------------
# Predict Function
# -----------------------------
def predict(model_type, file):
    df = pd.read_csv(file.name, sep=' ', header=None)
    df.dropna(axis=1, inplace=True)
    df.columns = [f"sensor_{i}" for i in range(1, df.shape[1]+1)]
    
    # Simple target for demo: Remaining Useful Life = reverse index
    df['RUL'] = df.index[::-1]
    X = df.drop(columns=['RUL'])
    y = df['RUL']

    if model_type == "RandomForest":
        model = train_random_forest(X, y)
        prediction = model.predict([X.iloc[-1]])
    else:
        lstm_model = train_lstm(X.values, y.values)
        X_last = X.values[-10:].reshape((1, 10, 1)) if X.shape[0] >= 10 else X.values.reshape((1, X.shape[0], 1))
        prediction = lstm_model.predict(X_last)

    return f"Predicted Remaining Useful Life: {prediction[0][0]:.2f}"

# -----------------------------
# Gradio Interface
# -----------------------------
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Radio(["RandomForest", "LSTM"], label="Select Model Type"),
        gr.File(label="Upload CMAPSS .txt file")
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="🛠️ Predictive Maintenance with NASA C-MAPSS",
    description="Upload a CMAPSS dataset file to predict the Remaining Useful Life (RUL) of an engine."
)

if __name__ == '__main__':
    demo.launch()
