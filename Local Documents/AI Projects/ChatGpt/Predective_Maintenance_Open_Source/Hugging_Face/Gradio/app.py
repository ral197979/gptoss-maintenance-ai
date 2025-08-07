import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------- Data Loading ----------
def load_data():
    # Only using FD001 for demo
    train_df = pd.read_csv("train_FD001.txt", sep=" ", header=None)
    test_df = pd.read_csv("test_FD001.txt", sep=" ", header=None)
    rul_df = pd.read_csv("RUL_FD001.txt", sep=" ", header=None)
    
    # Drop empty columns at the end
    train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
    test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
    
    # Assign column names
    cols = ['unit', 'cycle'] + [f'op_setting{i}' for i in range(1,4)] + [f'sensor{i}' for i in range(1,22)]
    train_df.columns = test_df.columns = cols

    return train_df, test_df, rul_df

# ---------- Feature Scaling ----------
def scale_data(train_df, test_df):
    scaler = MinMaxScaler()
    feature_cols = train_df.columns[2:]  # exclude unit and cycle

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_df, test_df

# ---------- RUL Labels ----------
def add_rul(train_df, test_df, rul_df):
    max_cycles = train_df.groupby('unit')['cycle'].max().reset_index()
    max_cycles.columns = ['unit', 'max_cycle']
    train_df = train_df.merge(max_cycles, on='unit')
    train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
    train_df.drop('max_cycle', axis=1, inplace=True)

    # For test
    test_rul = test_df.groupby('unit')['cycle'].max().reset_index()
    test_rul.columns = ['unit', 'last_cycle']
    test_rul['RUL'] = rul_df[0]
    test_df = test_df.merge(test_rul, on='unit')
    test_df['RUL'] = test_df['RUL'] + test_df['last_cycle'] - test_df['cycle']
    test_df.drop('last_cycle', axis=1, inplace=True)
    return train_df, test_df

# ---------- LSTM Model ----------
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def prepare_lstm_data(df, sequence_length=30):
    data = []
    labels = []
    feature_cols = df.columns[2:-1]  # exclude unit, cycle, and RUL
    for engine_id in df['unit'].unique():
        engine_data = df[df['unit'] == engine_id]
        for i in range(len(engine_data) - sequence_length):
            seq = engine_data.iloc[i:i+sequence_length][feature_cols].values
            label = engine_data.iloc[i+sequence_length]['RUL']
            data.append(seq)
            labels.append(label)
    return np.array(data), np.array(labels)

# ---------- Random Forest ----------
def train_random_forest(train_df):
    X = train_df.drop(['unit', 'cycle', 'RUL'], axis=1)
    y = train_df['RUL']
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

# ---------- Prediction ----------
def predict_rul(model_type):
    train_df, test_df, rul_df = load_data()
    train_df, test_df = scale_data(train_df, test_df)
    train_df, test_df = add_rul(train_df, test_df, rul_df)

    if model_type == "RandomForest":
        model = train_random_forest(train_df)
        latest_test = test_df.groupby("unit").last().reset_index()
        X_test = latest_test.drop(['unit', 'cycle', 'RUL'], axis=1)
        preds = model.predict(X_test)
    else:
        sequence_length = 30
        train_seq, train_labels = prepare_lstm_data(train_df, sequence_length)
        lstm_model = build_lstm_model((sequence_length, train_seq.shape[2]))
        lstm_model.fit(train_seq, train_labels, epochs=2, batch_size=64, verbose=0)

        test_seq, _ = prepare_lstm_data(test_df, sequence_length)
        preds = lstm_model.predict(test_seq[:10]).flatten()  # predict on first 10 sequences for demo

    pred_df = pd.DataFrame({"Predicted RUL": preds})
    return pred_df.head(10)

# ---------- Gradio UI ----------
demo = gr.Interface(
    fn=predict_rul,
    inputs=gr.Radio(["RandomForest", "LSTM"], label="Select Model"),
    outputs=gr.Dataframe(label="Top 10 RUL Predictions"),
    title="NASA C-MAPSS Predictive Maintenance",
    description="Select a model to predict Remaining Useful Life (RUL) using FD001 dataset."
)

if __name__ == "__main__":
    demo.launch()
