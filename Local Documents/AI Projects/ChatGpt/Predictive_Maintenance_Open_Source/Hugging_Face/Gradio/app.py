import os
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import joblib
import json
import io
from transformers import pipeline

# --- Utility Functions ---
def load_data(file):
    df = pd.read_csv(file.name)
    return df

def prepare_rf_data(df):
    df = df.copy()
    if 'RUL' not in df.columns:
        raise ValueError("Missing 'RUL' column in dataset")
    X = df.drop(['RUL'], axis=1)
    y = df['RUL']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return model, preds, rmse

def plot_predictions(y_test, preds):
    fig, ax = plt.subplots()
    ax.scatter(range(len(y_test)), y_test, label="Actual")
    ax.scatter(range(len(preds)), preds, label="Predicted", alpha=0.7)
    ax.set_title("Predicted vs Actual RUL")
    ax.legend()
    return fig

def plot_feature_importance(model, X_train):
    importances = model.feature_importances_
    features = X_train.columns
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots()
    sns.barplot(x=importances[indices], y=features[indices], ax=ax)
    ax.set_title("Feature Importances")
    return fig

def prepare_lstm_data(df):
    df = df.copy()
    if 'RUL' not in df.columns:
        raise ValueError("Missing 'RUL' column in dataset")
    features = df.drop('RUL', axis=1).values
    target = df['RUL'].values
    sequence_length = 10
    generator = TimeseriesGenerator(features, target, length=sequence_length, batch_size=32)
    X, y = [], []
    for i in range(len(generator)):
        batch_x, batch_y = generator[i]
        X.append(batch_x)
        y.append(batch_y)
    X = np.concatenate(X)
    y = np.concatenate(y)
    return generator, X, y, features.shape[1]

def train_lstm_model(generator, n_features):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(10, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(generator, epochs=10, verbose=0)
    return model

def predict_lstm(model, generator):
    preds = model.predict(generator)
    return preds

def calculate_rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def plot_lstm_predictions(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.plot(y_true, label='Actual')
    ax.plot(y_pred, label='Predicted', alpha=0.7)
    ax.set_title("LSTM Predicted vs Actual RUL")
    ax.legend()
    return fig

def export_predictions_csv(y_true, y_pred):
    df_out = pd.DataFrame({'Actual': y_true.flatten(), 'Predicted': y_pred.flatten()})
    buffer = io.StringIO()
    df_out.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer.read()

def export_feature_importance(model, X_train):
    importance_dict = dict(zip(X_train.columns, model.feature_importances_))
    return json.dumps(importance_dict, indent=2)

def explain_with_llm(prompt):
    summarizer = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta")
    response = summarizer(prompt, max_new_tokens=100)[0]['generated_text']
    return response

# --- Gradio UI ---
def run_rf_pipeline(file):
    try:
        df = load_data(file)
        X_train, X_test, y_train, y_test = prepare_rf_data(df)
        model, preds, rmse = train_random_forest(X_train, X_test, y_train, y_test)
        fig_pred = plot_predictions(y_test.values, preds)
        fig_imp = plot_feature_importance(model, X_train)
        csv_output = export_predictions_csv(y_test.values, preds)
        importance_json = export_feature_importance(model, X_train)
        explanation = explain_with_llm(f"Explain why these features are important in predicting RUL: {list(X_train.columns)}")
        return f"RMSE: {rmse:.2f}", fig_pred, fig_imp, csv_output, importance_json, explanation
    except Exception as e:
        return str(e), None, None, None, None, None

def run_lstm_pipeline(file):
    try:
        df = load_data(file)
        generator, X, y, n_features = prepare_lstm_data(df)
        model = train_lstm_model(generator, n_features)
        preds = predict_lstm(model, generator)
        rmse = calculate_rmse(y, preds)
        fig_pred = plot_lstm_predictions(y, preds)
        csv_output = export_predictions_csv(y, preds)
        explanation = explain_with_llm("Explain how LSTM is effective for predictive maintenance using time-series RUL data.")
        return f"LSTM RMSE: {rmse:.2f}", fig_pred, csv_output, explanation
    except Exception as e:
        return str(e), None, None, None

with gr.Blocks() as demo:
    gr.Markdown("""# Predictive Maintenance with NASA C-MAPSS\nUpload your processed CSV with RUL labels.""")

    with gr.Tab("RandomForest"): 
        rf_file = gr.File(label="Upload CSV")
        rf_button = gr.Button("Run RF Model")
        rf_output_text = gr.Textbox(label="RMSE")
        rf_pred_plot = gr.Plot(label="Prediction Plot")
        rf_importance_plot = gr.Plot(label="Feature Importance")
        rf_csv = gr.File(label="Download Predictions (CSV)", interactive=False)
        rf_json = gr.Textbox(label="Feature Importance JSON", lines=5)
        rf_explain = gr.Textbox(label="LLM Explanation", lines=5)

        rf_button.click(run_rf_pipeline, inputs=[rf_file], 
                        outputs=[rf_output_text, rf_pred_plot, rf_importance_plot, rf_csv, rf_json, rf_explain])

    with gr.Tab("LSTM"):
        lstm_file = gr.File(label="Upload CSV")
        lstm_button = gr.Button("Run LSTM Model")
        lstm_output_text = gr.Textbox(label="RMSE")
        lstm_plot = gr.Plot(label="Prediction Plot")
        lstm_csv = gr.File(label="Download Predictions (CSV)", interactive=False)
        lstm_explain = gr.Textbox(label="LLM Explanation", lines=5)

        lstm_button.click(run_lstm_pipeline, inputs=[lstm_file], 
                          outputs=[lstm_output_text, lstm_plot, lstm_csv, lstm_explain])

    gr.Markdown("""---\nℹ️ This demo supports both RandomForest and LSTM models. Upload a CSV with numeric features and a `RUL` column.""")

demo.launch()
