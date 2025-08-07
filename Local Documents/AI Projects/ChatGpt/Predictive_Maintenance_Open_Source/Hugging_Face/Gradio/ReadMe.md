# 🔧 Predictive Maintenance App with ML & LLMs

A complete end-to-end application for Predictive Maintenance powered by:
- ✅ Machine Learning (RandomForest, LSTM)
- ✅ Real industrial datasets (C-MAPSS, IEEE Motors, UCI Pumps, PHM Bearings)
- ✅ Visualizations: RMSE, Prediction Plots, Feature Importance
- ✅ AI Explanation: HuggingFace LLMs to explain model output

## 🔍 Features
- Upload your own CSV data with RUL column (Remaining Useful Life)
- Choose from preloaded datasets
- Run:
  - 🌲 RandomForest (Tabular)
  - 🧠 LSTM (Time-Series)
- Visualize predictions and errors
- Understand model behavior with LLM explanations
- Export CSV predictions and JSON feature importance

## 📊 Equipment Types Supported
| Equipment        | Sensors                      | Use Case                          |
|------------------|------------------------------|-----------------------------------|
| 🏗️ Pumps          | Flow, pressure, vibration    | Detect pump failure early         |
| ⚙️ Motors         | Current, temperature, RPM    | Predict motor burnout             |
| 🛢️ Compressors    | Pressure, load, oil temp     | Avoid breakdowns                  |
| 🚚 Vehicles       | Speed, battery temp, acc     | Fleet maintenance                 |
| 🏭 Bearings/Gear  | Acoustic, load, wear rate    | Predict mechanical wear           |
| 🔌 Generators     | Voltage, rotor temp          | Prevent turbine failures          |

## 📁 Preloaded Datasets
- **C-MAPSS (Jet Engines)** – NASA
- **IEEE Motor Faults** – Motor condition monitoring
- **UCI Hydraulic System** – Pump and valve behavior
- **PHM Bearings** – Vibration-based bearing degradation

## 🛠️ How to Use
1. Select a dataset or upload your CSV
2. Choose model tab (RandomForest or LSTM)
3. Click "Run Model"
4. Review results, plots, and download outputs

## 📦 Dependencies
Listed in `requirements.txt`. Includes:
- `scikit-learn`, `tensorflow`, `gradio`, `transformers`, `seaborn`, `matplotlib`

## 🧠 Powered by
- ML: `RandomForestRegressor`, `LSTM` (Keras)
- LLM: `HuggingFaceH4/zephyr-7b-beta` for explanations

---

### 📡 Live App
[🚀 Launch Predictive Maintenance App](https://huggingface.co/spaces/ral197979/Predictive-Maintenance-App)

---
