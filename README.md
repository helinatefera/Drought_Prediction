# Drought Prediction Using CNN and LSTM Techniques

This project investigates drought forecasting by leveraging various deep learning models, including Vanilla LSTM, Stacked LSTM, Bidirectional LSTM, CNN-LSTM, and ConvLSTM. The models predict future climatic variables—precipitation, maximum temperature, and minimum temperature using 33 years of Ethiopian meteorological data, with results used to compute the Standardized Precipitation Evapotranspiration Index (SPEI).

## 📁 Dataset

- **Source:** Ethiopian National Meteorology Agency
- **Features:**
  - Monthly precipitation, max temp, min temp
  - 76 cities and 81 geographical sites
  - Over 30 years of data
- **Target Variables:** Precipitation, TMPMAX, TMPMIN

## 🛠️ Data Preprocessing

- Missing geo-points estimated via equidistant interpolation.
- Elevation data filled via dCode.fr API.
- Removed incomplete entries and short records (<180 months).
- Converted monthly format to sequential time series.

## 📊 Model Architectures

- **Vanilla LSTM**
- **Stacked LSTM**
- **Bidirectional LSTM**
- **Stacked BiLSTM**
- **CNN-LSTM**
- **ConvLSTM**

All models use dropout and dense layers, compiled with Adam optimizer and trained with MSE loss.

## 📈 Evaluation Metrics

- **Regression:** MSE, MAE, MPE
- **Classification:** Accuracy, Precision, Recall, F1-Score (via SPEI classification)
- **SPEI Categories:**
  - ≥ 2.0 → Extremely Wet
  - < -2.0 → Extremely Dry

## 🧪 Experiment Setup

- Prediction horizons: 3, 6, 9, 12, and 24 months
- Train/Validation split: 80/20
- Batch size: 32 or 1
- Epochs: 100 with EarlyStopping & ReduceLROnPlateau

## 🥇 Results Summary (SPEI-3)

- **Vanilla LSTM:** Accuracy 0.80, F1-Score 0.80
- **ConvLSTM:** Accuracy 0.77, F1-Score 0.77
- **CNN-LSTM:** Accuracy 0.74, F1-Score 0.74
- **BiLSTM:** Accuracy 0.76, F1-Score 0.76

Performance degrades with increased prediction horizon.

