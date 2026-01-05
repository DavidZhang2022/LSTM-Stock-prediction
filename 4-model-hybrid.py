# ============================================
# CNN-LSTM Hybrid for Stock Price Prediction
# - Loads data (CSV or yfinance)
# - Preprocesses & creates sequences
# - Conv1D -> LSTM -> Dense
# - Trains, evaluates, plots, and forecasts
# ============================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------
# Config
# ----------------------------
USE_YFINANCE = True       # Set False if you want to load from a local CSV
TICKER = "MSFT"
START_DATE = "2010-01-01"
END_DATE = None            # None = up to today
CSV_PATH = "C:\\Users\\ZHIZHANG\\lstm-stock-price-prediction\\data\\raw\\google_stock_price_full_tsla.csv"      # Used when USE_YFINANCE=False

TARGET_COL = "Close"       # Predict next day's Close
FEATURES = ["Open", "High", "Low", "Close", "Volume"]  # multivariate inputs
# If you've computed indicators earlier (EMA/MACD/RSI/SMA), you can extend:
# FEATURES += ["ema_12","ema_26","macd_line","macd_signal","macd_hist","rsi_14","sma_10","sma_30"]

WINDOW_SIZE = 60           # past timesteps per sample
TEST_RATIO = 0.2
EPOCHS = 60
BATCH_SIZE = 32

# ----------------------------
# Load data
# ----------------------------
if USE_YFINANCE:
    import yfinance as yf
    df = yf.download(TICKER, start=START_DATE, end=END_DATE)
    df = df.reset_index()  # Date becomes a column
    # Columns are typically: Date, Open, High, Low, Close, Adj Close, Volume
else:
    # Expecting columns: Date, Open, High, Low, Close, Adj Close, Volume
    df = pd.read_csv(CSV_PATH)
    # Parse date if needed
    # df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")  # or dayfirst=True

# Ensure Date is datetime and sort
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").dropna().reset_index(drop=True)

# Keep only the columns we need (plus Date for plotting)
available = list(df.columns)
for col in [TARGET_COL] + FEATURES:
    if col not in available:
        raise ValueError(f"Column '{col}' not found in data. Available: {available}")
df = df[["Date"] + FEATURES + [TARGET_COL]].copy()

# ----------------------------
# Train/test split (chronological)
# ----------------------------
n = len(df)
split = int(n * (1 - TEST_RATIO))
df_train = df.iloc[:split].copy()
df_test  = df.iloc[split:].copy()

# ----------------------------
# Scaling (fit on train to avoid leakage)
# ----------------------------
feat_scaler = MinMaxScaler()
tgt_scaler  = MinMaxScaler()

X_train_raw = df_train[FEATURES].values
X_test_raw  = df_test[FEATURES].values
y_train_raw = df_train[[TARGET_COL]].values
y_test_raw  = df_test[[TARGET_COL]].values

X_train = feat_scaler.fit_transform(X_train_raw)
X_test  = feat_scaler.transform(X_test_raw)
y_train = tgt_scaler.fit_transform(y_train_raw)
y_test  = tgt_scaler.transform(y_test_raw)

# ----------------------------
# Sequence creation
# ----------------------------
def make_sequences(X, y, window):
    X_seq, y_seq = [], []
    for i in range(window, len(X)):
        X_seq.append(X[i-window:i])   # shape: (window, n_features)
        y_seq.append(y[i])            # next target value (scaled)
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = make_sequences(X_train, y_train, WINDOW_SIZE)
X_test_seq,  y_test_seq  = make_sequences(X_test,  y_test,  WINDOW_SIZE)

# Align dates for plotting (dates matching y values)
train_dates = df_train["Date"].iloc[WINDOW_SIZE:].reset_index(drop=True)
test_dates  = df_test["Date"].iloc[WINDOW_SIZE:].reset_index(drop=True)

print("Train sequences:", X_train_seq.shape, y_train_seq.shape)
print("Test  sequences:", X_test_seq.shape, y_test_seq.shape)

# ----------------------------
# Build CNN-LSTM model
# ----------------------------
# Input shape: (timesteps=WINDOW_SIZE, features=len(FEATURES))
model = Sequential([
    # Convolution to learn local temporal patterns; 'causal' avoids seeing future timesteps
    Conv1D(filters=64, kernel_size=5, activation="relu", padding="causal",
           input_shape=(WINDOW_SIZE, len(FEATURES))),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(filters=64, kernel_size=3, activation="relu", padding="causal"),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    # LSTM to capture longer-term dependencies after convolutional feature extraction
    LSTM(96, return_sequences=False),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dense(1)  # predict next-day Close (scaled)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)

# ----------------------------
# Train
# ----------------------------
history = model.fit(
    X_train_seq, y_train_seq,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ----------------------------
# Predict (scaled) and inverse-scale
# ----------------------------
y_train_pred_scaled = model.predict(X_train_seq)
y_test_pred_scaled  = model.predict(X_test_seq)

y_train_true = tgt_scaler.inverse_transform(y_train_seq)
y_test_true  = tgt_scaler.inverse_transform(y_test_seq)
y_train_pred = tgt_scaler.inverse_transform(y_train_pred_scaled)
y_test_pred  = tgt_scaler.inverse_transform(y_test_pred_scaled)

# ----------------------------
# Metrics
# ----------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

train_mae  = mean_absolute_error(y_train_true, y_train_pred)
train_rmse = rmse(y_train_true, y_train_pred)
test_mae   = mean_absolute_error(y_test_true, y_test_pred)
test_rmse  = rmse(y_test_true, y_test_pred)

print(f"Train MAE:  {train_mae:.4f} | RMSE: {train_rmse:.4f}")
print(f"Test  MAE:  {test_mae:.4f} | RMSE: {test_rmse:.4f}")

# ----------------------------
# Plot
# ----------------------------
train_df = pd.DataFrame({
    "Date": train_dates,
    "Actual": y_train_true.flatten(),
    "Pred":   y_train_pred.flatten()
})
test_df = pd.DataFrame({
    "Date": test_dates,
    "Actual": y_test_true.flatten(),
    "Pred":   y_test_pred.flatten()
})

plt.figure(figsize=(12,5))
plt.plot(train_df["Date"], train_df["Actual"], label="Train Actual", color="tab:blue", alpha=0.7)
plt.plot(train_df["Date"], train_df["Pred"],   label="Train Pred",   color="tab:orange", alpha=0.7)
plt.title(f"{TICKER} - CNN-LSTM Training Fit")
plt.xlabel("Date"); plt.ylabel("Close Price")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(test_df["Date"], test_df["Actual"], label="Test Actual", color="tab:blue", alpha=0.7)
plt.plot(test_df["Date"], test_df["Pred"],   label="Test Pred",   color="tab:orange", alpha=0.7)
plt.title(f"{TICKER} - CNN-LSTM Test Predictions")
plt.xlabel("Date"); plt.ylabel("Close Price")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

# ----------------------------
# One-step ahead forecast (next day)
# ----------------------------
full_features_scaled = feat_scaler.transform(df[FEATURES].values)
last_window = full_features_scaled[-WINDOW_SIZE:]              # shape (WINDOW_SIZE, n_features)
next_scaled = model.predict(last_window[np.newaxis, ...])      # add batch dimension
next_close  = tgt_scaler.inverse_transform(next_scaled)[0,0]
print(f"Predicted next {TARGET_COL} for {TICKER}: {next_close:.2f}")

# ----------------------------
# Save outputs
# ----------------------------
os.makedirs("outputs", exist_ok=True)
model.save("outputs/cnn_lstm_stock_model.h5")
train_df.to_csv("outputs/train_predictions.csv", index=False, date_format="%Y-%m-%d")
test_df.to_csv("outputs/test_predictions.csv",  index=False, date_format="%Y-%m-%d")

print("Done. Files saved in ./outputs/")
