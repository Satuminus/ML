# blatt9_lstm.py
from __future__ import annotations

import os
# optional: weniger TensorFlow-Logs (kannst du auch weglassen)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# (d/e) Keras/Tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

CSV_PATH = "pvtest.csv"

FEATURE_COLS = ["Edaily", "Dci", "Dcp", "Dcu", "Temp1", "hour"]
TARGET_COL = "Dcp"

WINDOW_SIZE = 36   # 3 Stunden = 36 * 5min
HORIZON = 1        # +5 Minuten


def load_and_prepare_timeseries(csv_path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)

    df_raw["Time"] = pd.to_datetime(df_raw["Time"], format="%d.%m.%y %H:%M", errors="coerce")
    # df_raw["Time"] = pd.to_datetime(df_raw["Time"], dayfirst=True, errors="coerce")
    df_raw = df_raw.dropna(subset=["Time"]).sort_values("Time").set_index("Time")

    df_raw["hour"] = df_raw.index.hour

    df = df_raw[FEATURE_COLS].copy()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()

    return df


def train_test_split_time(df: pd.DataFrame, train_frac: float = 0.8):
    df = df.sort_index()
    split_idx = int(len(df) * train_frac)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


# ---------- AUFGABE (c) ----------

def scale_data(train_df, test_df):
    """
    StandardScaler auf Trainingsdaten fitten
    Testdaten nur transformieren!
    """
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[[TARGET_COL]].values

    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[[TARGET_COL]].values

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y


def create_windows(X, y, window_size=36, horizon=1):
    """
    Erstellt Sliding Windows:
    - Input: letzte 3 Stunden (36 Schritte, 6 Features)
    - Target: Dcp +5min
    """
    X_windows = []
    y_windows = []

    for i in range(len(X) - window_size - horizon + 1):
        X_windows.append(X[i:i + window_size])
        y_windows.append(y[i + window_size + horizon - 1])

    return np.array(X_windows), np.array(y_windows)


# ---------- AUFGABE (d) ----------

def build_model(window_size: int, n_features: int, learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    LSTM + Dropout Modell definieren und kompilieren
    """
    model = Sequential([
        Input(shape=(window_size, n_features)),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mae",
        metrics=["mae"]
    )
    return model


def main():
    # (a)
    df = load_and_prepare_timeseries(CSV_PATH)
    print("DataFrame (Zeitreihe) erstellt:", df.shape)
    print("Zeitraum gesamt:", df.index.min(), "->", df.index.max())
    print(df.head(10))

    # (b)
    train_df, test_df = train_test_split_time(df)
    print("\nTrain:", train_df.shape, "Zeitraum:", train_df.index.min(), "->", train_df.index.max())
    print("Test :", test_df.shape, "Zeitraum:", test_df.index.min(), "->", test_df.index.max())
    print("\nLetzte 3 Zeilen Train:")
    print(train_df.tail(5))
    print("\nErste 3 Zeilen Test:")
    print(test_df.head(5))

    # (c) Standardisieren
    X_train_s, X_test_s, y_train_s, y_test_s, sx, sy = scale_data(train_df, test_df)

    # (c) Fenster erzeugen
    X_train_w, y_train_w = create_windows(X_train_s, y_train_s, WINDOW_SIZE, HORIZON)
    X_test_w, y_test_w = create_windows(X_test_s, y_test_s, WINDOW_SIZE, HORIZON)

    print("\nFensterdaten:")
    print("X_train:", X_train_w.shape)
    print("y_train:", y_train_w.shape)
    print("X_test :", X_test_w.shape)
    print("y_test :", y_test_w.shape)

    print("\nBeispiel-Fenster:")
    print("Ein Fenster:", X_train_w[0].shape)
    print("Zielwert :", y_train_w[0])

    # (d) Modell definieren & kompilieren
    model = build_model(window_size=WINDOW_SIZE, n_features=len(FEATURE_COLS))
    print("\nKeras Modell (d) Summary:")
    model.summary()

    # ---------- AUFGABE (e) ----------
    # Trainieren >= 100 Epochen, mit Callbacks zur ggf. schnelleren Konvergenz
    callbacks = [
        EarlyStopping(
            monitor="val_mae",
            patience=15,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.5,
            patience=7,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        X_train_w, y_train_w,
        validation_split=0.2,     # aus Trainingsdaten eine Validierungsmenge ziehen
        epochs=150,               # >= 100 (wir nehmen 150, EarlyStopping kann früher stoppen)
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # Auswertung auf dem Testset (standardisierte Skala!)
    test_loss, test_mae = model.evaluate(X_test_w, y_test_w, verbose=0)
    print(f"\nTest MAE (standardisiert): {test_mae:.4f}")

    # Optional: MAE zurück in Originalskala (Dcp) rechnen
    # Achtung: MAE auf Originalskala ist interpretierbarer (z.B. ~50), aber nur wenn Dcp-Einheit passt
    y_pred_test = model.predict(X_test_w, verbose=0)
    y_pred_test_orig = sy.inverse_transform(y_pred_test)
    y_test_orig = sy.inverse_transform(y_test_w)

    mae_orig = np.mean(np.abs(y_pred_test_orig - y_test_orig))
    print(f"Test MAE (Originalskala Dcp): {mae_orig:.2f}")

        # ---------- AUFGABE (f) ----------

    # Vorhersagen auf Testdaten
    y_pred_test = model.predict(X_test_w, verbose=0)

    # Rückskalieren in Originaleinheit
    y_pred_orig = sy.inverse_transform(y_pred_test)
    y_test_orig = sy.inverse_transform(y_test_w)

    # Plot (nur ein Ausschnitt, sonst zu voll)
    N = 200  # erste 200 Werte anzeigen

    plt.figure(figsize=(12,5))
    plt.plot(y_test_orig[:N], label="Gemessen")
    plt.plot(y_pred_orig[:N], label="Vorhergesagt")
    plt.title("Vergleich: gemessene vs. prognostizierte Dcp-Werte")
    plt.xlabel("Zeitindex (5-Minuten Schritte)")
    plt.ylabel("Dcp")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
