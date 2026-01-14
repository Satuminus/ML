import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# ------------------------------------------------------------
# Aufgabe 2a:
# - gleicher Lern- und Testdatensatz wie in Aufgabe 1
# - Sequential Model wie 1a (2 Dense Layers à 60 Neuronen)
# - EarlyStopping Callback definieren
# ------------------------------------------------------------

# 1) Datensatz laden
df = pd.read_csv("rawdata_luftqualitaet.csv")

# 2) Features und Label trennen
feature_cols = [
    "humidity_inside",
    "temperature_inside",
    "co2_inside",
    "temperature_heater",
    "temperature_wall_inside",
]
X = df[feature_cols].values
y = df["state_air_quality"].values  # Klassen: 0, 1, 2

# 3) Standardisierung (nur X)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4) Train-Test-Split (wie zuvor: 20% Test, reproduzierbar)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)

# 5) Modell wie Aufgabe 1a erstellen
model = Sequential()
model.add(Dense(units=60, input_shape=(5,), activation="relu"))
model.add(Dense(units=60, activation="relu"))
model.add(Dense(units=3, activation="softmax"))

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

model.summary()

# 6) EarlyStopping Callback definieren (Abbruch, wenn val_loss nicht besser wird)
early_stopping = EarlyStopping(
    monitor="val_loss",   # überwacht Verlust auf Validierungsdaten
    patience=5            # 5 Epochen ohne Verbesserung -> Abbruch
)

print("\nEarlyStopping Callback erstellt:")
print("monitor =", early_stopping.monitor)
print("patience =", early_stopping.patience)

# Hinweis: In Aufgabe 2b wird dieser Callback in model.fit(...) übergeben:
# callbacks=[early_stopping]