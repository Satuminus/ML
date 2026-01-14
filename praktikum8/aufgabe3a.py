import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2


# ------------------------------------------------------------
# Aufgabe 3a:
# - gleicher Lern- und Testdatensatz wie in Aufgabe 1 und 2
# - Sequential Model wie 1a (2 Dense Layers à 60 Neuronen)
# - dieses Mal mit L2-Regularisierung (ohne Callbacks)
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

# 3) Standardisierung (nur X, Label nicht!)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4) Train-Test-Split (wie zuvor: 20% Testdaten, reproduzierbar)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)

# 5) L2-Regularisierung definieren (Vorlesungsbereich: typisch 0.001 bis 0.0001)
reg = l2(0.001)

# 6) Modell wie 1a, aber mit kernel_regularizer=reg in den Dense-Layern
model = Sequential()
model.add(Dense(units=60, input_shape=(5,), activation="relu", kernel_regularizer=reg))
model.add(Dense(units=60, activation="relu", kernel_regularizer=reg))
model.add(Dense(units=3, activation="softmax", kernel_regularizer=reg))

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

# 7) Architektur ausgeben (Nachweis für 3a)
model.summary()

print("\nHinweis: In Aufgabe 3b wird dieses Modell ohne Callbacks trainiert (>=200 Epochen) und der Lernprozess geplottet.")