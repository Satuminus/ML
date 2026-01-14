import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Aufgabe 1a:
# - Datensatz laden
# - Features skalieren (StandardScaler)
# - Sequentielles Keras-Modell mit 2 Dense Layers à 60 Neuronen erstellen

# 1) Datensatz laden
df = pd.read_csv("rawdata_luftqualitaet.csv")

# 2) Features auswählen (5 Merkmale)
feature_cols = [
    "humidity_inside",
    "temperature_inside",
    "co2_inside",
    "temperature_heater",
    "temperature_wall_inside",
]
X = df[feature_cols].values

# 3) Standardisieren (nur X, Label nicht!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4) Sequentielles Modell erstellen
model = Sequential()
model.add(Dense(units=60, input_shape=(5,), activation="relu"))
model.add(Dense(units=60, activation="relu"))
model.add(Dense(units=3, activation="softmax"))  # 3 Klassen: 0, 1, 2

# 5) Kompilieren (Multiklassen-Klassifikation)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

# 6) Ausgabe der Architektur (Nachweis für 1a)
model.summary()
print("\nX_scaled shape:", X_scaled.shape)