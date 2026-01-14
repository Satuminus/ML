import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2


# ------------------------------------------------------------
# Aufgabe 3b:
# - gleicher Lern- und Testdatensatz wie in Aufgabe 1 und 2
# - Sequential Model wie 1a mit L2-Regularisierung (ohne Callbacks)
# - Training mit mind. 200 Epochen
# - Lernkurve visualisieren: train loss vs. val loss
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

# 4) Train-Test-Split (20% Testdaten, reproduzierbar)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)

# 5) L2-Regularisierung definieren
reg = l2(0.001)

# 6) Modell mit L2-Regularisierung (wie 3a)
model = Sequential()
model.add(Dense(units=60, input_shape=(5,), activation="relu", kernel_regularizer=reg))
model.add(Dense(units=60, activation="relu", kernel_regularizer=reg))
model.add(Dense(units=3, activation="softmax", kernel_regularizer=reg))

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

model.summary()

# 7) Training (mind. 200 Epochen)
epochs = 200
batch_size = 32

history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    verbose=True
)

# 8) Lernkurve plotten: train loss vs. val loss
loss_train = history.history["loss"]
loss_test = history.history["val_loss"]
epochs_axis = range(1, len(loss_train) + 1)

plt.figure()
plt.plot(epochs_axis, loss_train, label="train loss")
plt.plot(epochs_axis, loss_test, label="test loss")
plt.xlabel("epochs")
plt.ylabel("loss (sparse cross entropy)")
plt.legend()
plt.show()

# Optional: finale Evaluation
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinale Evaluation auf Testdaten: loss={loss:.5f}, accuracy={acc:.5f}")
print(f"Training lief über {len(loss_train)} Epochen (soll >= 200 sein).")