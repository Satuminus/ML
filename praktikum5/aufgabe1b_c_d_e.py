import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Daten laden
# ----------------------------------------------------
df = pd.read_csv("rawdata_luftqualitaet.csv")

# ----------------------------------------------------
# Eingangsmerkmale und Zielvariable
# ----------------------------------------------------
X = df[[
    "humidity_inside",
    "temperature_inside",
    "co2_inside",
    "temperature_heater",
    "temperature_wall_inside"
]]

y = df["state_air_quality"]

# ----------------------------------------------------
# Aufgabe 1b: Trainings- und Testdatensatz erzeugen
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Trainingsdatengröße:", X_train.shape)
print("Testdatengröße:     ", X_test.shape)

# ----------------------------------------------------
# Aufgabe 1c: Daten skalieren mit StandardScaler
# ----------------------------------------------------

# ----------------------------------------------------
# Aufgabe 1c: Daten skalieren mit MinMaxScaler (0..1)
# ----------------------------------------------------
scaler = MinMaxScaler()

# Fit nur auf den Trainingsdaten!
scaler.fit(X_train)

# Transformieren
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Skalierung (0..1) abgeschlossen.")

# ----------------------------------------------------
# VISUALISIERUNG: Histogramm vorher vs. nachher
# ----------------------------------------------------
scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

fig, axes = plt.subplots(5, 2, figsize=(12, 15))
axes = axes.ravel()

for i, col in enumerate(X_train.columns):
    # Original
    axes[2*i].hist(X_train[col], bins=30, color="steelblue")
    axes[2*i].set_title(f"{col} – vor Skalierung")

    # Skaliert
    axes[2*i+1].hist(scaled_df[col], bins=30, color="orange")
    axes[2*i+1].set_title(f"{col} – nach Skalierung")

plt.tight_layout()
plt.show()

from sklearn.neural_network import MLPClassifier

# ----------------------------------------------------
# Aufgabe 1d: MLPClassifier trainieren
# ----------------------------------------------------

# Modell erstellen (VL-konform)
mlp = MLPClassifier(
    hidden_layer_sizes=(50,),  # 1 versteckte Schicht mit 50 Neuronen
    activation='relu',         # ReLU-Aktivierungsfunktion (VL-Standard)
    solver='adam',             # Adam-Optimizer (Standard)
    max_iter=1000,             # genügend Iterationen
    random_state=42            # Reproduzierbarkeit
)

# Modell auf den TRAININGSDATEN trainieren
print("\nTraining läuft, bitte warten...")
mlp.fit(X_train_scaled, y_train)

print("Training abgeschlossen.")

# Trainings-Accuracy
train_acc = mlp.score(X_train_scaled, y_train)
print("Trainingsgenauigkeit:", train_acc)

# Test-Accuracy
test_acc = mlp.score(X_test_scaled, y_test)
print("Testgenauigkeit:", test_acc)

# Loss-curve plotten
import matplotlib.pyplot as plt
plt.plot(mlp.loss_curve_)
plt.title("LossCurve während des Trainings")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Aufgabe 1e: Modellgüte ausgeben
# ----------------------------------------------------

# Vorhersage auf den Testdaten
y_pred = mlp.predict(X_test_scaled)

# Genauigkeit
acc = accuracy_score(y_test, y_pred)
print("Genauigkeit (Accuracy) auf Testdaten:", acc)

# Konfusionsmatrix
cm = confusion_matrix(y_test, y_pred)

print("\nKonfusionsmatrix:")
print(cm)

# Visualisierung der Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="magma", fmt="d")
plt.title("Confusion Matrix des MLP")
plt.xlabel("Vorhergesagte Klasse")
plt.ylabel("Wahre Klasse")
plt.tight_layout()
plt.show()

# Optional: detaillierter Bericht
print("\nClassification Report:")
print(classification_report(y_test, y_pred))