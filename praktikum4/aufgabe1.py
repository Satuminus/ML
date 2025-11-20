import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# 1. Daten einlesen
# ---------------------------
# Falls iris.csv in einem anderen Ordner liegt, Pfad entsprechend anpassen
df = pd.read_csv("iris.csv")

# Spaltennamen für die vier numerischen Merkmale
features = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

# Klassen (Objektklassen)
classes = ["Setosa", "Versicolor", "Virginica"]

# Farben für die Klassen
colors = {
    "Setosa": "b",       # blau
    "Versicolor": "r",   # rot
    "Virginica": "g"     # grün
}

# Achsentitel (wie in der Aufgabenstellung mit (cm))
axis_labels = {
    "sepal.length": "sepal length (cm)",
    "sepal.width": "sepal width (cm)",
    "petal.length": "petal length (cm)",
    "petal.width": "petal width (cm)"
}

# ---------------------------
# 2. Histogramme zeichnen
# ---------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()  # 2x2-Array in flache Liste umwandeln

for ax, feature in zip(axes, features):
    for cls in classes:
        subset = df[df["variety"] == cls][feature]
        ax.hist(
            subset,
            bins=10,
            alpha=0.7,
            label=cls.lower(),   # Beschriftung setosa / versicolor / virginica
            color=colors[cls],
            edgecolor="black"
        )
    ax.set_xlabel(axis_labels[feature])
    ax.set_ylabel("Häufigkeit")
    ax.legend()

plt.tight_layout()
plt.show()
