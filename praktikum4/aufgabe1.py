import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# 1. Daten einlesen
# ---------------------------
df = pd.read_csv("iris.csv")

features = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
classes = ["Setosa", "Versicolor", "Virginica"]

colors = {
    "Setosa": "b",
    "Versicolor": "r",
    "Virginica": "g"
}

axis_labels = {
    "sepal.length": "sepal length (cm)",
    "sepal.width": "sepal width (cm)",
    "petal.length": "petal length (cm)",
    "petal.width": "petal width (cm)"
}

# ---------------------------
# 2. Histogramme (2x2) wie in VL (über subplot)
# ---------------------------
plt.figure(figsize=(12, 8))

# Erstes Histogramm
plt.subplot(221)
for cls in classes:
    subset = df[df["variety"] == cls][features[0]]
    plt.hist(subset, bins=10, alpha=0.7, color=colors[cls], edgecolor="black", label=cls.lower())
plt.xlabel(axis_labels[features[0]])
plt.ylabel("Häufigkeit")
plt.legend()

# Zweites Histogramm
plt.subplot(222)
for cls in classes:
    subset = df[df["variety"] == cls][features[1]]
    plt.hist(subset, bins=10, alpha=0.7, color=colors[cls], edgecolor="black", label=cls.lower())
plt.xlabel(axis_labels[features[1]])
plt.ylabel("Häufigkeit")
plt.legend()

# Drittes Histogramm
plt.subplot(223)
for cls in classes:
    subset = df[df["variety"] == cls][features[2]]
    plt.hist(subset, bins=10, alpha=0.7, color=colors[cls], edgecolor="black", label=cls.lower())
plt.xlabel(axis_labels[features[2]])
plt.ylabel("Häufigkeit")
plt.legend()

# Viertes Histogramm
plt.subplot(224)
for cls in classes:
    subset = df[df["variety"] == cls][features[3]]
    plt.hist(subset, bins=10, alpha=0.7, color=colors[cls], edgecolor="black", label=cls.lower())
plt.xlabel(axis_labels[features[3]])
plt.ylabel("Häufigkeit")
plt.legend()

plt.tight_layout()
plt.show()
