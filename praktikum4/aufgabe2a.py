import pandas as pd
import matplotlib.pyplot as plt

# Iris-Daten einlesen
df = pd.read_csv("iris.csv")

# Figur mit 1 Zeile und 2 Spalten erzeugen
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Plot links: Sepal ---
axes[0].scatter(df["sepal.length"], df["sepal.width"], marker="+")
axes[0].set_xlabel("sepal length (cm)")
axes[0].set_ylabel("sepal width (cm)")
axes[0].set_title("Sepal Length vs. Width")

# --- Plot rechts: Petal ---
axes[1].scatter(df["petal.length"], df["petal.width"], color="green")
axes[1].set_xlabel("petal length (cm)")
axes[1].set_ylabel("petal width (cm)")
axes[1].set_title("Petal Length vs. Width")

plt.tight_layout()
plt.show()
