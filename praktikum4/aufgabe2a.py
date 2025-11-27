import pandas as pd
import matplotlib.pyplot as plt

# Iris-Daten einlesen
df = pd.read_csv("iris.csv")

# Figur erzeugen
plt.figure(figsize=(12, 5))

# -----------------------------------
# Plot 1: Sepal
# -----------------------------------
plt.subplot(121)
plt.scatter(df["sepal.length"], df["sepal.width"], marker="+")
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.title("Sepal Length vs. Width")

# -----------------------------------
# Plot 2: Petal
# -----------------------------------
plt.subplot(122)
plt.scatter(df["petal.length"], df["petal.width"], color="green")
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.title("Petal Length vs. Width")

plt.tight_layout()
plt.show()
