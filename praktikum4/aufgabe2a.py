import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(df["sepal.length"], df["sepal.width"], marker="+")
axes[0].set_xlabel("sepal length (cm)")
axes[0].set_ylabel("sepal width (cm)")

axes[1].scatter(df["petal.length"], df["petal.width"], color="green")
axes[1].set_xlabel("petal length (cm)")
axes[1].set_ylabel("petal width (cm)")

plt.tight_layout()
plt.show()
