import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

features = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

classes = ["Setosa", "Versicolor", "Virginica"]

colors = {
    "Setosa": "b",       # blau
    "Versicolor": "r",   # rot
    "Virginica": "g"     # grün
}

axis_labels = {
    "sepal.length": "sepal length (cm)",
    "sepal.width": "sepal width (cm)",
    "petal.length": "petal length (cm)",
    "petal.width": "petal width (cm)"
}


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()

for ax, feature in zip(axes, features): 
    for cls in classes:
        subset = df[df["variety"] == cls][feature]
        ax.hist(
            subset,
            bins=10,
            alpha=0.6,
            label=cls.lower(),
            color=colors[cls],
            edgecolor="black"
        )
    ax.set_xlabel(axis_labels[feature])
    ax.legend()

plt.tight_layout()
plt.show()
