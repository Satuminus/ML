import os
import pandas as pd

df = pd.read_csv(
    "bundeslaender.txt",
    sep=r"\s+",
    header=0
)

# Hochskalieren, da sonst berechnung mit fläche nicht realistisch
df["male"] = df["male"] * 1000
df["female"] = df["female"] * 1000

df["population"] = df["male"] + df["female"]
df["density"] = df["population"] / df["area"]

df = df[["land", "area", "female", "male", "population", "density"]]

dichte_hoch = df[df["density"] > 1000]

print(dichte_hoch)