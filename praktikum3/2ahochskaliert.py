import os
import pandas as pd

df = pd.read_csv(
    "bundeslaender.txt",
    sep=r"\s+",
    header=0
)

# Werte sind in Tausendern angegeben → hochskalieren
df["male"] = df["male"] * 1000
df["female"] = df["female"] * 1000

# neue Spalten
df["population"] = df["male"] + df["female"]
df["density"] = df["population"] / df["area"]

print(df.head())

# neue Datei erzeugen
df.to_csv("bundeslaender_neu_hochskaliert.csv", index=False)
