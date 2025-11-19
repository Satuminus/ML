import os
import pandas as pd

fname = "bundeslaender.txt"

# Datei einlesen
df = pd.read_csv(
    fname,
    sep=r"\s+",     # beliebig viele Leerzeichen als Trennzeichen
    header=0
)

# neue Spalten erzeugen
df["population"] = df["male"] + df["female"]
df["density"] = df["population"] / df["area"]

# Ausgabe der ersten Zeilen
print(df.head())

# neue Datei speichern
df.to_csv("bundeslaender_neu.csv", index=False)
