import pandas as pd

# Datei einlesen
df = pd.read_csv(
    "bundeslaender.txt",
    sep=r"\s+",
    header=0
)

# neue Spalten berechnen
df["population"] = df["male"] + df["female"]
df["density"] = df["population"] / df["area"]

# gewünschte Spaltenreihenfolge
df = df[["land", "area", "female", "male", "population", "density"]]

# Ergebnis anzeigen
print(df.head())

# neue Datei speichern
df.to_csv("bundeslaender_neu.csv", index=False)