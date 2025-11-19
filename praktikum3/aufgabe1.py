import os
import pandas as pd


fname = "countries_population.csv"

df = pd.read_csv(
    fname,
    sep=r"\s+(?=\d)",      # trenne am Leerzeichen VOR der Zahl
    engine="python",       # nötig für Regex-Separator mit Lookahead
    header=None,           # es gibt keine Kopfzeile in der Datei
    names=["country", "population"],  # Spaltennamen setzen
    thousands=",",         # Kommas in Zahlen als Tausendertrennzeichen behandeln
    quotechar="'",         # Länder stehen in einfachen Anführungszeichen
)

print(df.head())
