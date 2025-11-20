import pandas as pd

# pvtest.csv einlesen
df = pd.read_csv("pvtest.csv")

# erste 10 Zeilen anzeigen
print(df.head(10))
