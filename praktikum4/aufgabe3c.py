import pandas as pd

# pvtest.csv laden
df = pd.read_csv("pvtest.csv")

# Zeilen herausfiltern, bei denen mindestens einer der Werte 0 ist
df_filtered = df[(df['Dci'] != 0) & 
                 (df['Dcp'] != 0) & 
                 (df['Dcu'] != 0) & 
                 (df['Temp1'] != 0)]

# Ergebnis prüfen
print("Anzahl Zeilen nach dem Filtern:", len(df_filtered))

# erste 10 Zeilen anzeigen
print(df_filtered[['Dci', 'Dcp', 'Dcu', 'Temp1']].head(10))
