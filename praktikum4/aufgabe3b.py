import pandas as pd

# pvtest.csv laden
df = pd.read_csv("pvtest.csv")

# Spaltennamen anzeigen
print("Spaltennamen:")
print(df.columns)

# relevante Spalten auswählen
df_selected = df[['Dci', 'Dcp', 'Dcu', 'Temp1']]

# erste 10 Zeilen zeigen
print("\nErste 10 Zeilen der ausgewählten Spalten:")
print(df_selected.head(10))
