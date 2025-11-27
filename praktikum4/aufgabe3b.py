import pandas as pd

df = pd.read_csv("pvtest.csv")

print("Spaltennamen:")
print(df.columns)

df_selected = df[['Dci', 'Dcp', 'Dcu', 'Temp1']]

print("\nErste 10 Zeilen der ausgewählten Spalten:")
print(df_selected.head(10))
