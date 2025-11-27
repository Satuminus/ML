import pandas as pd

df = pd.read_csv("pvtest.csv")

df_filtered = df[(df['Dci'] != 0) & 
                 (df['Dcp'] != 0) & 
                 (df['Dcu'] != 0) & 
                 (df['Temp1'] != 0)]

print("Anzahl Zeilen nach dem Filtern:", len(df_filtered))

print(df_filtered[['Dci', 'Dcp', 'Dcu', 'Temp1']].head(10))
