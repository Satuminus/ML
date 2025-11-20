import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# pvtest.csv laden
df = pd.read_csv("pvtest.csv")

# Null-Werte herausfiltern wie in Aufgabe 3c
df_filtered = df[(df['Dci'] != 0) &
                 (df['Dcp'] != 0) &
                 (df['Dcu'] != 0) &
                 (df['Temp1'] != 0)]

# Korrelation berechnen
corr = df_filtered[['Dci', 'Dcp', 'Dcu', 'Temp1']].corr()

# Heatmap zeichnen
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap der vier Merkmale (Dci, Dcp, Dcu, Temp1)")
plt.show()
