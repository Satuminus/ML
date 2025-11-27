import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("pvtest.csv")

df_filtered = df[(df['Dci'] != 0) &
                 (df['Dcp'] != 0) &
                 (df['Dcu'] != 0) &
                 (df['Temp1'] != 0)]

corr = df_filtered[['Dci', 'Dcp', 'Dcu', 'Temp1']].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="magma", fmt=".2f")
plt.title("Heatmap der vier Merkmale (Dci, Dcp, Dcu, Temp1)")
plt.show()
