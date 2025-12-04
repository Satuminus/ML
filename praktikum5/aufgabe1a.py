import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas.plotting as pd_plot

# 1. Datensatz laden
df = pd.read_csv("rawdata_luftqualitaet.csv")

print("Erste 10 Zeilen der Datentabelle:")
print(df.head(10))
print()

#print("Info zum DataFrame:")
#print(df.info())
#print()

# 2. Statistische Kennwerte
stats = df.describe().T
print("Statistische Kennwerte:")
print(stats)
print()

# Luftwualität rausfiltern und sortieren
if "state_air_quality" in df.columns:
    print("Häufigkeit pro Klasse:")
    print(df["state_air_quality"].value_counts())
    print()

# Numerische Spalten auswählen
numeric_cols = df.select_dtypes(include="number").columns
feature_cols = [c for c in numeric_cols if c != "state_air_quality"]

print("Numerische Features:")
print(feature_cols)
print()

# Liniendiagramm erstellen
subset = df[feature_cols].iloc[:300]

plt.figure(figsize=(12, 6))
subset.plot(ax=plt.gca())
plt.title("Liniendiagramm der Messwerte (Ausschnitt)")
plt.xlabel("Messindex")
plt.ylabel("Messwerte")
plt.tight_layout()
plt.show(block=False)

# Heatmap erstellen
corr = df[feature_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="magma")
plt.title("Korrelations-Heatmap der Messmerkmale")
plt.tight_layout()
plt.show(block=False)

# Scattermatrix erstellen
pd_plot.scatter_matrix(df[feature_cols], figsize=(10, 10), diagonal="hist")
plt.suptitle("Scattermatrix der Messmerkmale", y=1.02)
plt.tight_layout()
plt.show(block=False)

input("Enter drücken um alles zu schließen")