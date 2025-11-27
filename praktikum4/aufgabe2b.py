import pandas as pd

# Iris-Daten einlesen
df = pd.read_csv("iris.csv")

# Korrelation berechnen (Series.corr ist in den VLs erlaubt)
corr_petal = df["petal.length"].corr(df["petal.width"])
corr_sepal = df["sepal.length"].corr(df["sepal.width"])

# Ausgabe exakt wie gefordert (ohne F-Strings, wie in VL üblich)
print("The correlation coefficient between petal length (cm) and petal width (cm) attributes is",
      round(corr_petal, 5), ".")

print("The correlation coefficient between sepal length (cm) and sepal width (cm) attributes is",
      round(corr_sepal, 5), ".")
