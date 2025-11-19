import os
import pandas as pd

df = pd.read_csv(
    "bundeslaender.txt",
    sep=r"\s+",
    header=0
)

# Nur Zeilen, in denen Frauen > Männer
mehr_frauen = df[df["female"] > df["male"]]

print(mehr_frauen)
print("Anzahl Bundesländer mit mehr Frauen als Männern:", len(mehr_frauen))
