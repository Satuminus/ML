import pandas as pd

df = pd.read_csv(
    "bundeslaender.txt",
    sep=r"\s+",
    header=0
)

df["population"] = df["male"] + df["female"]
df["density"] = df["population"] / df["area"]

df = df[["land", "area", "female", "male", "population", "density"]]

print(df.head())

df.to_csv("bundeslaender_neu.csv", index=False)