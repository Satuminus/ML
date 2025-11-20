import pandas as pd

df = pd.read_csv(
    "bundeslaender.txt",
    sep=r"\s+",
    header=0
)

df["population"] = df["male"] + df["female"]

df["density"] = (df["population"] * 1000) / df["area"]

df["density"] = df["density"].round(0)

dichte_hoch = df[df["density"] > 1000]

dichte_hoch = dichte_hoch[["land", "area", "female", "male", "population", "density"]]

print(dichte_hoch)
