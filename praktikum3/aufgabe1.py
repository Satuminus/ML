import pandas as pd

df = pd.read_csv(
    "countries_population.csv",
    sep=r"\s+(?=\d)",
    engine="python",
    header=None,
    names=["Country", "Population"],
    thousands=","
)

df["Country"] = df["Country"].str.strip("'")

df = df.set_index("Country")

print(df.head())
