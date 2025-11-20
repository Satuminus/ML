import os
import pandas as pd


fname = "countries_population.csv"

df = pd.read_csv(
    fname,
    sep=r"\s+(?=\d)",      
    engine="python",
    header=None,
    names=["country", "population"]
)

print(df.head())













#thousands=",""