import pandas as pd

df = pd.read_csv("iris.csv")

corr_petal = df["petal.length"].corr(df["petal.width"])
corr_sepal = df["sepal.length"].corr(df["sepal.width"])

print(f"The correlation coefficient between petal length (cm) and petal width (cm) attributes is {corr_petal:.5f}.")
print(f"The correlation coefficient between sepal length (cm) and sepal width (cm) attributes is {corr_sepal:.5f}.")
