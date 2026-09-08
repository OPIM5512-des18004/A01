from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)
print(df.columns)

cols = df.columns

plt.boxplot(df,orientation='horizontal',tick_labels=cols)
plt.show()