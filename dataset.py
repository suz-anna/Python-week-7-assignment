# Import required libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Load the Iris dataset
try:
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame
    print("Dataset loaded successfully!")
except Exception as e:
    print("Error loading dataset:", e)

# Display the first few rows
print(df.head())

# Check data types and missing values
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# Clean dataset (if any missing values)
if df.isnull().values.any():
    df.fillna(df.mean(numeric_only=True), inplace=True)
    print("\nMissing values filled.")
else:
    print("\nNo missing values found.")


# Basic statistics
print("\nBasic Statistics:\n", df.describe())

# Group by 'target' (species) and calculate mean
grouped = df.groupby('target').mean(numeric_only=True)
print("\nMean values per species:\n", grouped)

# Map target numbers to species names for readability
df['species'] = df['target'].map(dict(zip(range(3), iris_data.target_names)))
print("\nData with species names:\n", df[['target', 'species']].drop_duplicates())


import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set(style="whitegrid")

# 1. Line Chart: Simulated petal length trend (not time-series by default)
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["petal length (cm)"])
plt.title("Petal Length Trend Over Index")
plt.xlabel("Index")
plt.ylabel("Petal Length (cm)")
plt.grid(True)
plt.show()

# 2. Bar Chart: Average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(x="species", y="petal length (cm)", data=df, palette="viridis")
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram: Sepal length distribution
plt.figure(figsize=(8, 5))
plt.hist(df["sepal length (cm)"], bins=15, color='skyblue', edgecolor='black')
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot: Sepal length vs Petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="sepal length (cm)", y="petal length (cm)", hue="species", palette="Set1")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
