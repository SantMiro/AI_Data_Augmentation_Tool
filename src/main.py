import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

data_path = '../data/creditcard_2023.csv'

df = pd.read_csv(data_path)

# Basic exploration
print("Dataset Shape:", df.shape)
print("Dataset Columns:", df.columns)
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("Missing values in the dataset:")
print(df.isnull().sum())

# Summary statistics
print("Summary Statistics:")
print(df.describe())

# Visualize the class distribution
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.show()

# Visualize the distribution of transaction amounts
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title('Transaction Amount Distribution')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()