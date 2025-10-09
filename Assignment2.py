import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer
from sklearn.metrics.pairwise import cosine_similarity

# Part I - Feature Selection

# Load the dataset (Assume it's already downloaded locally from Kaggle)
df = pd.read_csv('AdventureWorks.csv')

# Selected features for analysis (these are assumed based on the assignment)
df_selected = df[['Age', 'Gender', 'YearlyIncome', 'CommuteDistance', 'Occupation', 'MaritalStatus', 'BikeBuyer']]

# Identify data types (Nominal, Ordinal, Continuous)
# For instance:
# - Nominal: 'Gender', 'Occupation', 'MaritalStatus'
# - Ordinal: Not specifically provided, but could be inferred if any variable is ordinal
# - Continuous: 'Age', 'YearlyIncome', 'CommuteDistance'


# Part II - Data Preprocessing

# Handling Nulls: Drop rows with any missing values (you can also choose to fill them)
df_selected = df_selected.dropna()

# Normalization (MinMax Scaling): Rescale 'Age' and 'YearlyIncome' to range [0, 1]
scaler = MinMaxScaler()
df_selected[['Age', 'YearlyIncome']] = scaler.fit_transform(df_selected[['Age', 'YearlyIncome']])

# Discretization: Binning 'YearlyIncome' into 4 equal-width bins
kb = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
df_selected['Income_bin'] = kb.fit_transform(df_selected[['YearlyIncome']])

# One-Hot Encoding: Convert categorical variables ('Gender', 'Occupation', 'MaritalStatus') to dummy variables
df_selected = pd.get_dummies(df_selected, columns=['Gender', 'Occupation', 'MaritalStatus'], drop_first=True)

# Standardization: Standardize 'Age' and 'YearlyIncome' to have a mean of 0 and variance of 1
std_scaler = StandardScaler()
df_selected[['Age', 'YearlyIncome']] = std_scaler.fit_transform(df_selected[['Age', 'YearlyIncome']])

# Display the processed DataFrame to verify
print(df_selected.head())

# Part III - Similarity & Correlation

# Example Similarity Calculation
# Take the first two rows (objects) from the dataset
obj1 = df_selected.iloc[0]
obj2 = df_selected.iloc[1]

# Simple Matching, Jaccard, Cosine Similarity
# Using Cosine Similarity from sklearn
cosine = cosine_similarity([obj1], [obj2])

# Print Cosine Similarity
print(f"Cosine Similarity between Object 1 and Object 2: {cosine[0][0]}")

# Correlation: Find correlation between 'CommuteDistance' and 'YearlyIncome'
# Note that we need to ensure both features are continuous (which they are after scaling/standardization)
corr = df_selected['CommuteDistance'].corr(df_selected['YearlyIncome'])

# Print correlation result
print(f"Correlation between CommuteDistance and YearlyIncome: {corr}")


# --------------------------------------------------
# Name: Adityavir Singh Randhawa
# Roll No: 102483009
# Sub Group: 3C53
# --------------------------------------------------
