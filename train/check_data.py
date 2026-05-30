import pandas as pd

# load data
data = pd.read_csv('train/../data/processed_plant_vase1.csv')

# basic checks
print("Shape:", data.shape)
print("\nColumns:", data.columns)
print("\nFirst 5 rows:\n", data.head())

# check missing values
print("\nMissing values:\n", data.isnull().sum())