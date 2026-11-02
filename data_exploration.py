
import pandas as pd
import numpy as np
import os

# -----------------------------------------
# STEP 1: Load the dataset
# -----------------------------------------

file_path = os.path.join("dataset", "mcu_trim_fuse.csv")

df = pd.read_csv(file_path)

print("\n========== DATASET INFORMATION ==========")

# Display dataset dimensions
print("\nDataset Shape:", df.shape)

# Display first five records
print("\nFirst Five Records:")
print(df.head())

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display data types
print("\nData Types:")
print(df.dtypes)

# -----------------------------------------
# STEP 2: Check missing values
# -----------------------------------------

print("\n========== MISSING VALUE ANALYSIS ==========")

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])

print("\nTotal Missing Values:", df.isnull().sum().sum())

# -----------------------------------------
# STEP 3: Check duplicate records
# -----------------------------------------

print("\n========== DUPLICATE ANALYSIS ==========")

duplicate_count = df.duplicated().sum()

print("Total Duplicate Rows:", duplicate_count)

# -----------------------------------------
# STEP 4: Statistical summary
# -----------------------------------------

print("\n========== STATISTICAL SUMMARY ==========")

print(df.describe())

# -----------------------------------------
# STEP 5: Target variable analysis
# -----------------------------------------

target = "bg_trim_code_opt"

print("\n========== TARGET VARIABLE ANALYSIS ==========")

print("\nTarget Variable:", target)

print("\nTarget Statistics:")
print(df[target].describe())

print("\nTrim Code Distribution:")
print(df[target].value_counts().sort_index())

print("\n========== EXPLORATION COMPLETED ==========")
