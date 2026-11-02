
import pandas as pd

# Load dataset
df = pd.read_csv("dataset/mcu_trim_fuse.csv")

print("\n========== DATA QUALITY CHECK ==========")

# 1. Check MCU derivative distribution
print("\n1. DERIVATIVE DISTRIBUTION")
print(df["derivative"].value_counts())

# 2. Check wafer distribution
print("\n2. WAFER DISTRIBUTION")
print(df["wafer_id"].value_counts())

# 3. Check categorical columns
print("\n3. CATEGORICAL COLUMNS")

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].nunique(), "unique values")

# 4. Check Boolean columns
print("\n4. BOOLEAN COLUMN DISTRIBUTION")

boolean_columns = df.select_dtypes(
    include=["bool"]
).columns

for column in boolean_columns:
    print(f"\n{column}:")
    print(df[column].value_counts())

# 5. Check target range
print("\n5. TARGET VALIDATION")

target = "bg_trim_code_opt"

print("Minimum:", df[target].min())
print("Maximum:", df[target].max())

print(
    "Invalid trim codes:",
    (~df[target].between(0, 63)).sum()
)

# 6. Check important feature statistics
print("\n6. IMPORTANT FEATURE STATISTICS")

features = [
    "bg_untrim_V",
    "pcm_vth_n_mV",
    "pcm_vth_p_mV",
    "pcm_rs_ohm_sq",
    "pcm_tox_A",
    "pcm_mobility_rel",
    "radius_mm"
]

print(df[features].describe())

print("\n========== QUALITY CHECK COMPLETED ==========")
