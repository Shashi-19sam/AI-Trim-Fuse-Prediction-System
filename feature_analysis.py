
import pandas as pd

# Load dataset
df = pd.read_csv("dataset/mcu_trim_fuse.csv")

print("\n========== FEATURE ANALYSIS ==========")

# Define target
target = "bg_trim_code_opt"

# Candidate input features
features = [
    "bg_untrim_V",
    "pcm_vth_n_mV",
    "pcm_vth_p_mV",
    "pcm_rs_ohm_sq",
    "pcm_tox_A",
    "pcm_mobility_rel",
    "die_x",
    "die_y",
    "radius_mm",
    "package",
    "temp_grade",
    "derivative"
]

# Display selected features
print("\nSelected Input Features:")
for feature in features:
    print(feature)

# Display target
print("\nTarget Variable:")
print(target)

# Check missing values in selected data
print("\nMissing Values:")
print(df[features + [target]].isnull().sum())

# Display sample input data
print("\nSample Input Features:")
print(df[features].head())

# Display target samples
print("\nTarget Samples:")
print(df[target].head())

# Check categorical features
print("\nCategorical Features:")
print(df[features].select_dtypes(
    include=["object", "str"]
).columns.tolist())

# Display correlation of numeric features with target
print("\nFeature Correlation with Target:")

numeric_data = df[features].select_dtypes(
    include="number"
).copy()

numeric_data[target] = df[target]

print(
    numeric_data.corr()[target]
    .sort_values(ascending=False)
)

print("\n========== FEATURE ANALYSIS COMPLETED ==========")
