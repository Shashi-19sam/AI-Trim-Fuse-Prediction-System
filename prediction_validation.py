
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Load dataset
df = pd.read_csv("dataset/mcu_trim_fuse.csv")

features = [
    "bg_untrim_V",
    "pcm_vth_n_mV",
    "pcm_vth_p_mV",
    "pcm_rs_ohm_sq",
    "pcm_tox_A",
    "pcm_mobility_rel",
    "die_x",
    "die_y",
    "radius_mm"
]

target = "bg_trim_code_opt"

# Recreate the same train-test split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        df[features],
        df[target],
        groups=df["wafer_id"]
    )
)

# Extract unseen testing records
test_data = df.iloc[test_idx]

print("\n========== INPUT FEATURE RANGES ==========")
print(df[features].describe().loc[["min", "max"]])

print("\n========== FIVE TESTING RECORDS ==========")

sample = test_data[features + [target]].head(5)

print(sample.to_string(index=False))

print("\nActual Trim Codes:")
print(sample[target].tolist())

print("\n========== VALIDATION DATA READY ==========")
