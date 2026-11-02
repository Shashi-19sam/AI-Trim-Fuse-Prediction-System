
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/mcu_trim_fuse.csv")

print("========== DATA PREPROCESSING ==========")
print("\nOriginal Dataset Shape:", df.shape)

# ==========================================
# 2. SELECT INPUT FEATURES
# ==========================================

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

# Target variable
target = "bg_trim_code_opt"

# Grouping column
group_column = "wafer_id"

# ==========================================
# 3. VALIDATE REQUIRED COLUMNS
# ==========================================

required_columns = features + [target, group_column]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

# ==========================================
# 4. CREATE INPUT AND TARGET DATA
# ==========================================

X = df[features].copy()
y = df[target].copy()
groups = df[group_column].copy()

print("\nInput Feature Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================================
# 5. CHECK MISSING VALUES
# ==========================================

print("\nMissing Values in Input Features:")
print(X.isnull().sum())

print("\nMissing Values in Target:")
print(y.isnull().sum())

if X.isnull().any().any() or y.isnull().any():
    raise ValueError(
        "Missing values found. Handle them before training."
    )

# ==========================================
# 6. GROUPED TRAIN-TEST SPLIT
# ==========================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

train_groups = groups.iloc[train_index]
test_groups = groups.iloc[test_index]

# ==========================================
# 7. VERIFY WAFER SEPARATION
# ==========================================

overlapping_wafers = (
    set(train_groups.unique())
    & set(test_groups.unique())
)

print("\n========== SPLIT RESULTS ==========")

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

print("Training Wafers:", train_groups.nunique())
print("Testing Wafers:", test_groups.nunique())

print("\nTraining Percentage:",
      round(len(X_train) / len(X) * 100, 2), "%")

print("Testing Percentage:",
      round(len(X_test) / len(X) * 100, 2), "%")

print("\nOverlapping Wafers:", overlapping_wafers)

if overlapping_wafers:
    raise ValueError("Wafer leakage detected!")

print("\nSUCCESS: No wafer overlap detected.")

# ==========================================
# 8. DISPLAY SAMPLE TRAINING DATA
# ==========================================

print("\nSample Training Features:")
print(X_train.head())

print("\nSample Training Targets:")
print(y_train.head())

print("\n========== PREPROCESSING COMPLETED ==========")
