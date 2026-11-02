import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/mcu_trim_fuse.csv")

print("========== MODEL TRAINING ==========")

# ==========================================
# 2. SELECT FEATURES AND TARGET
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

target = "bg_trim_code_opt"

X = df[features]
y = df[target]
groups = df["wafer_id"]

# ==========================================
# 3. GROUPED TRAIN-TEST SPLIT
# ==========================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# ==========================================
# 4. INITIALIZE MODEL
# ==========================================

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# ==========================================
# 5. TRAIN MODEL
# ==========================================

print("\nTraining the Gradient Boosting Model...")

model.fit(X_train, y_train)

print("Model Training Completed!")

# ==========================================
# 6. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# Trim codes are integer values.
# Round predictions to the nearest integer
# and keep them within the valid 0-63 range.

y_pred_codes = pd.Series(y_pred).round().clip(
    lower=0,
    upper=63
).astype(int)

# ==========================================
# 7. EVALUATE MODEL
# ==========================================

mae = mean_absolute_error(y_test, y_pred_codes)

mse = mean_squared_error(y_test, y_pred_codes)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred_codes)

within_one = (
    (abs(y_test.reset_index(drop=True) -
         y_pred_codes) <= 1).mean() * 100
)

print("\n========== MODEL EVALUATION ==========")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2 Score : {r2:.4f}")

print(f"Within +/- 1 Trim Code : {within_one:.2f}%")

# ==========================================
# 8. FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance.to_string(index=False))

# ==========================================
# 9. ACTUAL VS PREDICTED VALUES
# ==========================================

comparison = pd.DataFrame({
    "Actual Trim Code": y_test.values,
    "Predicted Trim Code": y_pred_codes.values
})

comparison["Absolute Error"] = abs(
    comparison["Actual Trim Code"] -
    comparison["Predicted Trim Code"]
)

print("\n========== SAMPLE PREDICTIONS ==========")
print(comparison.head(15).to_string(index=False))

print("\n========== TRAINING COMPLETED ==========")

# ==========================================
# 10. SAVE TRAINED MODEL
# ==========================================

joblib.dump(model, "gradient_boosting_model.pkl")

print("\nModel saved successfully!")