
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. LOAD DATASET AND TRAINED MODEL
# ==========================================

df = pd.read_csv("dataset/mcu_trim_fuse.csv")

model = joblib.load("gradient_boosting_model.pkl")

print("========== MODEL EVALUATION ==========")

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

# ==========================================
# 3. RECREATE THE SAME TEST SPLIT
# ==========================================

from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups=df["wafer_id"])
)

X_test = X.iloc[test_index]
y_test = y.iloc[test_index]

# ==========================================
# 4. MAKE PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

predicted_codes = np.clip(
    np.round(predictions), 0, 63
).astype(int)

# ==========================================
# 5. CALCULATE PERFORMANCE METRICS
# ==========================================

mae = mean_absolute_error(
    y_test, predicted_codes
)

rmse = np.sqrt(
    mean_squared_error(y_test, predicted_codes)
)

r2 = r2_score(
    y_test, predicted_codes
)

accuracy = (
    np.abs(y_test.to_numpy() - predicted_codes) <= 1
).mean() * 100

print("\n========== PERFORMANCE RESULTS ==========")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"Within +/- 1 Code: {accuracy:.2f}%")

# ==========================================
# 6. ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test, predicted_codes,
    alpha=0.5
)

plt.plot(
    [0, 63], [0, 63],
    linestyle="--"
)

plt.xlabel("Actual Trim Code")
plt.ylabel("Predicted Trim Code")
plt.title("Actual vs Predicted Trim Codes")
plt.grid(True)

plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

# ==========================================
# 7. PREDICTION ERROR DISTRIBUTION
# ==========================================

errors = y_test.to_numpy() - predicted_codes

plt.figure(figsize=(8, 6))

sns.histplot(
    errors,
    bins=20,
    kde=True
)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")
plt.grid(True)

plt.tight_layout()
plt.savefig("prediction_error.png")
plt.show()

# ==========================================
# 8. FEATURE IMPORTANCE GRAPH
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(9, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance Analysis")

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ==========================================
# 9. ACTUAL VS PREDICTED COMPARISON
# ==========================================

comparison = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": predicted_codes
})

comparison = comparison.head(30)

plt.figure(figsize=(10, 6))

plt.plot(
    comparison["Actual"].values,
    marker="o",
    label="Actual"
)

plt.plot(
    comparison["Predicted"].values,
    marker="x",
    label="Predicted"
)

plt.xlabel("Test Sample")
plt.ylabel("Trim Code")
plt.title("Actual vs Predicted Comparison")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("trim_code_comparison.png")
plt.show()

# ==========================================
# 10. MODEL PERFORMANCE SUMMARY
# ==========================================

metrics = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2 Score",
        "Within +/- 1 Code"
    ],
    "Value": [
        mae,
        rmse,
        r2,
        accuracy
    ]
})

metrics.to_csv(
    "model_performance.csv",
    index=False
)

print("\n========== EVALUATION COMPLETED ==========")
print("Graphs saved successfully!")
print("Performance report saved successfully!")
