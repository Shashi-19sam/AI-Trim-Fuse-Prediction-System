
import pandas as pd
import joblib

# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

model = joblib.load("gradient_boosting_model.pkl")

print("====================================")
print("       AI TRIM FUSE PREDICTION")
print("====================================")

# ==========================================
# 2. INPUT FEATURES
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

print("\nEnter MCU Parameters:")

input_values = {}

for feature in features:
    while True:
        try:
            value = float(input(f"Enter {feature}: "))
            input_values[feature] = value
            break
        except ValueError:
            print("Please enter a valid numeric value.")

# ==========================================
# 3. PREPARE INPUT DATA
# ==========================================

input_df = pd.DataFrame(
    [input_values],
    columns=features
)

# ==========================================
# 4. PREDICT TRIM CODE
# ==========================================

prediction = model.predict(input_df)

predicted_code = int(round(prediction[0]))

# Keep code within valid range
predicted_code = max(0, min(63, predicted_code))

# ==========================================
# 5. DISPLAY RESULT
# ==========================================

print("\n========== PREDICTION RESULT ==========")

print("Predicted BG Trim Code:", predicted_code)

print("Valid Trim Code Range: 0 to 63")

print("\n====================================")
print("       PREDICTION COMPLETED")
print("====================================")
