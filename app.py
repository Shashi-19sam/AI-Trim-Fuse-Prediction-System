
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Trim Fuse Prediction",
    page_icon="⚙️",
    layout="wide"
)

# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("gradient_boosting_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load trained model: {e}")
    st.stop()

# ==========================================
# 3. PROJECT HEADER
# ==========================================

st.title("⚙️ AI Trim Fuse Prediction System")

st.markdown(
    """
    ### Machine Learning Based MCU Bandgap Trim Code Prediction

    This application predicts the optimal Bandgap (BG)
    Trim Code using MCU manufacturing parameters.
    """
)

st.divider()

# ==========================================
# 4. INPUT SECTION
# ==========================================

st.subheader("Enter MCU Parameters")

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        bg_untrim_V = st.number_input(
            "BG Untrim Voltage (V)",
            value=1.19017,
            format="%.5f"
        )

        pcm_vth_n_mV = st.number_input(
            "PCM VTH N (mV)",
            value=452.04,
            format="%.2f"
        )

        pcm_vth_p_mV = st.number_input(
            "PCM VTH P (mV)",
            value=-477.49,
            format="%.2f"
        )

    with col2:
        pcm_rs_ohm_sq = st.number_input(
            "PCM Resistance (Ohm/Sq)",
            value=284.88,
            format="%.2f"
        )

        pcm_tox_A = st.number_input(
            "PCM TOX (A)",
            value=20.175,
            format="%.3f"
        )

        pcm_mobility_rel = st.number_input(
            "PCM Mobility Relative",
            value=0.9986,
            format="%.5f"
        )

    with col3:
        die_x = st.number_input(
            "Die X",
            min_value=1,
            max_value=23,
            value=9
        )

        die_y = st.number_input(
            "Die Y",
            min_value=1,
            max_value=23,
            value=1
        )

        radius_mm = st.number_input(
            "Radius (mm)",
            min_value=0.0,
            value=91.21,
            format="%.2f"
        )

    st.divider()

    predict_button = st.form_submit_button(
        "🔍 Predict BG Trim Code",
        use_container_width=True
    )

# ==========================================
# 5. PREDICTION
# ==========================================

if predict_button:

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

    input_data = pd.DataFrame(
        [[
            bg_untrim_V,
            pcm_vth_n_mV,
            pcm_vth_p_mV,
            pcm_rs_ohm_sq,
            pcm_tox_A,
            pcm_mobility_rel,
            die_x,
            die_y,
            radius_mm
        ]],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    predicted_code = int(
        round(prediction)
    )

    predicted_code = max(
        0,
        min(63, predicted_code)
    )

    # ======================================
    # 6. DISPLAY RESULT
    # ======================================

    st.success("Prediction Completed Successfully!")

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted BG Trim Code",
            predicted_code
        )

    with col2:
        st.metric(
            "Valid Trim Code Range",
            "0 – 63"
        )

    st.progress(
        predicted_code / 63
    )

    st.subheader("Input Parameters")

    st.dataframe(
        input_data,
        use_container_width=True
    )

    st.info(
        "The predicted trim code is a model estimate. "
        "Validate it with real hardware measurements "
        "before any fuse programming."
    )

# ==========================================
# 7. FOOTER
# ==========================================

st.divider()

st.caption(
    "AI Trim Fuse Prediction Project | "
    "Powered by Python, Streamlit and Gradient Boosting"
)
