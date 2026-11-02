
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# ==========================================
# 2. LOAD MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.pkl"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    threshold = float(joblib.load(THRESHOLD_PATH))
    return model, threshold


if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
    st.error("Model files not found. Check your models folder.")
    st.stop()

model, threshold = load_model()

# ==========================================
# 3. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.header-box {
    padding: 25px;
    border-radius: 12px;
    background: linear-gradient(120deg, #14213d, #1b263b);
    border: 1px solid #29364d;
    margin-bottom: 25px;
}

.header-title {
    font-size: 32px;
    font-weight: 700;
    color: white;
}

.header-subtitle {
    font-size: 15px;
    color: #bdc9d8;
}

.section-title {
    font-size: 21px;
    font-weight: 650;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. HEADER
# ==========================================

st.markdown("""
<div class="header-box">
    <div class="header-title">
        ⚙️ Industrial Predictive Maintenance
    </div>
    <div class="header-subtitle">
        Machine Failure Prediction & Operational Monitoring
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 5. SIDEBAR
# ==========================================

with st.sidebar:

    st.title("⚙️ Control Panel")

    st.write("Model Information")

    st.success("XGBoost Model Loaded")

    st.metric(
        "Classification Threshold",
        f"{threshold:.4f}"
    )

    st.divider()

    st.caption(
        "Industrial Predictive Maintenance Prototype"
    )

    st.caption(
        "Built using Python, Scikit-learn, "
        "XGBoost and Streamlit."
    )

# ==========================================
# 6. MACHINE INPUT SECTION
# ==========================================

st.markdown(
    '<div class="section-title">Machine Operating Parameters</div>',
    unsafe_allow_html=True
)

input_col1, input_col2 = st.columns(2)

with input_col1:

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"],
        help="L = Low, M = Medium, H = High product quality variant."
    )

    air_temperature = st.number_input(
        "Air Temperature (K)",
        min_value=250.0,
        max_value=350.0,
        value=300.0,
        step=0.1
    )

    process_temperature = st.number_input(
        "Process Temperature (K)",
        min_value=250.0,
        max_value=400.0,
        value=310.0,
        step=0.1
    )

with input_col2:

    rotational_speed = st.number_input(
        "Rotational Speed (rpm)",
        min_value=0,
        max_value=5000,
        value=1500,
        step=10
    )

    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear (min)",
        min_value=0,
        max_value=300,
        value=100,
        step=1
    )

st.divider()

# ==========================================
# 7. PREDICTION BUTTON
# ==========================================

predict_button = st.button(
    "⚡ Analyze Machine",
    type="primary",
    use_container_width=True
)

# ==========================================
# 8. PREDICTION LOGIC
# ==========================================

if predict_button:

    input_data = pd.DataFrame([{

        "Type": machine_type,

        "Air temperature [K]": air_temperature,

        "Process temperature [K]": process_temperature,

        "Rotational speed [rpm]": rotational_speed,

        "Torque [Nm]": torque,

        "Tool wear [min]": tool_wear

    }])

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    prediction = int(probability >= threshold)

    # ======================================
    # 9. RESULTS DASHBOARD
    # ======================================

    st.divider()

    st.markdown(
        '<div class="section-title">Machine Analysis Results</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Estimated Failure Probability",
            f"{probability * 100:.6f}%"
        )

        st.progress(
            min(max(probability, 0.0), 1.0),
            text="Model failure probability"
        )

    with result_col2:

        if prediction == 1:

            st.error(
                "⚠️ MACHINE FAILURE PREDICTED"
            )

            st.write(
                "The model classifies this machine as a "
                "potential failure."
            )

        else:

            st.success(
                "✅ NO MACHINE FAILURE PREDICTED"
            )

            st.write(
                "The model does not classify this machine "
                "as a failure."
            )

    # ======================================
    # 10. MACHINE PARAMETERS
    # ======================================

    st.divider()

    st.markdown(
        '<div class="section-title">Machine Parameter Summary</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        input_data,
        use_container_width=True,
        hide_index=True
    )

    # ======================================
    # 11. MODEL DIAGNOSTICS
    # ======================================

    with st.expander("View Model Diagnostics"):

        diagnostic_col1, diagnostic_col2 = st.columns(2)

        with diagnostic_col1:

            st.metric(
                "Raw Probability",
                f"{probability:.10f}"
            )

        with diagnostic_col2:

            st.metric(
                "Optimized Threshold",
                f"{threshold:.10f}"
            )

        st.write(
            "**Classification:**",
            "Failure" if prediction == 1 else "No Failure"
        )

        st.caption(
            "Classification rule: Failure is predicted when "
            "the model probability is greater than or equal "
            "to the optimized threshold."
        )

# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "Industrial Predictive Maintenance | "
    "Powered by XGBoost Machine Learning"
)

st.caption(
    "Disclaimer: This is an ML prototype based on the "
    "AI4I 2020 benchmark dataset. Predictions are not "
    "certified industrial safety assessments."
)
