import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="CVD Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------
st.markdown("""
<style>
    /* Overall page */
    .main {
        background-color: #0e1117;
    }

    /* Header banner */
    .hero {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 40%, #b91c1c 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 24px rgba(153, 27, 27, 0.35);
    }
    .hero h1 {
        color: #ffffff;
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
        font-weight: 800;
    }
    .hero p {
        color: #fecaca;
        font-size: 1.05rem;
        margin: 0;
    }

    /* Section cards */
    .section-card {
        background-color: #1a1d24;
        border: 1px solid #2a2e38;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f3f4f6;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Predict button */
    div.stButton > button {
        background: linear-gradient(135deg, #dc2626, #b91c1c);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.7rem 1.5rem;
        border-radius: 10px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 14px rgba(220, 38, 38, 0.4);
        transition: transform 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(220, 38, 38, 0.55);
    }

    /* Result cards */
    .result-card {
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1rem;
        border: 1px solid #2a2e38;
    }
    .result-score {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.2rem 0;
    }
    .result-label {
        font-size: 1rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .risk-low { background: linear-gradient(135deg, #052e1a, #064e2c); border-color: #16a34a; }
    .risk-low .result-score { color: #4ade80; }
    .risk-moderate { background: linear-gradient(135deg, #3a2a05, #4d3a08); border-color: #ca8a04; }
    .risk-moderate .result-score { color: #facc15; }
    .risk-high { background: linear-gradient(135deg, #3a0505, #4d0808); border-color: #dc2626; }
    .risk-high .result-score { color: #f87171; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #14161c;
    }

    /* Footer note */
    .footer-note {
        text-align: center;
        color: #6b7280;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #2a2e38;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("cvd_risk_model.pkl")
    # scaler = joblib.load("scaler.pkl")  # uncomment if you used one
    return model

model = load_model()

# ---------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>❤️ Cardiovascular Disease Risk Predictor</h1>
    <p>Enter patient health metrics to estimate their CVD risk score using a trained regression model.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Sidebar — about / instructions
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ℹ️ About this tool")
    st.write(
        "This app uses a linear regression model trained on patient "
        "clinical and lifestyle data to estimate cardiovascular disease risk."
    )
    st.markdown("---")
    st.markdown("### 📋 How to use")
    st.write(
        "1. Fill in the patient's details across the sections.\n"
        "2. Click **Predict CVD Risk**.\n"
        "3. Review the estimated score and risk band."
    )
    st.markdown("---")
    st.caption("⚠️ For educational purposes only — not a substitute for professional medical advice.")

# ---------------------------------------------------------------------
# Input form, grouped into sections
# ---------------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧍 Demographics & Body Metrics</div>', unsafe_allow_html=True)
    age = st.number_input("Age", 1, 120, 40)
    sex = st.selectbox("Sex", ["Female", "Male"])
    height = st.number_input("Height (cm)", 100.0, 250.0, 170.0)
    weight = st.number_input("Weight (kg)", 20.0, 250.0, 70.0)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    abd_circ = st.number_input("Abdominal Circumference (cm)", 40.0, 200.0, 90.0)
    wh_ratio = st.number_input("Waist-to-Height Ratio", 0.1, 1.5, 0.5)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🩺 Blood Pressure</div>', unsafe_allow_html=True)
    sbp = st.number_input("Systolic BP", 60.0, 250.0, 120.0)
    dbp = st.number_input("Diastolic BP", 40.0, 150.0, 80.0)
    bp_cat = st.selectbox(
        "Blood Pressure Category",
        [0, 1, 2, 3],
        format_func=lambda x: {0: "Normal", 1: "Elevated", 2: "Hypertension Stage 1", 3: "Hypertension Stage 2"}[x],
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧪 Blood Work</div>', unsafe_allow_html=True)
    chol = st.number_input("Total Cholesterol (mg/dL)", 50.0, 400.0, 180.0)
    hdl = st.number_input("HDL (mg/dL)", 10.0, 150.0, 50.0)
    ldl = st.number_input("Estimated LDL (mg/dL)", 0.0, 300.0, 100.0)
    fbs = st.number_input("Fasting Blood Sugar (mg/dL)", 50.0, 400.0, 90.0)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🚬 Lifestyle & History</div>', unsafe_allow_html=True)
    smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Yes" if x else "No")
    diabetes = st.selectbox("Diabetes Status", [0, 1], format_func=lambda x: "Yes" if x else "No")
    activity = st.selectbox(
        "Physical Activity Level", [0, 1, 2],
        format_func=lambda x: {0: "Low", 1: "Moderate", 2: "High"}[x],
    )
    family_hist = st.selectbox("Family History of CVD", [0, 1], format_func=lambda x: "Yes" if x else "No")
    st.markdown('</div>', unsafe_allow_html=True)
    cvd_level = st.selectbox("CVD risk level", [0, 1, 2],format_func=lambda x: {0: "Low", 1: "Moderate", 2: "High"}[x],)


# ---------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------
sex_f = 1 if sex == "Female" else 0
sex_m = 1 if sex == "Male" else 0

predict_clicked = st.button("🔍 Predict CVD Risk")

if predict_clicked:
    input_dict = {
        "Age": [age],
        "Weight (kg)": [weight],
        "BMI": [bmi],
        "Abdominal Circumference (cm)": [abd_circ],
        "Total Cholesterol (mg/dL)": [chol],
        "HDL (mg/dL)": [hdl],
        "Fasting Blood Sugar (mg/dL)": [fbs],
        "Smoking Status": [smoking],
        "Diabetes Status": [diabetes],
        "Physical Activity Level": [activity],
        "Family History of CVD": [family_hist],
        "Height (cm)": [height],
        "Waist-to-Height Ratio": [wh_ratio],
        "Systolic BP": [sbp],
        "Diastolic BP": [dbp],
        "Blood Pressure Category": [bp_cat],
        "Estimated LDL (mg/dL)": [ldl],
        "CVD Risk Level" : [cvd_level],
        "Sex_F": [sex_f],
        "Sex_M": [sex_m],
    }
    input_df = pd.DataFrame(input_dict)

    # Reorder columns to exactly match training order
    input_df = input_df[model.feature_names_in_]

    # input_df = scaler.transform(input_df)  # if scaler was used

    prediction = model.predict(input_df)[0]

    # Bucket the numeric score into a risk band for display
    # NOTE: adjust these thresholds to match your actual target scale
    if prediction < 1.5:
        band_class, band_label = "risk-low", "Low Risk"
    elif prediction < 2.5:
        band_class, band_label = "risk-moderate", "Moderate Risk"
    else:
        band_class, band_label = "risk-high", "High Risk"

    st.markdown(f"""
    <div class="result-card {band_class}">
        <div class="result-label">Predicted CVD Risk Score</div>
        <div class="result-score">{prediction:.2f}</div>
        <div class="result-label">{band_label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">Built with Streamlit • Model: Linear Regression • For educational use only</div>',
    unsafe_allow_html=True,
)