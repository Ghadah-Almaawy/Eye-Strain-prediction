import joblib
import pandas as pd
import streamlit as st

# The model was trained on questionnaire answers encoded as small integers,
# so the app asks the same questions and sends the same codes to the model.
QUESTIONS = {
    "Device habits": {
        "phone_usage_hours": ("Daily phone usage", ["Less than 2 hours", "2–4 hours", "4–6 hours", "More than 6 hours"], 2),
        "computer_usage_hours": ("Daily computer usage", ["Less than 2 hours", "2–4 hours", "4–6 hours", "More than 6 hours"], 2),
        "use_before_sleep": ("Use devices before sleeping", ["Never", "Sometimes", "Always"], 1),
        "break_frequency": ("Breaks taken during use", ["No breaks", "Once", "2–3 times", "More than 3 times"], 1),
        "screen_distance": ("Distance from screen", ["Very close", "Close", "Moderate", "Far"], 2),
    },
    "Environment": {
        "screen_brightness": ("Screen brightness", ["Low", "Medium", "High"], 1),
        "room_lighting": ("Room lighting", ["Dim", "Moderate", "Bright"], 1),
        "eye_protection_usage": ("Eye protection tools (glasses, filters)", ["Never", "Sometimes", "Always"], 1),
        "night_mode_usage": ("Night mode or blue light filter", ["Never", "Sometimes", "Always"], 1),
        "sleep_hours": ("Sleep per night", ["Less than 5 hours", "5–6 hours", "7–8 hours", "More than 8 hours"], 2),
    },
    "About you and symptoms": {
        "age": ("Age", ["Under 18", "18–30", "Over 30"], 1),
        "gender": ("Gender", ["Female", "Male"], 0),
        "employment_status": ("Employment status", ["Student", "Employee", "Other"], 0),
        "eye_dryness": ("Eye dryness", ["Never", "Sometimes", "Often"], 1),
        "headache_fatigue": ("Headaches or eye fatigue", ["Never", "Sometimes", "Often"], 1),
    },
}


@st.cache_resource
def load_model():
    return joblib.load("eye_strain_model.pkl")


model = load_model()

st.set_page_config(page_title="Eye Strain Prediction", page_icon="👁️")
st.title("Eye Strain Prediction")
st.write(
    "Answer a few questions about your screen habits and environment to estimate your risk of "
    "digital eye strain. The model is a tuned Logistic Regression with 76.7% accuracy on held out test data."
)

answers = {}
for section, questions in QUESTIONS.items():
    st.subheader(section)
    for feature, (label, options, default) in questions.items():
        choice = st.selectbox(label, options, index=default, key=feature)
        answers[feature] = options.index(choice)

if st.button("Predict", type="primary"):
    input_data = pd.DataFrame([answers])[list(model.feature_names_in_)]
    probability = model.predict_proba(input_data)[0][1]

    if probability >= 0.5:
        st.error(f"Higher eye strain risk: estimated likelihood {probability:.0%}")
    else:
        st.success(f"Lower eye strain risk: estimated likelihood {probability:.0%}")

    st.caption("This is an educational project and not a medical diagnosis.")
