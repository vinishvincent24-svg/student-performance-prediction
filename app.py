
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/student_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# Custom design
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #10091F, #172554);
    color: white;
}

h1, h2, h3, p, label {
    color: white !important;
}

.stButton > button {
    background-color: #8B5CF6;
    color: white;
    border-radius: 12px;
    width: 100%;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #6D28D9;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🎓 Student Performance Predictor")

st.write(
    "Welcome! Enter your academic details "
    "to estimate your final examination score."
)

st.info(
    "This application uses Machine Learning "
    "to estimate student academic performance."
)

st.divider()

# Student input form
st.header("📝 Student Details")

with st.form("student_form"):

    study_option = st.selectbox(
        "Weekly Study Time",
        [
            "Less than 2 hours",
            "2 to 5 hours",
            "5 to 10 hours",
            "More than 10 hours"
        ]
    )

    study_mapping = {
        "Less than 2 hours": 1,
        "2 to 5 hours": 2,
        "5 to 10 hours": 3,
        "More than 10 hours": 4
    }

    g1 = st.slider(
        "First Period Grade (G1)",
        min_value=0,
        max_value=20,
        value=10
    )

    g2 = st.slider(
        "Second Period Grade (G2)",
        min_value=0,
        max_value=20,
        value=12
    )

    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=100,
        value=0
    )

    submit = st.form_submit_button(
        "🚀 Predict My Performance"
    )

# Prediction
if submit:

    student_data = pd.DataFrame({
        "studytime": [study_mapping[study_option]],
        "G1": [g1],
        "G2": [g2],
        "absences": [absences]
    })

    prediction = model.predict(student_data)[0]

    # Keep displayed grade within the dataset scale
    prediction = max(0, min(20, prediction))

    st.divider()

    st.header("📊 Your Prediction Result")

    st.metric(
        label="Predicted Final Grade",
        value=f"{prediction:.2f} / 20"
    )

    if prediction >= 16:
        st.success("Excellent! Keep up your great work! 🌟")

    elif prediction >= 12:
        st.success("Good performance! Keep improving! 👍")

    elif prediction >= 10:
        st.warning("Average performance. More practice can help!")

    else:
        st.error("You may need additional study and support.")

    st.caption(
        "Note: This is an estimated score generated "
        "by a Machine Learning model, not a guaranteed exam result."
    )

st.divider()

st.caption("Student Performance Prediction System | Python & Machine Learning")
