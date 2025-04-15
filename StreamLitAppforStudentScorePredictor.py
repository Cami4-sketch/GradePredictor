import streamlit as st
import numpy as np
import pickle  # for loading your saved model

# Load trained model (use joblib or pickle after training)
model = pickle.load(open("student_model.pkl", "rb"))

st.title("📚 Final Grade Predictor")
st.subheader("Enter student info below:")

# Input fields
#G1 = st.slider("G1 (First Period Grade)", 0, 20, 10)
#G2 = st.slider("G2 (Second Period Grade)", 0, 20, 10)
G1_percent = st.slider("G1 (First Period Grade in %)", 0, 100, 75)
G2_percent = st.slider("G2 (Second Period Grade in %)", 0, 100, 80)

# Convert percentages back to 0–20 scale
G1 = G1_percent / 5
G2 = G2_percent / 5

failures = st.slider("Past Class Failures", 0, 4, 0)
studytime = st.slider("Weekly Study Time (1=Low, 4=High)", 1, 4, 2)
absences = st.slider("Number of Absences", 0, 50, 5)

# Predict when user clicks button
#if st.button("Predict Final Grade"):
 #   input_data = np.array([[G1, G2, failures, studytime, absences]])
  #  prediction = model.predict(input_data)[0]
#percentage = (prediction / 20) * 100
#st.success(f"🎯 Predicted Final Grade: {round(percentage, 2)}%")

if st.button("Predict Final Grade"):
    # Create a 2D array with inputs
    input_data = np.array([[G1, G2, failures, studytime, absences]])
    
    # Get prediction from the model
    prediction = model.predict(input_data)[0]  # Make sure this line is inside the if block

    # Convert to percentage (0–100 scale)
    percentage = min(max((prediction / 20) * 100, 0), 100)

    # Determine letter grade
    if percentage >= 90:
        letter = "A"
    elif percentage >= 80:
        letter = "B"
    elif percentage >= 70:
        letter = "C"
    elif percentage >= 60:
        letter = "D"
    else:
        letter = "F"

    # Show result
    st.success(f"🎯 Predicted Final Grade: {round(percentage, 2)}% — Letter Grade: {letter}")


#streamlit run "c:/Users/camrw/OneDrive - adams.edu/Documents/ASU FILES/FALL 2024 - Senior/Senior Capstone/Project Files/Python/AI Project/StreamLitAppforStudentScorePredictor.py"

