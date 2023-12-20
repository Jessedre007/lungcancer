import streamlit as st
import pandas as pd
import pickle

# Load your trained model
with open("lung_cancer2.pkl", "rb") as file:  # Replace with your model filename
    model = pickle.load(file)

# Define the column names
column_names = ['Age', 'Gender', 'Air Pollution', 'Alcohol use',
                'Dust Allergy', 'OccuPational Hazards', 'Genetic Risk',
                'chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking',       
                'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue',      
                'Weight Loss', 'Shortness of Breath', 'Wheezing',
                'Swallowing Difficulty', 'Clubbing of Finger Nails', 'Frequent Cold',
                'Dry Cough', 'Snoring' ]

# Create a dictionary to store user inputs
user_input = {}

# Sidebar with a side note
st.sidebar.title(" About this App")
st.sidebar.write("""Mr Aramide write something you want 
your users to know about this application here.
                 
                 """)

# Create a Streamlit app
st.title("Lung Cancer Risk Prediction App")
st.markdown("Predict your risk of lung cancer based on the provided information.")
st.image("lung cancer.jpg", use_column_width=True)

# Create input fields for user to enter values
for column in column_names:
    user_input[column] = st.number_input(column, min_value=0, step=1)

# Create a button to make predictions
if st.button("Predict Now"):
    
    # Create a dataframe from user input
    input_data = pd.DataFrame([user_input])

    # Make predictions with the model
    prediction = model.predict(input_data)

    # Map model output to "Yes" or "No"
    prediction_text = "Medium" if prediction[0] == 1 else 'Low' if prediction[0] == 0 else "High"
    test_result = f'Opps! Your are having lung cancer at a {prediction_text} Levels'
    st.success(f"Prediction: {test_result}.")

# Add a footer with acknowledgments
st.markdown("""
---
*App created by Mr Aramide.*
[GitHub Repository](https://github.com)
""")
