import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# Load model
model = tf.keras.models.load_model("model.keras")

# Load encoders and scaler
with open("encoded_data_geography.pkl", "rb") as file:
    encoded_geo_model = pickle.load(file)

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.title("Customer Churn Prediction")

# User Inputs
geography = st.selectbox("Geography", encoded_geo_model.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92, 35)
credit_score = st.number_input("Credit Score", value=600)
balance = st.number_input("Balance", value=60000.0)
estimated_salary = st.number_input("Estimated Salary", value=50000.0)
tenure = st.slider("Tenure", 0, 10, 3)
num_of_products = st.slider("Number of Products", 1, 4, 2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

if st.button("Predict"):

    # Create DataFrame
    input_df = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [gender],
        "Geography": [geography],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    # Label Encode Gender
    input_df["Gender"] = label_encoder_gender.transform(input_df["Gender"])

    # One-Hot Encode Geography
    geo_encoded = encoded_geo_model.transform(
        input_df[["Geography"]]
    ).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=["0", "1", "2"]
    )

    # Drop Geography
    input_df = input_df.drop("Geography", axis=1)

    # Combine
    final_input = pd.concat(
        [input_df.reset_index(drop=True),
         geo_encoded_df.reset_index(drop=True)],
        axis=1
    )

    # Scale
    final_input = scaler.transform(final_input)

    # Prediction
    prediction = model.predict(final_input)
    prediction_prob = prediction[0][0]

    st.subheader(f"Churn Probability: {prediction_prob:.2%}")

    if prediction_prob > 0.5:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")