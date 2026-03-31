import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import os

# ==============================
# 🔧 Load Models & Artifacts
# ==============================
@st.cache_resource
def load_artifacts():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "model.keras")

    model = tf.keras.models.load_model(model_path)

    label_path = os.path.join(BASE_DIR, "label_encoder_gender.pkl")
    geo_path = os.path.join(BASE_DIR, "onehot_encoder_geo.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

    with open(label_path, 'rb') as f:
        label_encoder_gender = pickle.load(f)

    with open(geo_path, 'rb') as f:
        onehot_encoder_geo = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, label_encoder_gender, onehot_encoder_geo, scaler


model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()

# ==============================
# 🎯 UI - Title
# ==============================
st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("💳 Customer Churn Prediction")

# ==============================
# 📥 User Inputs (Sidebar)
# ==============================
st.sidebar.header("Customer Details")

geography = st.sidebar.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox("Gender", label_encoder_gender.classes_)

age = st.sidebar.slider("Age", 18, 92, 30)
tenure = st.sidebar.slider("Tenure", 0, 10, 5)
num_of_products = st.sidebar.slider("Number of Products", 1, 4, 1)

balance = st.sidebar.number_input("Balance", value=0.0)
credit_score = st.sidebar.number_input("Credit Score", value=600)
estimated_salary = st.sidebar.number_input("Estimated Salary", value=50000.0)

has_cr_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
is_active_member = st.sidebar.selectbox("Is Active Member", [0, 1])

# ==============================
# 🧠 Preprocessing Function
# ==============================
def preprocess_input():
    # Base input
    input_df = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # Geography encoding (FIXED - using DataFrame)
    geo_df = pd.DataFrame([[geography]], columns=['Geography'])
    geo_encoded = onehot_encoder_geo.transform(geo_df).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    # Combine
    final_df = pd.concat([input_df, geo_encoded_df], axis=1)

    # Scale
    final_scaled = scaler.transform(final_df)

    return final_scaled


# ==============================
# 🔮 Prediction
# ==============================
if st.button("Predict Churn 🚀"):
    processed_input = preprocess_input()

    prediction = model.predict(processed_input)
    probability = prediction[0][0]

    st.subheader("📊 Prediction Result")

    st.metric("Churn Probability", f"{probability:.2%}")

    if probability > 0.5:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is not likely to churn")