# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 10:20:16 2025

@author: Justin
"""

import streamlit as st
import pandas as pd
import joblib

# Load trained model and feature columns
model = joblib.load("house_price_model.pkl")
feature_columns = joblib.load("model_features.pkl")

# App Title
st.title("🏠 House Price Prediction App")
st.write("Enter house details to predict the price.")

# Sidebar / Input fields
user_input = {}
for col in feature_columns:
    if "sqft" in col.lower() or "area" in col.lower():
        user_input[col] = st.number_input(f"{col}", min_value=0.0, step=10.0)
    else:
        user_input[col] = st.number_input(f"{col}", min_value=0.0, step=1.0)

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated House Price: ₹ {prediction:,.2f}")