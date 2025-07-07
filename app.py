import numpy as np
import pandas as pd 
import pickle
import streamlit as st

# Lode the model
model = pickle.load(open("prediction_model.pkl" , "rb"))

# Lode the Data
data = pd.read_csv("cleandata.csv")

# Give the input variables 
loc = st.selectbox("Choose the location" , data["location"])
sqft = st.number_input("Enter total sqft")
bath = st.number_input("Enter total bathrooms")
beds = st.number_input("Enter total bedrooms")

# Now take input for prediction
input = pd.DataFrame([[loc , sqft, bath , beds]] , columns=['location'	,'total_sqft',	'bath'	,'bedrooms'])



#  Now put the button
if st.button("Predict"):
    result = model.predict(input)
    st.write("Prediction:", result)
