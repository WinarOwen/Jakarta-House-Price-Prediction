import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load("lasso_reg.joblib")

@st.cache_data
def load_districts():
    districts = pd.read_csv('csvs/important_districts.csv')['district'].to_list()
    districts.append("Others")
    return districts

def rupiah(x):
    return "Rp: " + f"{x:,.0f}".replace(",",".")

model = load_model()
st.title('Jakarta House Price Prediction')

bedrooms = st.slider("Number of Bedrooms",0,20)
bathrooms = st.slider("Number of bathrooms",0,20)
carport = st.slider("Number of carport",0,20)
land_area = st.number_input("Enter land area m^2",min_value=1,max_value=999)
building_area = st.number_input("Enter building area m^2", min_value=1,max_value=999)
districts = st.selectbox("Districts",load_districts())
cities = st.selectbox("Cities", ["Jakarta Selatan","Jakarta Barat", "Jakarta Utara", "Jakarta Timur","Jakarta Pusat"])

if st.button("Predict"):
    X = pd.DataFrame([{
        'bed_rooms':bedrooms,
        'bath_rooms':bathrooms,
        'carport':carport,
        'land_area':land_area,
        'building_area':building_area,
        'district':districts,
        'city':cities
    }])
    st.write("House Price:",rupiah(np.expm1(model.predict(X)[0])))

