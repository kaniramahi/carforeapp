import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import *
import pickle
import streamlit as st

# Load the saved pipeline object from the file
with open('pipeline_lr.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)

# Load the dataset
df = pd.read_pickle(open('car_details_data.pkl', 'rb'))

def main():
    # Set page config
    st.set_page_config(
        page_title="🚗CAR DEKHO🚗",
        page_icon=":car:",
        layout="centered",
    )
    # Background color
    st.markdown(
        """
        <style>
        .reportview-container {
            background: linear-gradient(rgba(255,165,0,1), rgba(255,165,0,0.8));
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title('🚗 Used Car Selling Price Prediction App')
    st.header('Fill in the details to predict the used car selling price')

    Brand_name = st.selectbox('Choose the Brand', df['Brand_name'].unique())
    year = st.selectbox('Choose the manufactured year of the car', df['year'].unique())
    Km_driven = st.number_input('Enter the kilometers reading of the vehicle', value=3000)
    fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
    seller_type = st.selectbox('Seller type', df['seller_type'].unique())
    transmission = st.selectbox('Select the type of Transmission', df['transmission'].unique())
    owner = st.selectbox('Select the Type of Owner', df['owner'].unique())

    d = {
        "Brand_name": Brand_name,
        "year": year,
        "km_driven": Km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner
    }

    test = pd.DataFrame(data=d, index=[0])

    if st.button('Predict Car Selling Price \u20B9'):
        predict_price = loaded_pipeline.predict(test)
        st.success(f'The predicted selling price is \u20B9 {predict_price[0]:,.2f}')

if __name__ == '__main__':
    main()
