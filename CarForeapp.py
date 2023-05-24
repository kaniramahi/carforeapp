import pandas as pd   # data preprocessing
import numpy as np    # mathematical computation
from sklearn import *
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure()  # Create a Figure object

# Load the model and dataset
model = pickle.load(open('dt_modelc.pkl','rb'))
df = pickle.load(open('datac.pkl','rb'))

def main():
        # Set page config
    st.set_page_config(
        page_title="🚗CarForesight🚗",
        page_icon=":car:",
        layout="centered",
        #background_color="#FFA500" # Orange background color
        )


    # Page title
    st.title('Used Car Data EDA By Krishna Singh')

    
    #st.set_page_config(page_title="🚗 Used Car Price EDA", page_icon=":car:", layout="centered")
    #st.title('🚗 Used Car Price Prediction')

    #Add a scatter plot of the age and selling price
    
    st.subheader('Age vs Selling Price :money_with_wings:')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="year", y="selling_price", ax=ax) # corrected y-axis label
    st.pyplot(fig)

    # Add a pair plot of the numerical features
    st.subheader('Pair Plot of Numerical Features :bar_chart:')
    num_cols = ['name','year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'selling_price','model']
    fig = sns.pairplot(df[num_cols], diag_kind='kde', plot_kws={'alpha': 0.4})
    fig.fig.set_size_inches(12, 10)
    st.pyplot(fig)

    # Add a correlation heatmap
    st.subheader('Correlation Heatmap :fire:')
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', square=True, ax=ax)
    st.pyplot(fig)

    # Add the input form for the prediction using sidebar
    st.sidebar.header('📝 Fill the details to predict the Car Price')
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Enter Car Details")

    brand = st.sidebar.selectbox('Brand', df['name'].unique())
    km_driven = st.sidebar.number_input('Enter the kilometers reading of the vehicle', value=300000)
    fuel = st.sidebar.selectbox('Fuel', df['fuel'].unique())
    seller_type = st.sidebar.selectbox('Seller Type', df['seller_type'].unique())
    transmission = st.sidebar.selectbox('Transmission', df['transmission'].unique(), index=0)
    owner = st.sidebar.selectbox('Owner', df['owner'].unique())
    Age = st.sidebar.number_input('Age of vehicle in years (1-32)', value=10)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Prediction Results")
    st.sidebar.markdown("Click the button below to see the predicted price.")

    if st.sidebar.button('Used Car Price Prediction :money_with_wings:'):
        test = np.array([brand, km_driven, fuel, seller_type, transmission, owner, Age])
        test = test.reshape([1,7])
        predicted_price = model.predict(test)[0]
        predicted_price_rounded = round(predicted_price, 2)
        st.sidebar.success(f'Predicted Price: {predicted_price_rounded} :dollar:')
    
   
    
    # Footer section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Follow me on")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button('GitHub'):
            st.sidebar.markdown('[Visit my GitHub profile and check the codes](https://github.com/kaniramahi/laptop_price.git)')
    with col2:
        if st.sidebar.button('LinkedIn'):
            st.sidebar.markdown('[Visit my LinkedIn profile](https://www.linkedin.com/in/krishna-singh-3a086b193')

if __name__ == "__main__":
    main()
