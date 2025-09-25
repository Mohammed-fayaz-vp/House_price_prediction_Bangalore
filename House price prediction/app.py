import pandas as pd
import pickle as pk
import streamlit as st

model = pk.load(open('House_prediction_model.pkl','rb'))

st.header('Bangalore House Price Predictor')
data = pd.read_csv('cleaned_data.csv')

loc = st.selectbox('Choose the location', data['location'].unique())
sqft = st.number_input('Enter Total sqft')
beds = st.number_input('Enter No of bedrooms')
bath = st.number_input('Enter No of bathrooms')
balc = st.number_input('Enter No of balconies')

input_df = pd.DataFrame([[loc, sqft, bath, balc, beds]], columns=['location', 'total_sqft', 'bath', 'balcony', 'bedrooms'])

if st.button("Predicted Price"):
    output = model.predict(input_df)
    st.write('Predicted price of the house: ' + str(output[0] * 100000))
