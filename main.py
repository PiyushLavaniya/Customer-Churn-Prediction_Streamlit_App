import streamlit as st
import numpy as np
import pandas as pd
import pickle
from preprocess import preprocess, get_model


def main():
    
    st.title('Telco Customer Churn Prediction App')

      #Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict customer churn in a ficitional telecommunication use case.
    The application is functional for both online prediction and batch data prediction. n
    """)
    
    st.subheader("Demographic data")
    seniorcitizen = st.selectbox('Senior Citizen:', ('Yes', 'No'))
    location = st.selectbox('Location:', ('Los Angeles', 'New York', 'Miami', 'Chicago', 'Houston'))
    gender = st.selectbox('Gender:', ('Male', 'Female'))
    st.subheader("Payment data")
    tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0)
    monthlycharges = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150, value=0)
    total_GB_usage = st.number_input('The total amount of GB used by the customer',min_value=0, max_value=10000, value=0)
    
    data = {
            'SeniorCitizen': seniorcitizen,
            'Location': location,
            'Gender': gender,
            'tenure':tenure,
            'MonthlyCharges': monthlycharges,
            'total_GB_usage': total_GB_usage
           }
    
    features_df = pd.DataFrame.from_dict([data])
    
    st.write('Overview of input is shown below')
    st.dataframe(features_df)
    
    preprocess_df = preprocess(features_df)
    model = get_model('Models/model.pkl')

    prediction = model.predict(preprocess_df)

    if st.button('Predict'):
        if prediction == 1:
            st.warning('Yes, the customer will terminate the service.')
        else:
            st.success('No, the customer is happy with Telco Services.')
    
    



if __name__ == "__main__":
    main()

