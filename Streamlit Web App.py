
import numpy as np
import pandas as pd
import pickle
import streamlit as st

#Loading Model And Feature Extractor

Spam_Filter = pickle.load(open('Spam_Filtration.sav', 'rb'))
Ft_Extract = pickle.load(open('FtExtractor.sav', 'rb'))
#Creating a Function for Prediction

def Spam_Filtration(input_mail):
    input_mail = [input_mail]
    #converting to feature vectors

    input_features = Ft_Extract.transform(input_mail)

    #making Prediction

    prediction = Spam_Filter.predict(input_features)

    if prediction[0]==0:
        return "It is not a Spam Mail!"
    else:
        return "It is a Spam Mail!"

# Creating Web Interface

def main():
    
    
    #Giving Title
    st.title('Spam Mail Detection System')
    
    #Getting Input from User
    Mail = st.text_input('Enter the message')
    
    #code for prediction
    prediction  = ""
    
    #Creating a button for Output
    if st.button('Check Mail'):
        prediction = Spam_Filtration(Mail)
    
    st.success(prediction)
    
if __name__ == '__main__':
    main()




