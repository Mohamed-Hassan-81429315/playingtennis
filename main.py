import streamlit as st 
import pickle 
import joblib
import pandas as pd

model = joblib.load('model.pkl')

with open ('scaler.pkl', 'rb')  as scaler_file :
    scaler  = pickle.load(scaler_file)

st.title(f"Playing Tennis Prediction\nThis is a form that tells you to safely play tennis or not based on the data that you will fill \n\n Now please fill this form with correct data\n\n ")
day = st.number_input("Type The number of today look like \'10/12/2020\' then number is 10", min_value= 1, max_value=31)
Outlook = st.selectbox("The state of the weather now" , options=['Overcast' ,'Sunny' , 'Rainy'])
Outlook = int(1) if Outlook =='Overcast' else  int(2) if Outlook =='Sunny'  else int(0) if Outlook == 'Rainy' else int(3)
temperature =  st.number_input("The Temperature of the area where you live" , step= 1 ) 
Temperature = int(0)  if temperature < 10 else int(1)  if temperature < 35 else int(2)
humidity  = st.selectbox("Is the weather wet/humid now ?" , options=["Yes" , "No"] )
Humidity = int(1) if humidity == "Yes" else int(0)
wind  = st.selectbox("Does the weather contains any winds now ?" , options=["Yes" , "No"] )
Wind = int(1) if wind == "Yes" else int(0) 

st.write("After you fill the form , you can press on the button below to see if you can safely play tennis or not")


if st.button("Can I play tennis now ?" , icon="🎯" , type='primary' ) :
    data = pd.DataFrame([{
       "Day"  : day ,
       "Outlook" : Outlook ,
       "Temperature" : Temperature ,
       "Humidity" : Humidity ,
       "Wind" : Wind
     }])
    data = scaler.transform(data)
    prediction = model.predict(data)
    result = ""
    if prediction == 1 :
        result = "🎯Now You can play tennis safely"
    else :
        result = "🤦‍♂️ Unfortunately , You can not play tennis safely now"
    st.success(f"{result}")