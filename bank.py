# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:07:22 2023

@author: Manasi Pawar
"""

import pickle
import pandas as pd
import streamlit as st
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Load the dataset
data=pd.read_excel("bankruptcy-prevention.xlsx")


label_encoder = preprocessing.LabelEncoder()
data[' class'] = label_encoder.fit_transform(data[' class'])


# Split the data into features (X) and target (y)
x = data.drop(' class', axis=1)
y = data[' class']
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Create the AdaBoost classifier
kfold=KFold(n_splits=5,random_state=72,shuffle=True)
model_ab= AdaBoostClassifier(n_estimators=60, random_state=8)        
result_ab = cross_val_score(model_ab, x, y, cv=kfold)
#Accuracy
print(result_ab.mean())



filename = 'final_Adaboost_model.pkl'
pickle.dump(model_ab, open(filename,'wb'))
model_ab.fit(x,y)
pk=model_ab.predict(x_test)


Industrial_risk = st.selectbox('industrial_risk', data['industrial_risk'].unique())
Management_risk = st.selectbox(' management_risk', data[' management_risk'].unique())
Financial_flexibility = st.selectbox(' financial_flexibility', data[' financial_flexibility'].unique())
Credibility = st.selectbox(' credibility', data[' credibility'].unique())
Competitiveness = st.selectbox(' competitiveness', data[' competitiveness'].unique())
Operating_risk = st.selectbox(' operating_risk', data[' operating_risk'].unique())


if st.button('Prevention Type'):
    df = {
        'industrial_risk': Industrial_risk,
        ' management_risk': Management_risk,
        ' financial_flexibility': Financial_flexibility,
        ' credibility': Credibility,
        ' competitiveness': Competitiveness,
        ' operating_risk': Operating_risk
    }

    df1 = pd.DataFrame(df, index=[1])
    predictions = model_ab.predict(df1)

    if predictions.any() == 1:
        prediction_value = 'Non-Bankruptcy'
    else:
        prediction_value = 'Bankruptcy'

    st.title("Business type is " + str(prediction_value))
