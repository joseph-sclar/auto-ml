import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import json
from sklearn.preprocessing import LabelEncoder


#custum functions
from functions.utils import isNumerical



try:
    with open('data/metadata/model_params.json') as file:
        data = json.load(file)
        default_X = data['X']
        default_Y = data['y']
        default_pred_type = data['pred_type']
        #st.write(default_X, default_Y, default_pred_type)
except:
    pass

try:
    with open('data/metadata/saved_model.json') as file:
        model_name = json.load(file)
except:
    pass




st.write(f"""
# Penguin Prediction App

This app predicts the **Palmer Penguin** species using **{model_name}**!

""")

st.sidebar.header('User Input Features')





data = pd.read_csv('data/main_data.csv')
col_metadata = pd.read_csv('data/metadata/column_type_desc.csv')
columns_df = pd.read_csv('data/metadata/column_type_desc.csv') 




def user_dinamic_input_features():
    features_dic = {}
    for i in range(columns_df.shape[0]):
            if columns_df.iloc[i]['type'] == "categorical" and columns_df.iloc[i]['column_name'] != default_Y and columns_df.iloc[i]['column_name'] in default_X:
                input = st.sidebar.selectbox(columns_df.iloc[i]['column_name'], data[columns_df.iloc[i]['column_name']].unique())
                features_dic.update({columns_df.iloc[i]['column_name']: input})
            if columns_df.iloc[i]['type'] == "numerical" and columns_df.iloc[i]['column_name'] != default_Y and columns_df.iloc[i]['column_name'] in default_X:
                input = st.sidebar.slider(columns_df.iloc[i]['column_name'], float(data[columns_df.iloc[i]['column_name']].min()* .5), float(data[columns_df.iloc[i]['column_name']].max() * 2), float(data[columns_df.iloc[i]['column_name']].mean()))
                features_dic.update({columns_df.iloc[i]['column_name']: input})
    features = pd.DataFrame(features_dic, index=[0])            
    return features 
user_dinamic_input_features = user_dinamic_input_features()  




# Encoding of ordinal features
data_to_encode = data.drop(columns=[default_Y], axis=1)
df = pd.concat([user_dinamic_input_features,data_to_encode],axis=0)
df = df.dropna(axis=1)


columns_to_encode = columns_df[(columns_df['column_name'] != default_Y) & (columns_df['column_name'].isin(default_X))]
encode = columns_to_encode[columns_to_encode['type'] == "categorical"]['column_name']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
feutures_to_predict = df[:1] # Selects only the first row (the user input data) 

# Displays the user input features
st.subheader('User Input features')

    
    
    
# Reads in saved classification model
#model = pickle.load(open('data/metadata/model.pkl', 'rb'))

# Load the pickled model from a file
with open('data/metadata/model.pkl', 'rb') as file:
    model = pickle.load(file)


st.dataframe(feutures_to_predict)

# Apply model to make predictions
prediction = model.predict(feutures_to_predict)
prediction_proba = model.predict_proba(feutures_to_predict)


#Getting Classes to Predict
y = data[default_Y]
le = LabelEncoder()
y = le.fit_transform(y)
classes = list(le.classes_)


st.subheader('Prediction')
penguins_species = np.array(classes)
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)    
    
    
    
