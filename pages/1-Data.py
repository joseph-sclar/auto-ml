import collections
from numpy.core.defchararray import lower
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from functions import utils, extras
import os
    
st.markdown("## Data Upload")

# Upload the dataset and save as cs
st.markdown("### Upload a csv file for analysis.") 
st.write("\n")

# Code to read a single file 
uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])
global data
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        data = pd.read_excel(uploaded_file)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        sample_data = load_iris()
        X = pd.DataFrame(sample_data.data, columns=sample_data.feature_names) 
        Y = pd.Series(sample_data.target, name='class')
        sample_data = pd.concat( [X,Y], axis=1 )
    
        
        sample_data.to_csv('data/main_data.csv', index=False)

        # Collect the categorical and numerical columns 
        
        numeric_cols = sample_data.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = list(set(list(sample_data.columns)) - set(numeric_cols))
        
        # Save the columns as a dataframe or dictionary
        columns = []

        # Iterate through the numerical and categorical columns and save in columns 
        columns = utils.genMetaData(sample_data) 
        
        # Save the columns as a dataframe with categories
        # Here column_name is the name of the field and the type is whether it's numerical or categorical
        columns_df = pd.DataFrame(columns, columns = ['column_name', 'type'])
        columns_df.to_csv('data/metadata/column_type_desc.csv', index = False)
    
    
            
        


''' Load the data and save the columns with categories as a dataframe. 
This section also allows changes in the numerical and categorical columns. '''
if uploaded_file is not None and st.button("Load Data"):
    
    # Raw data 
    data.to_csv('data/main_data.csv', index=False)

    # Collect the categorical and numerical columns 
    
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = list(set(list(data.columns)) - set(numeric_cols))
    
    # Save the columns as a dataframe or dictionary
    columns = []

    # Iterate through the numerical and categorical columns and save in columns 
    columns = utils.genMetaData(data) 
    
    # Save the columns as a dataframe with categories
    # Here column_name is the name of the field and the type is whether it's numerical or categorical
    columns_df = pd.DataFrame(columns, columns = ['column_name', 'type'])
    columns_df.to_csv('data/metadata/column_type_desc.csv', index = False)
    
if 'main_data.csv' not in os.listdir('data'):
    st.markdown("Please upload data through `Upload Data` page!")
else:
    
    data = pd.read_csv('data/main_data.csv')
    # Raw data 
    st.dataframe(data)
    data.to_csv('data/main_data.csv', index=False)

    # Collect the categorical and numerical columns 
    
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = list(set(list(data.columns)) - set(numeric_cols))
    
    # Save the columns as a dataframe or dictionary
    columns = []

    # Iterate through the numerical and categorical columns and save in columns 
    columns = utils.genMetaData(data) 
    
    # Save the columns as a dataframe with categories
    # Here column_name is the name of the field and the type is whether it's numerical or categorical
    columns_df = pd.DataFrame(columns, columns = ['column_name', 'type'])
    columns_df.to_csv('data/metadata/column_type_desc.csv', index = False) 
    
# Add a button to navigate to another page
if st.button("Next"):
    extras.switch_page("metadata")    