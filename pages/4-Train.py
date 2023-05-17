# Import necessary libraries
import json
import joblib

import pandas as pd
import streamlit as st

# Machine Learning 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# Custom classes 
from functions.utils import isNumerical
import os
import pickle
import numpy as np





#Loading Default Model Parameters
try:
    with open('data/metadata/model_params.json') as file:
        data = json.load(file)
        default_X = data['X']
        default_Y = data['y']
        default_pred_type = data['pred_type']
except:
    pass


####### DATA LOADING #################################################################

# Load the data 
if 'main_data.csv' not in os.listdir('data'):
    st.markdown("Please upload data through `Upload Data` page!")
else:
    st.markdown("## Model Building")
    data = pd.read_csv('data/main_data.csv')

    # Read the column meta data for this dataset 
    col_metadata = pd.read_csv('data/metadata/column_type_desc.csv')

    ''' The EDA section allows you to explore your dataset and gain insights 
        into its characteristics. Here, you can view basic statistics, 
        visualize the distribution of the data, and investigate relationships 
        between variables.
    '''
     
    
    

    data = pd.read_csv('data/main_data.csv')
    
    
    ####### VARIABLE SELECTION #################################################################
    
    st.markdown("""---""")
    st.markdown("### Select the Model Variables")   

    # Create the model parameters dictionary 
    params = {}

    # Use two column technique 
    col1, col2 = st.columns(2)
    

    # Design column 1 
    columns_for_index = data.columns.tolist()
    index = columns_for_index.index(default_Y)
    y_var = col2.selectbox("Select the target variable (y)", options=data.columns, index=index)

    # Design column 2 
    X_var = col1.multiselect("Select the features to be used for prediction (X)", options=data.columns)

    # Check if len of x is not zero 
    if len(X_var) == 0:
        st.warning("You have to put in some X variable and it cannot be left empty.")

    # Check if y not in X 
    if y_var in X_var:
        st.error("Warning! Y variable cannot be present in your X-variable.")
    
    # Create the model parameters dictionary
    X = data[X_var]
    y = data[y_var]

    
    # Perform encoding
    try:
        X = pd.get_dummies(X)
        
            
                # Check if y needs to be encoded
        if not isNumerical(y):
            st.markdown("##### Classification Classes:") 
            le = LabelEncoder()
            y = le.fit_transform(y)
            pred_type = "Classification"
            
            # Print all the classes 
            classes = list(le.classes_)
            columns = st.columns(len(classes))
            for i in range(len(classes)):
                with columns[i]:
                    st.markdown(f"**{classes[i]}** = {i}")
                
        else:
            pred_type = "Regression"
    
        # Add to model parameters 
        params = {
                'X': X_var,
                'y': y_var, 
                'pred_type': pred_type,
        }


    
        ####### TRAIN TEST SPLIT #################################################################
    
        st.markdown("""---""")

        # Perform train test splits 
        st.markdown("### Train Test Split")
        size = st.slider("Percentage of value division",
                            min_value=0.1, 
                            max_value=0.95, 
                            step = 0.05, 
                            value=0.8, 
                            help="This is the value which will be used to divide the data for training and testing. Default = 80%")

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=42)
        st.write("Number of training samples:", X_train.shape[0])
        st.write("Number of testing samples:", X_test.shape[0])

            
            
        ####### SELECT MODELS #################################################################
        st.markdown("""---""")    

        
        st.markdown("### Model Selection")
        
        #''' REGRESSION MODELS '''
        
        if pred_type == "Regression":
            #Model Selection
            regression_models = ['Linear Regression', 'Decision Tree Regression', 'Ridge Regression', 'Lasso Regression', 'Polynomial Regression']
            models_to_train = st.multiselect("Select the model/s to be used for regression", regression_models)
        
            #Hyperparameter selection
            st.markdown("#### Hyperparameter Selection")
            
            if 'Ridge Regression' in models_to_train or 'Lasso Regression' in models_to_train:
                st.write("Ridge & Lasso Regression Hyperparameters")
                alpha =  st.slider("Alpha", min_value=0.1, max_value=10.0, step = 0.1, value=0.1, help="Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates. Larger values specify stronger regularization. Alpha corresponds to C^-1 in other linear models such as LogisticRegression or LinearSVC.")   
            
            
            if 'Polynomial Regression' in models_to_train:
                st.write("Polynomial Regression Hyperparameters")
                degrees = st.slider("Degree", min_value=1, max_value=10, step = 1, value=2, help="Degree of the polynomial features. The default degree is 2.")
            
            #Model Training
            st.markdown("#### Train the Model")
            model_results = []
            
            if 'Linear Regression' in models_to_train:
                # Linear regression model 
                lr_model = LinearRegression()
                lr_model.fit(X_train, y_train)
                r2 = lr_model.score(X_test, y_test)
                mse = np.sqrt(mean_squared_error(y_test, lr_model.predict(X_test))).round(2)
                model_results.append(['Linear Regression', r2, mse])

            if 'Decision Tree Regression' in models_to_train:
                # Linear regression model 
                dt_model = DecisionTreeRegressor(random_state=0)
                dt_model.fit(X_train, y_train)
                r2 = dt_model.score(X_test, y_test)
                mse = np.sqrt(mean_squared_error(y_test, dt_model.predict(X_test))).round(2)
                model_results.append(['Decision Tree Regression', r2, mse])        

            if 'Ridge Regression' in models_to_train:
                # Linear regression model
                ridge = Ridge(alpha=alpha, random_state=0)
                ridge.fit(X_train, y_train)
                r2 = ridge.score(X_test, y_test)
                mse = np.sqrt(mean_squared_error(y_test, ridge.predict(X_test))).round(2)
                model_results.append(['Ridge', r2, mse]) 
            
            
            if 'Polynomial Regression' in models_to_train:
                # Linear regression model
                poly = PolynomialFeatures(degree=degrees)
                X_poly = poly.fit_transform(X_train)
                poly_model = LinearRegression()
                poly_model.fit(X_poly, y_train)
                
                x_test_poly = poly.transform(X_test)
                r2 = poly_model.score(x_test_poly, y_test)
                mse =  np.sqrt(mean_squared_error(y_test, poly_model.predict(x_test_poly))).round(2)
                model_results.append(['Poly', r2, mse])
                
            if 'Lasso Regression' in models_to_train:
            # Linear regression model
                lasso = Lasso(alpha=alpha)
                lasso.fit(X_train, y_train)
                r2 = lasso.score(X_test, y_test)
                mse = np.sqrt(mean_squared_error(y_test, lasso.predict(X_test))).round(2)
                model_results.append(['Lasso', r2, mse])      
                
            

            
            # Make a dataframe of results 
            results = pd.DataFrame(model_results, columns=['Models', 'R2 Score', 'RMSE']).sort_values(by='R2 Score', ascending=False)
            st.dataframe(results)
            
            model_to_save = st.selectbox('Select Model', results['Models'])

                    
            
            if st.button("Save Model"):
                if model_to_save == 'Linear Regression':
                    pickle.dump(lr_model, open("data/metadata/model.pkl", "wb"))
                    st.success(f'{model_to_save} Model Saved')  
                if model_to_save == 'Decision Tree Regression':
                    pickle.dump(dt_model, open("data/metadata/model.pkl", "wb"))
                    st.success(f'{model_to_save} Model Saved')  
                if model_to_save == 'Ridge':
                    pickle.dump(ridge, open("data/metadata/model.pkl", "wb"))
                    st.success(f'{model_to_save} Model Saved')  
                if model_to_save == 'Lasso':
                    pickle.dump(lasso, open("data/metadata/model.pkl", "wb"))
                    st.success(f'{model_to_save} Model Saved')  
                if model_to_save == 'Poly':
                    pickle.dump(poly_model, open("data/metadata/model.pkl", "wb"))
                    st.success(f'{model_to_save} Model Saved')  
                
            
        #''' CLASSIFICATION MODELS '''
        
        # Initialization

        
        if pred_type == "Classification":
            
                    #Loading Default Model Parameters
            try:
                with open('data/metadata/models_to_train.json') as file:
                    default_models = json.load(file)
            except:
                pass
            
                
            
            #Model Selection
            classification_models = ['Logistic Regression', 'Decision Tree Classificator', 'k-Nearest Neighbors (kNN)', 'Support Vector Machines (SVM)', 'Naive Bayes']
            try:
                models_to_train = st.multiselect("Select the model/s to be used for classification", classification_models)
            except:
                models_to_train = st.multiselect("Select the model/s to be used for classification", classification_models)

            
            # Save the models to train as a json file
            with open('data/metadata/models_to_train.json', 'w') as json_file:
                json.dump(models_to_train, json_file)
            
            # Table to store model and accurcy 
            model_acc = []
            
            
            if 'Logistic Regression' in models_to_train:
                # Logitic regression model 
                lc_model = LogisticRegression()
                lc_model.fit(X_train, y_train)
                acc = lc_model.score(X_test, y_test)
                model_acc.append(['Logistic Regression', acc])
                
            if 'Decision Tree Classificator' in models_to_train:
                # Decision Tree model 
                dtc_model = DecisionTreeClassifier()
                dtc_model.fit(X_train, y_train)
                acc = dtc_model.score(X_test, y_test)
                model_acc.append(['Decision Tree Classification', acc])
                
            if 'k-Nearest Neighbors (kNN)' in models_to_train:
                # Decision Tree model 
                knn_model = KNeighborsClassifier()
                knn_model.fit(X_train, y_train)
                acc = knn_model.score(X_test, y_test)
                model_acc.append(['k-Nearest Neighbors (kNN)', acc])
                
            if 'Support Vector Machines (SVM)' in models_to_train:
                # Decision Tree model 
                svm_model = SVC(probability=True)
                svm_model.fit(X_train, y_train)
                acc = svm_model.score(X_test, y_test)
                model_acc.append(['Support Vector Machines (SVM)', acc])
                
                
            if 'Naive Bayes' in models_to_train:
                # Decision Tree model 
                nb_model = GaussianNB()
                nb_model.fit(X_train, y_train)
                acc = nb_model.score(X_test, y_test)
                model_acc.append(['Naive Bayes', acc])
                
            results = pd.DataFrame(model_acc, columns=['Models', 'Accuracy']).sort_values(by='Accuracy', ascending=False)
            st.dataframe(results)
            
            model_to_save = st.selectbox('Select Model', results['Models'])

                    
            
            if st.button("Save Model"):
                
                model_path = "data/metadata/model.pkl"

                # Check if the file exists
                if os.path.exists(model_path):
                    # Delete the file
                    os.remove(model_path)

                
                if model_to_save == 'Logistic Regression':
                    pickle.dump(lc_model, open("data/metadata/model.pkl", "wb"))
                    st.success(f'{model_to_save} Model Saved')  
                if model_to_save == 'Decision Tree Classification':
                    pickle.dump(dtc_model, open("data/metadata/model.pkl", "wb"))
                    st.success(f'{model_to_save} Model Saved')
                if model_to_save == 'k-Nearest Neighbors (kNN)':
                    pickle.dump(knn_model, open("data/metadata/model.pkl", "wb"))
                    st.success(f'{model_to_save} Model Saved')
                if model_to_save == 'Support Vector Machines (SVM)':
                    pickle.dump(svm_model, open("data/metadata/model.pkl", "wb"))
                    st.success(f'{model_to_save} Model Saved')
                if model_to_save == 'Naive Bayes':
                    pickle.dump(nb_model, open("data/metadata/model.pkl", "wb"))
                    st.success(f'{model_to_save} Model Saved')
        
            # Save the model params as a json file
            with open('data/metadata/saved_model.json', 'w') as json_file:
                json.dump(model_to_save, json_file)
                
            with open('data/metadata/model_params.json', 'w') as json_file:
                json.dump(params, json_file) 

        
        
        
    except:
        pass

