# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 15:32:11 2023

@author: geean
"""


#%% import the functions from the script, make sure they are in the same directory
import anita_hmda_models_v03_fns as models
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


#%% import data
csv_file_path = 'C:/Users/geean/Documents/2023Fall_ADEC743001/MLAlgorithms1Prj/HMDA/hmda_and_census/hmda_and_census.csv'
output_directory = 'C:/Users/geean/Documents/2023Fall_ADEC743001/MLAlgorithms1Prj/HMDA/hmda_and_census/'
cleaned_df = models.cleaningday(csv_file_path, output_directory)

#%% split data
X_train, X_test, y_train, y_test = models.split_data(cleaned_df)

#%% Streamlit app
st.title("Model Evaluation and Visualization Overall")

# set this to get rid of the warning on the dashboard for the random forest diagram
st.set_option('deprecation.showPyplotGlobalUse', False)

# Assuming you have a DataFrame named 'cleaned_df' 
X = cleaned_df.drop(['action_taken'], axis=1)
y = cleaned_df['action_taken']

# Logistic Regression
log_model = models.train_and_evaluate_logistic_regression(
    X_train, X_test, y_train, y_test,
    solver='liblinear',
    penalty='l1',
    C=1.0
)

# Random Forest
rf_model = models.train_and_evaluate_random_forest(X_train, X_test, y_train, y_test)

# Sidebar for user input - Model
st.sidebar.title("Select Model")
selected_model = st.sidebar.selectbox(
    "Select Model:",
    ["Logistic Regression", "Random Forest"]
)

# Display model-specific information
if selected_model == "Logistic Regression":
    st.header("Logistic Regression")
    st.subheader("Model Coefficients")
    coef_df = pd.DataFrame(log_model.coef_, columns=X_train.columns)
    st.write(coef_df)
    
    # Add ROC curve for Logistic Regression
    st.subheader("ROC Curve - Logistic Regression")
    models.plot_roc_curve(log_model, X_test, y_test, 'Logistic Regression')
    st.pyplot()

# In the "Random Forest" section
elif selected_model == "Random Forest":
    st.header("Random Forest")
    st.subheader("Random Forest Tree Visualization")
    models.visualize_random_forest_tree(rf_model, X_train)
    st.pyplot()  

    st.subheader("Random Forest Feature Importance")
    models.visualize_feature_importance(rf_model, X_train)
    st.pyplot() 
    
    # Add ROC curve for Random Forest
    st.subheader("ROC Curve - Random Forest")
    models.plot_roc_curve(rf_model, X_test, y_test, 'Random Forest')
    st.pyplot() 

# To show the dashboard, in command prompt or terminal of the virtual environment, type streamlit run best_my_dashboard.py
