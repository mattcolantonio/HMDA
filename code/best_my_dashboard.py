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
st.title("Model Evaluation and Visualization")

# Sidebar for user input - Race
#st.sidebar.title("Select Race")

selected_race_column = st.sidebar.selectbox(
    "Select Race:",
    [
        'derived_ethnicity_Hispanic or Latino',
        'derived_ethnicity_Joint',
        'derived_ethnicity_Not Hispanic or Latino',
        'derived_race_American Indian or Alaska Native',
        'derived_race_Asian',
        'derived_race_Black or African American',
        'derived_race_Native Hawaiian or Other Pacific Islander',
        'derived_race_White'
    ]
)

# set this to get rid of the warning on the dashboard for the random forest diagram
st.set_option('deprecation.showPyplotGlobalUse', False)

# Assuming you have a DataFrame named 'cleaned_df' and you have already split your data into X_train, X_test, y_train, y_test
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

# In the "Random Forest" section
elif selected_model == "Random Forest":
    st.header("Random Forest")
    st.subheader("Random Forest Tree Visualization")
    models.visualize_random_forest_tree(rf_model, X_train)
    st.pyplot()  

    st.subheader("Random Forest Feature Importance")
    models.visualize_feature_importance(rf_model, X_train)
    st.pyplot() 

# To show the dashboard, in command prompt or terminal of the virtual environment, type streamlit run best_my_dashboard.py
# =============================================================================
#     # KL Divergence
#     st.header("KL Divergence")
#     kl_divergence_value = models.kl_divergence(log_model, rf_model, X_test)
#     st.write(f"KL Divergence Value: {kl_divergence_value}")
# =============================================================================
