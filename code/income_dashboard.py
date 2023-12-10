# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:48:14 2023

@author: geean
"""

#%% stratified income dashboard

#%% import packages
import dashboard_fns as models
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


#%% import data
csv_file_path = 'C:/Users/geean/Documents/2023Fall_ADEC743001/MLAlgorithms1Prj/HMDA/hmda_and_census/hmda_and_census.csv'
output_directory = 'C:/Users/geean/Documents/2023Fall_ADEC743001/MLAlgorithms1Prj/HMDA/hmda_and_census/'
cleaned_df = models.cleaningday(csv_file_path, output_directory)

#%% split data
X_train, X_test, y_train, y_test = models.split_data(cleaned_df)

# %% Stratifying to account for Income
st.title(" ROC Curves Stratifying to Account for Income")
# set this to get rid of the warning on the dashboard for the random forest diagram
st.set_option('deprecation.showPyplotGlobalUse', False)
# Add this line to stratify cleaned_df into quintiles based on the 'income' column
cleaned_df['income_group'] = pd.qcut(cleaned_df['income'], q=5, labels=False)
# Now 'income_group' column represents the quintiles, 0 to 4

# we have geographic features- these are identifiers only
exclude_columns = ['census_tract', 'derived_msa.md', 'county_code']

# Split data within each income stratum
stratified_splits = {}
for income_group in cleaned_df['income_group'].unique():
    income_df = cleaned_df[cleaned_df['income_group'] == income_group]
    X_train, X_test, y_train, y_test = models.split_data(
        income_df, exclude_columns=exclude_columns)
    stratified_splits[income_group] = (X_train, X_test, y_train, y_test)
    
# stratified_splits is a dictionary-like structure where keys are income groups, 
#and values are tuples containing training and testing data.
# Sidebar for user input - Model
st.sidebar.title("Select Model")
selected_model = st.sidebar.selectbox(
    "Select Model:",
    ["Logistic Regression", "Random Forest"]
)

# Sidebar for user input - Income Stratum
st.sidebar.title("Select Income Stratum")
selected_income_stratum = st.sidebar.selectbox(
    "Select Income Stratum:",
    cleaned_df['income_group'].unique()
)

# Split data within the selected income stratum
selected_stratum_data = stratified_splits[selected_income_stratum]
X_train, X_test, y_train, y_test = selected_stratum_data

# Display model-specific information
if selected_model == "Logistic Regression":
    st.header(f"Logistic Regression - Income Stratum: {selected_income_stratum}")
    st.subheader("Model Coefficients")
    log_model = models.train_and_evaluate_logistic_regression(
        X_train, X_test, y_train, y_test
    )
    coefficients = log_model.coef_[0]
    feature_names = X_train.columns
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    st.write(coef_df)

    # ROC curve for Logistic Regression
    st.subheader("ROC Curve - Logistic Regression")
    models.plot_roc_curve(log_model, X_test, y_test, 'Logistic Regression')
    st.pyplot()

elif selected_model == "Random Forest":
    st.header(f"Random Forest - Income Stratum: {selected_income_stratum}")
    st.subheader("Random Forest Tree Visualization")
    rf_model = models.train_and_evaluate_random_forest(X_train, X_test, y_train, y_test)
    models.visualize_random_forest_tree(rf_model, X_train)
    st.pyplot()

    st.subheader("Random Forest Feature Importance")
    models.visualize_feature_importance(rf_model, X_train)
    st.pyplot()

    # ROC curve for Random Forest
    st.subheader("ROC Curve - Random Forest")
    models.plot_roc_curve(rf_model, X_test, y_test, 'Random Forest')
    st.pyplot()
    
    # in cmd or terminal run streamlit run income_dashboard.py