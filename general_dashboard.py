# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 15:32:11 2023

@author: geean
"""


#%% import the data

#%% import the functions from the script, make sure they are in the same directory
from dashboard_fns import cleaningday, data_summary, split_data, train_and_evaluate_logistic_regression, train_and_evaluate_random_forest, plot_roc_curve, visualize_random_forest_tree, visualize_feature_importance
import streamlit as st
import pandas as pd




#%% import data
csv_file_path = 'C:/Users/geean/Documents/2023Fall_ADEC743001/MLAlgorithms1Prj/HMDA/hmda_and_census/hmda_and_census.csv'
output_directory = 'C:/Users/geean/Documents/2023Fall_ADEC743001/MLAlgorithms1Prj/HMDA/hmda_and_census/'
cleaned_df = cleaningday(csv_file_path, output_directory)

#%% split data
X_train, X_test, y_train, y_test = split_data(cleaned_df)

#%% Streamlit app
st.title("Model Evaluation and Visualization Overall")

# set this to get rid of the warning on the dashboard for the random forest diagram
st.set_option('deprecation.showPyplotGlobalUse', False)

# Assuming you have a DataFrame named 'cleaned_df' 
X = cleaned_df.drop(['action_taken'], axis=1)
y = cleaned_df['action_taken']

# Logistic Regression
log_model = train_and_evaluate_logistic_regression(
    X_train, X_test, y_train, y_test,
    solver='liblinear',
    penalty='l1',
    C=1.0
)

# Random Forest
rf_model = train_and_evaluate_random_forest(X_train, X_test, y_train, y_test)


# Sidebar for user input - Model
st.sidebar.title("Select Model")
selected_model = st.sidebar.selectbox(
    "Select Model:",
    ["Logistic Regression", "Random Forest", "Correlation and Statistics"]
)

# Display model-specific information
if selected_model == "Logistic Regression":
    st.header("Logistic Regression")
    st.subheader("Model Coefficients")
    coef_df = pd.DataFrame(log_model.coef_, columns=X_train.columns)
    st.write(coef_df)
    
    # Add ROC curve for Logistic Regression
    st.subheader("ROC Curve - Logistic Regression")
    plot_roc_curve(log_model, X_test, y_test, 'Logistic Regression')
    st.pyplot()

# In the "Random Forest" section
elif selected_model == "Random Forest":
    st.header("Random Forest")
    st.subheader("Random Forest Tree Visualization")
    visualize_random_forest_tree(rf_model, X_train)
    st.pyplot()  

    st.subheader("Random Forest Feature Importance")
    visualize_feature_importance(rf_model, X_train)
    st.pyplot() 
    
    # Add ROC curve for Random Forest
    st.subheader("ROC Curve - Random Forest")
    plot_roc_curve(rf_model, X_test, y_test, 'Random Forest')
    st.pyplot() 
    
    
    # In the Correlation section
elif selected_model == "Correlation and Statistics":

    st.header("Correlation and Statistics")
    
    # Call the function to get the correlation plot and summary statistics
    correlation_data = data_summary(cleaned_df)

    # Display the summary statistics
    st.subheader("Summary Statistics")
    st.text(correlation_data['summary_statistics'])

    # Display the correlation plot using Streamlit
    st.subheader("Correlation Plot")
    st.pyplot(correlation_data['correlation_matrix']) # correlation_matrix is the matrix

    
# To show the dashboard, in command prompt or terminal of the virtual environment, type streamlit run best_my_dashboard.py
