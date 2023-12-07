# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:03:25 2023

@author: geean
"""

#%% race dashboard
#%% import packages
import anita_hmda_models_v03_fns as models
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


#%% import data
csv_file_path = 'C:/Users/geean/Documents/2023Fall_ADEC743001/MLAlgorithms1Prj/HMDA/hmda_and_census/hmda_and_census.csv'
output_directory = 'C:/Users/geean/Documents/2023Fall_ADEC743001/MLAlgorithms1Prj/HMDA/hmda_and_census/'
cleaned_df = models.cleaningday(csv_file_path, output_directory)

#%% split data
X_train, X_test, y_train, y_test = models.split_data(cleaned_df)

#%% dashboard/streamlit app

st.title("Count of Action Taken based on each Ethnicity is True vs. False")

# Sidebar for user input - Race
st.sidebar.title("Select Race")

selected_race_column = st.sidebar.selectbox(
    "Select Race:",
    [
        'derived_ethnicity_Hispanic or Latino',
        'derived_ethnicity_Not Hispanic or Latino',
        'derived_race_American Indian or Alaska Native',
        'derived_race_Asian',
        'derived_race_Black or African American',
        'derived_race_Native Hawaiian or Other Pacific Islander',
        'derived_race_White'
    ]
)

# Action Taken Distribution
st.title("Count of Action Taken based on each Ethnicity is True vs. False")

# Create a DataFrame with selected race column and action_taken from X_train and y_train
plot_df = pd.concat([X_train[[selected_race_column]], y_train], axis=1)

# Create a grouped DataFrame for True and False values of the selected race column
grouped_df = plot_df.groupby([selected_race_column, 'action_taken']).size().unstack()

# Plot the grouped bar plot
fig, ax = plt.subplots(figsize=(10, 6))
grouped_df.plot(kind='bar', stacked=True, ax=ax)
plt.title(f"Action Taken Distribution for {selected_race_column}")
plt.xlabel(selected_race_column)
plt.ylabel("Count")

# Annotate each bar with its count
for i, (index, row) in enumerate(grouped_df.iterrows()):
    for j, value in enumerate(row):
        ax.text(i+j  * 0.05, value + 0.05, str(value), ha='center', va='bottom')

# Display the plot
st.pyplot(fig)



# now run in cmd or terminal streamlit run race_dashboard.py