# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 15:32:11 2023

@author: geean
"""

#%% cleaned_df data
import os
os.getcwd()
# path="/Users/matthewcolantonio/Documents/Research/HMDA"
# os.chdir(path)
# want to make sure the virtual env is activated to avoid package version confusion amongst the team
import sys
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    # in a virtual environment
    print(f"You are in a virtual environment: {sys.prefix}")
else:
    # not in a virtual environment
    print("You are not in a virtual environment.")
    
#%% Load and explore data
import pandas as pd
import datetime

csv_file_path = 'C:/Users/geean/Documents/2023Fall_ADEC743001/MLAlgorithms1Prj/HMDA/hmda_and_census/hmda_and_census.csv'
df = pd.read_csv(csv_file_path)
# what are we working with
info = df.info()
all_stats = df.describe(include = 'all')

#%% Which variables are most important to keep ?
# dependent (y) variable --> 'action_taken' e.g., loan originated or denied or 'other'
# predictors (X matrix) --> applicant info and census tract demographics
column_names = df.columns.tolist()
variables_to_keep = ["census_tract", "derived_msa.md", "county_code", "derived_loan_product_type",
                     "derived_dwelling_category", "derived_ethnicity", "derived_race", "derived_sex",
                     "action_taken", "purchaser_type", "loan_type", "loan_purpose", "reverse_mortgage", 
                     "loan_amount", "loan_to_value_ratio", "property_value", "occupancy_type", "income",
                     "debt_to_income_ratio", "applicant_credit_score_type", "applicant_age", 
                     "tract_population", "tract_minority_population_percent", "ffiec_msa_md_median_family_income",
                     "tract_to_msa_income_percentage", "tract_owner_occupied_units", "tract_median_age_of_housing_units",
                     "Unemployment_Rate", "Median_HH_Income", "Vacant_Housing_Units_Percentage",
                     "Tenure_Owner_Occupied_Percentage", "Median_Value_Owner_Occupied", "Median_Gross_Rent",
                     "Total_Population_White_Percentage", "Total_Population_Black_Percentage"]
new_df = df[variables_to_keep]
new_df.info() 
# some variables are coded, and to interpret results from any models we decide to run we should be aware of the codes
# if variables included in the new_df object are deemed unecessary, they can be removed from the variables_to_keep list

#%%
# reminder: object = Text or mixed numeric and non-numeric values
# float64 = Floating point numbers
# to facilitate analysis, we should understand what we have in terms of object values

# Get descriptive statistics for object-type columns
object_stats = new_df.describe(include='O')

# Get unique values and value counts for each object-type column
object_unique_values = {}
object_value_counts = {}

for column in new_df.select_dtypes(include='O').columns:
    object_unique_values[column] = new_df[column].unique()
    object_value_counts[column] = new_df[column].value_counts()
#%% 
# different data manipulations need to be done based on the type of object
cleaned_df = new_df.dropna() # a lot of data is categorical, imputing won't do us any good here
# action_taken is target. 8 values won't work for logit, so we are reducing to accepted, denied, or 'other'
action_taken_mapping = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3, 7:3, 8:3}
cleaned_df['action_taken'] = cleaned_df['action_taken'].replace(action_taken_mapping)
print(cleaned_df['action_taken'].value_counts()) # verify

# ie, a model predicting every applicant will be accepted will be 80% accurate- our model needs to be better than this rate at least

cleaned_df['loan_to_value_ratio'] = pd.to_numeric(cleaned_df['loan_to_value_ratio'], errors='coerce') # coerce creates NaN for values that can't be converted to numeric
print(cleaned_df['loan_to_value_ratio'].dtype) # verify
cleaned_df['property_value'] = pd.to_numeric(cleaned_df['property_value'], errors='coerce')
print(cleaned_df['property_value'].dtype)

#debt_to_income_ratio needs some work

print(" Diego Original unique values:\n", cleaned_df['debt_to_income_ratio'].unique())
# Removing special characters
cleaned_df['debt_to_income_ratio'] = cleaned_df['debt_to_income_ratio'].replace('[%<>]', '', regex=True)
print("Diego After removing special characters:\n", cleaned_df['debt_to_income_ratio'].unique())

# Replace 'Exempt' with NaN (you can use 0 or any other value as needed)
cleaned_df['debt_to_income_ratio'] = cleaned_df['debt_to_income_ratio'].replace('Exempt', pd.NA)
print("Diego After replacing 'Exempt':\n", cleaned_df['debt_to_income_ratio'].unique())
# Convert ranges to average values

# Diego: I think here is the error. 
# cleaned_df['debt_to_income_ratio'] = cleaned_df['debt_to_income_ratio'].apply(lambda x: eval(x.replace('-', '+')) / 2 if pd.notna(x) else x)

def convert_range_to_average(value):
    if '-' in str(value):
        lower, upper = value.split('-')
        return (float(lower) + float(upper)) / 2
    return value

# Convert ranges to average values
cleaned_df['debt_to_income_ratio'] = cleaned_df['debt_to_income_ratio'].apply(convert_range_to_average)
# Convert to float, handling missing values
cleaned_df['debt_to_income_ratio'] = pd.to_numeric(cleaned_df['debt_to_income_ratio'], errors='coerce')
print("After converting to average values and float:\n", cleaned_df['debt_to_income_ratio'].unique())

cleaned_df = cleaned_df.dropna()
print("After converting to average values and float:\n", cleaned_df['debt_to_income_ratio'].unique())
# Let's call it an imputation- we will mention in our final analysis. ie, we keep the averages as imputed debt-to-income 


# Creating Dummy variables for categorical variables
# dummies are best since order doesn't matter (as opposed to label encoding)
categorical_vars = ["derived_ethnicity", "derived_race", "derived_sex", "derived_loan_product_type",  "applicant_age", "derived_dwelling_category"  ] 

for var in categorical_vars:
    dummies = pd.get_dummies(cleaned_df[var], prefix=var, drop_first=True)
    cleaned_df = pd.concat([cleaned_df, dummies], axis=1)
    cleaned_df.drop(var, axis=1, inplace=True)  # Drop original column after creating dummies

# new columns names
print(cleaned_df.columns) # it looks like some weird values made it thru (eg applicant_age_8888)- these won't help in modeling
# 'True' values for any of the following are useless to us (essentially = to null values)
dummy_columns_to_check = [
    'derived_ethnicity_Free Form Text Only',
    'derived_race_Free Form Text Only',
    'derived_sex_Sex Not Available',
    'applicant_age_8888',
    'derived_race_Joint',
    'derived_sex_Joint',
    'derived_race_Race Not Available'
]

# Step 1: Remove rows where any of the specified dummy columns has a value
cleaned_df = cleaned_df[~cleaned_df[dummy_columns_to_check].any(axis=1)]
# Step 2: Remove the specified dummy columns
cleaned_df = cleaned_df.drop(columns=dummy_columns_to_check)
# Optional: Reset the index if needed
cleaned_df = cleaned_df.reset_index(drop=True)

print(cleaned_df.columns) # looks better

# For dummies, it is important to know the reference group- coefficents from models on dummies are coefficents relative to other groups
# dummy variables for categorical_vars
#%% my dashboard code
# =============================================================================
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# 
# # Load your DataFrame
# # Assuming you have a DataFrame named 'cleaned_df'
# # You should replace this with the actual name of your DataFrame
# # cleaned_df = pd.read_csv("your_data.csv")
# 
# # Streamlit app
# st.title("Interactive Line Chart Dashboard")
# 
# # Sidebar for user input
# selected_x = st.sidebar.selectbox("Select X-axis:", cleaned_df.columns)
# selected_y = st.sidebar.selectbox("Select Y-axis:", cleaned_df.columns)
# selected_color = st.sidebar.selectbox("Select Color:", cleaned_df.columns)
# 
# # Line chart
# fig = px.line(
#     cleaned_df,
#     x=selected_x,
#     y=selected_y,
#     color=selected_color,
#     hover_data=['applicant_age_35-44', 'applicant_age_45-54', 'applicant_age_55-64', 'applicant_age_65-74', 'applicant_age_<25', 'applicant_age_>74'],  # Add additional columns to hover over
#     title="Line Chart",
# )
# 
# # Display the plot
# st.write(fig)
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# 
# # Assuming you have a DataFrame named 'cleaned_df'
# # You should replace this with the actual name of your DataFrame
# # cleaned_df = pd.read_csv("your_data.csv")
# 
# # Streamlit app
# st.title("Action Taken Comparison by Race")
# 
# # Sidebar for user input
# selected_race = st.sidebar.selectbox("Select Race:", cleaned_df[['derived_ethnicity_Hispanic or Latino', 'derived_ethnicity_Joint',
#        'derived_ethnicity_Not Hispanic or Latino',
#        'derived_race_American Indian or Alaska Native', 'derived_race_Asian',
#        'derived_race_Black or African American',
#        'derived_race_Native Hawaiian or Other Pacific Islander',
#        'derived_race_White']].columns.unique())
# 
# # Filter DataFrame based on selected race
# filtered_df = cleaned_df[cleaned_df[selected_race].notnull()]
# 
# # Bar chart
# fig = px.bar(
#     filtered_df,
#     x='action_taken',
#     title=f"Action Taken Comparison for {selected_race}",
# )
# 
# # Display the plot
# st.write(fig)
# =============================================================================
#%% filter by race

# =============================================================================
# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# 
# # Assuming you have a DataFrame named 'cleaned_df'
# # cleaned_df = pd.read_csv("your_data.csv")
# 
# # Streamlit app
# st.title("Action Taken Distribution by Ethnicity and Race")
# 
# # Sidebar for user input
# selected_column = st.sidebar.selectbox(
#     "Select Column:",
#     [
#         'derived_ethnicity_Hispanic or Latino',
#         'derived_ethnicity_Joint',
#         'derived_ethnicity_Not Hispanic or Latino',
#         'derived_race_American Indian or Alaska Native',
#         'derived_race_Asian',
#         'derived_race_Black or African American',
#         'derived_race_Native Hawaiian or Other Pacific Islander',
#         'derived_race_White'
#     ]
# )
# 
# # Filter DataFrame based on selected column and where 'action_taken' is true
# filtered_df = cleaned_df[cleaned_df[selected_column] == True]
# 
# # Get the counts of each category of "action_taken"
# action_taken_counts = filtered_df['action_taken'].value_counts()
# 
# # Plot distribution of 'action_taken'
# fig, ax = plt.subplots(figsize=(8, 6))
# sns.countplot(x='action_taken', data=filtered_df, ax=ax)
# plt.title(f"Action Taken Distribution for {selected_column}")
# plt.xlabel("Action Taken")
# plt.ylabel("Count")
# 
# # Annotate each bar with its count
# for i, count in enumerate(action_taken_counts):
#     ax.text(i, count + 0.1, str(count), ha='center', va='bottom')
# 
# # Display the plot
# st.pyplot(fig)
# =============================================================================

#%% grouped version hope this works
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'cleaned_df'
# cleaned_df = pd.read_csv("your_data.csv")

# Streamlit app
st.title("Action Taken Distribution by Ethnicity and Race")

# Sidebar for user input
selected_column = st.sidebar.selectbox(
    "Select Column:",
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

# Create a grouped DataFrame for True and False values of the selected column
grouped_df = cleaned_df.groupby([selected_column, 'action_taken']).size().unstack()

# Plot the grouped bar plot
fig, ax = plt.subplots(figsize=(10, 6))
grouped_df.plot(kind='bar', stacked=True, ax=ax)
plt.title(f"Action Taken Distribution for {selected_column}")
plt.xlabel(selected_column)
plt.ylabel("Count")

# Annotate each bar with its count
for i, (index, row) in enumerate(grouped_df.iterrows()):
    for j, value in enumerate(row):
        ax.text(i + (j - 0.5) * 0.2, value + 0.1, str(value), ha='center', va='bottom')

# Display the plot
st.pyplot(fig)







# now in terminal or command prompt run starlight run my_dashboard.py in the virtual environment, make sure the file is in the virtual environment
