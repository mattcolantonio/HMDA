#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:59:48 2023

@author: yeshimonipede
"""


#%% Load packages

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#%% Loading the data 

import os
os.getcwd()
path = '/Users/yeshimonipede/Documents/GitHub/HMDA/hmda_and_census.csv'
#os.chdir(path)

import sys
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    # in a virtual environment
    print(f"You are in a virtual environment: {sys.prefix}")
else:
    # not in a virtual environment
    print("You are not in a virtual environment.")

#df = pd.read_csv('hmda_and_census.csv')
df = pd.read_csv(path)


# Looking at the columns 

print(df.columns)

# what are we working with

# Creating a summary of statistics 

info = df.info()
all_stats = df.describe(include = 'all')

#%% Which variables are most important to keep ?

# dependent (y) variable --> 'action_taken' e.g., loan originated or denied

# predictors (X matrix) --> applicant info and census tract demographics

# Creating a list for all of our variables: 
    
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

'''
some variables are coded, and to interpret results from any models 
we decide to run we should be aware of the codes.

'''
'''
 if variables included in the new_df object are deemed unecessary, 
 they can be removed from the variables_to_keep list
'''

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

# Print or display the results

print("\nDescriptive Statistics for Object-Type Columns:")
print(object_stats)

print("\nUnique Values for Object-Type Columns:")
for column, values in object_unique_values.items():
    print(f"{column}: {values}")

print("\nValue Counts for Object-Type Columns:")
for column, counts in object_value_counts.items():
    print(f"{column}:\n{counts}")

#%% Label Encoding (Transforming to numeric format)

# Create a new DataFrame to store the converted data

from sklearn.preprocessing import LabelEncoder
hmda_encoded = new_df.copy()

# Apply label encoding to each object-type column

label_encoder = LabelEncoder()
for column in hmda_encoded.select_dtypes(include='O').columns:
    hmda_encoded[column] = label_encoder.fit_transform(hmda_encoded[column])

# Now, 'hmda_encoded' contains the converted data for use in analysis

hmda_encoded.info()



#%% Creating dummys variables for our data 

# Creating a new copy just in case
practice_df = hmda_encoded.copy()


# List of categorical variables
categorical_vars = ["derived_ethnicity", "derived_race", "derived_sex", "loan_type", "applicant_age", "derived_dwelling_category"  ] 

# Create dummy variables for each categorical variable

for var in categorical_vars:
    dummies = pd.get_dummies(practice_df[var], prefix=var, drop_first=True)
    practice_df = pd.concat([practice_df, dummies], axis=1)
    practice_df.drop(var, axis=1, inplace=True)  # Drop original column after creating dummies

#%% Futher Cleaning of our data 

nan_columns = practice_df.isna().any()

print("Columns with NaN values:", nan_columns)

nan_count = practice_df.isna().sum()

print("Number of NaN values in each column:\n", nan_count)

# Cleaning the Nan values (Not sure if this is the best way)
#I tried replacing NaN values with medians instead but accuracy decreased - yesh

practice_df_clean = practice_df.dropna()


#%% Logistic model


# Define features (X) and target variable (y)
X=practice_df_clean.drop('action_taken',axis=1)
y=practice_df_clean['action_taken']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the model on the training set
log_model=LogisticRegression()

#Fit the model
log_model.fit(X_train, y_train)


#Make predictions on the test set
y_pred=log_model.predict(X_test)

#Evaluate the model
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

