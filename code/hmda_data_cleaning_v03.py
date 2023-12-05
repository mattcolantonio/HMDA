#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:12:30 2023

@author: matthewcolantonio
"""

#%%
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

csv_file_path = '/Users/matthewcolantonio/Documents/Research/HMDA/saveddata/hmda_and_census.csv'
df = pd.read_csv(csv_file_path)
# what are we working with
info = df.info()
all_stats = df.describe(include = 'all')

#%% Which variables are most important to keep ? Only ones re: applicant info
# info on Loan means they were accepted (not helpful for predicting)
# dependent (y) variable --> 'action_taken' e.g., loan originated or denied or 'other'
# predictors (X matrix) --> applicant info and census tract demographics
column_names = df.columns.tolist()
variables_to_keep = ["census_tract", "derived_msa.md", "county_code", "derived_loan_product_type",
                     "derived_dwelling_category", "derived_ethnicity", "derived_race", "derived_sex",
                     "action_taken", "loan_type", "loan_purpose", "reverse_mortgage", 
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
#cleaned_df = new_df.dropna() # a lot of data is categorical, imputing won't do us any good here
# action_taken is target. 8 values won't work for logit, so we are reducing to accepted, denied, or 'other'
cleaned_df = new_df
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
# Let's call it an imputation- we will mention in our final analysis. ie, we keep the averages as imputed debt-to-income 


# Creating Dummy variables for categorical variables
# dummies are best since order doesn't matter (as opposed to label encoding)
categorical_vars = ["derived_ethnicity", "derived_race", "derived_sex", "derived_loan_product_type", "applicant_age", "derived_dwelling_category"]

values_to_remove = [
    'Free Form Text Only',
    'Sex Not Available',
    '8888',
    'Joint',
    'Race Not Available'
]

# Remove observations where any value in values_to_remove is present in any of the categorical variable columns
cleaned_df = cleaned_df[~cleaned_df[categorical_vars].isin(values_to_remove).any(axis=1)]

# Create dummies
for var in categorical_vars:
    dummies = pd.get_dummies(cleaned_df[var], prefix=var, drop_first=True)
    cleaned_df = pd.concat([cleaned_df, dummies], axis=1)
    cleaned_df.drop(var, axis=1, inplace=True)  # Drop original column after creating dummies


# new columns names (with dummies)
print(cleaned_df.columns) # looks better

cleaned_df = cleaned_df.dropna() # remaining nan should not be imputed
cleaned_df = cleaned_df[cleaned_df['action_taken'] != 3] # remove 'others' , there are <100



#%% Export cleaned data frame

# Specify the directory
output_directory = "/Users/matthewcolantonio/Documents/Research/HMDA/saveddata/"

# Create the directory if it doesn't exist
# os.makedirs(output_directory, exist_ok=True)
# Specify the file path
output_file_path = os.path.join(output_directory, "cleaned_data.csv")

# Export the DataFrame to CSV
cleaned_df.to_csv(output_file_path, index=False)
print(f"Data exported to: {output_file_path}")


#%%  Export as .shp for mapping or spatial analysis
import geopandas as gpd
from geodatasets import get_path

path_to_shape = "/Users/matthewcolantonio/Documents/Research/HMDA/rawdata/tl_2020_25_tract.zip" 
# yes, upload the entire zipfile
census_shape = gpd.read_file(path_to_shape)

print(cleaned_df['census_tract'].dtype)
print(census_shape['GEOID'].dtype)
census_shape['GEOID'] = pd.to_numeric(census_shape['GEOID'], errors='coerce')


shp_data = census_shape.merge(cleaned_df, left_on='GEOID', right_on='census_tract', how='inner')
# remove unecessary columns resulting from the merge
columns_to_remove = ['STATEFP', 'COUNTYFP', 'TRACTCE', 'NAME', 'NAMELSAD', 'MTFCC', 'FUNCSTAT', 'ALAND', 'AWATER', 'census_tract']
shp_data = shp_data.drop(columns=columns_to_remove)

# Save the filtered data to a new shapefile
shp_data.to_file("/Users/matthewcolantonio/Documents/Research/HMDA/saveddata/shp_data.shp", driver='ESRI Shapefile')



















