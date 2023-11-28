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
# calcualting no information rate, aka target for model accuracy:
count_action_1 = cleaned_df['action_taken'].eq(1).sum()
count_action_2_3 = cleaned_df['action_taken'].isin([2, 3]).sum()
nir = count_action_1 / (count_action_1 + count_action_2_3)
print("No Information Rate:", nir)
# ie, a model predicting every applicant will be accepted will be 80% accurate- our model needs to be better than this rate at least

cleaned_df['loan_to_value_ratio'] = pd.to_numeric(cleaned_df['loan_to_value_ratio'], errors='coerce') # coerce creates NaN for values that can't be converted to numeric
print(cleaned_df['loan_to_value_ratio'].dtype) # verify
cleaned_df['property_value'] = pd.to_numeric(cleaned_df['property_value'], errors='coerce')
print(cleaned_df['property_value'].dtype)

#debt_to_income_ratio needs some work
cleaned_df['debt_to_income_ratio'] = cleaned_df['debt_to_income_ratio'].replace('[%<>]', '', regex=True)
# Replace 'Exempt' with NaN (you can use 0 or any other value as needed)
cleaned_df['debt_to_income_ratio'] = cleaned_df['debt_to_income_ratio'].replace('Exempt', pd.NA)
# Convert ranges to average values
cleaned_df['debt_to_income_ratio'] = cleaned_df['debt_to_income_ratio'].apply(lambda x: eval(x.replace('-', '+')) / 2 if pd.notna(x) else x)
# Convert 'debt_to_income_ratio' to float
cleaned_df['debt_to_income_ratio'] = pd.to_numeric(cleaned_df['debt_to_income_ratio'], errors='coerce')
# Define the bin edges for the desired ranges
bin_edges = [-float('inf'), 20, 30, 36, 43, 49, 59, float('inf')]
# Define the bin labels
bin_labels = ['<20', '20-30', '30-36', '37-43', '44-49', '50-60', '>60']
# Create a new column with the specified bins
cleaned_df['debt_to_income_ratio_range'] = pd.cut(cleaned_df['debt_to_income_ratio'], bins=bin_edges, labels=bin_labels)
# Verify the result
print(cleaned_df['debt_to_income_ratio_range'].value_counts())
# Drop the original 'debt_to_income_ratio' column if you want
# cleaned_df = cleaned_df.drop(columns=['debt_to_income_ratio']) # run when ratio_range is correct
# maybe label encode the ratio, since order matters (higher number is meanigful)

# Creating Dummy variables for categorical variables
# dummies are best since order doesn't matter (as opposed to label encoding)
categorical_vars = ["derived_ethnicity", "derived_race", "derived_sex", "derived_loan_product_type",  "applicant_age", "derived_dwelling_category"  ] 

for var in categorical_vars:
    dummies = pd.get_dummies(cleaned_df[var], prefix=var, drop_first=True)
    cleaned_df = pd.concat([cleaned_df, dummies], axis=1)
    cleaned_df.drop(var, axis=1, inplace=True)  # Drop original column after creating dummies

# new columns names
print(cleaned_df.columns)

# Fore dummies, it is important to know the reference group- coefficents from models on dummies are coefficents relative to other groups
# dummy variables for categorical_vars

reference_categories = {}

for var in categorical_vars:
    # Get the unique categories after creating dummies
    categories_after_dummies = [col for col in cleaned_df.columns if var in col and '_1' in col]

    # Check if dummy variables were created
    if not categories_after_dummies:
        print(f"No dummy variables created for {var}. Check if there's an issue.")
        continue

    # Extract the original category name (before creating dummies)
    original_category = categories_after_dummies[0].replace('_1', '')

    reference_categories[var] = original_category

# Print the reference categories
print("Reference Categories:")
for var, ref_category in reference_categories.items():
    print(f"{var}: {ref_category}")


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

path_to_shape = "/Users/matthewcolantonio/Documents/Research/HMDA/rawdata/CENSUS2020_BLK_BG_TRCT_MA.zip" 
# yes, upload the entire zipfile
census_shape = gpd.read_file(path_to_shape)
census_shape['TRACTCE20'] = census_shape['TRACTCE20'].astype(float)

#cleaned_df2 = cleaned_df['census_tract'] % 1000000


#shp_data = census_shape.merge(cleaned_df2, left_on='TRACTCE20', right_on='census_tract', how='inner')


# Save the filtered data to a new shapefile
#shp_data.to_file("/Users/matthewcolantonio/Documents/Research/HMDA/saveddata/shp_data.shp", driver='ESRI Shapefile')



















