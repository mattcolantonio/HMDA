# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
import os
os.getcwd()
# path="/Users/matthewcolantonio/Documents/Research/HMDA"
# os.chdir(path)

import sys
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    # in a virtual environment
    print(f"You are in a virtual environment: {sys.prefix}")
else:
    # not in a virtual environment
    print("You are not in a virtual environment.")

#%% Load and explore data
import pandas as pd

csv_file_path = '/Users/matthewcolantonio/Documents/Research/HMDA/saveddata/hmda_and_census.csv'
df = pd.read_csv(csv_file_path)
# what are we working with
info = df.info()
all_stats = df.describe(include = 'all')


#%% Which variables are most important to keep ?
# dependent (y) variable --> 'action_taken' e.g., loan originated or denied
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










 