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

#%% Label Encoding
# Create a new DataFrame to store the converted data
from sklearn.preprocessing import LabelEncoder
hmda_encoded = new_df.copy()
# Apply label encoding to each object-type column
label_encoder = LabelEncoder()
for column in hmda_encoded.select_dtypes(include='O').columns:
    hmda_encoded[column] = label_encoder.fit_transform(hmda_encoded[column])

# Now, 'hmda_encoded' contains the converted data for use in analysis

hmda_encoded.info()

#%% Some data exploration
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd

# Choose the variables for correlation plots
correlation_vars = ['loan_amount', 'income', 'applicant_age', 'debt_to_income_ratio', 'action_taken']

# Subset the DataFrame with the selected variables
correlation_data = hmda_encoded[correlation_vars]

# Create a pair plot for the selected variables
sns.pairplot(correlation_data, hue='action_taken', palette='viridis')
plt.show()

# Create a correlation heatmap
correlation_matrix = correlation_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

#%% Mapping - need shapefile
# Creating a map displaying average loan_amount in each census tract
# tract_avg_loan_amount = hmda_encoded.groupby('census_tract')['loan_amount'].mean().reset_index()

# Load a shapefile with census tract geometries (replace 'your_shapefile.shp' with the path to your shapefile)
# census_tracts = gpd.read_file('your_shapefile.shp')

# Merge the tract_avg_loan_amount data with the census_tracts GeoDataFrame
#tracts_with_loan_amount = census_tracts.merge(tract_avg_loan_amount, on='census_tract')

# Plot the map
#fig, ax = plt.subplots(1, 1, figsize=(12, 8))
#tracts_with_loan_amount.plot(column='loan_amount', cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
#plt.title('Average Loan Amount by Census Tract')
#plt.show()










 
