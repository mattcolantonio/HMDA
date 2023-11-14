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
import numpy as np

csv_file_path = 'C:/Users/geean/Documents/2023Fall_ADEC743001/MLAlgorithms1Prj/HMDA/hmda_and_census/hmda_and_census.csv'
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
#pip install -U scikit-learn scipy matplotlib
#pip3 install -U scikit-learn scipy matplotlib
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
#pip install geopandas
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

#%% drop NaN values for the loan amount
df_without_nan = hmda_encoded.dropna()
#%% Decide which variables are highly correlated-create a correlogram
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df_without_nan' is your DataFrame
correlation_matrix = df_without_nan.corr()

# Set up the matplotlib figure with a larger size
plt.figure(figsize=(14, 12))

# Create a heatmap using seaborn with adjusted spacing
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={'size': 8})

# Show the plot
plt.show()

# application credit score type and reverse mortgage have 0.99, keep reverse mortgage
# loan type and derived_loan_product type keep dervied loan product type
# derived msa 


#%% Decision Tree code from class, chose a random depth of 10
# get a decision tree algorithm
# from sklearn.tree import DecisionTreeClassifier
# myt1 = DecisionTreeClassifier(
#     criterion = 'gini',
#     max_depth = 10,
#     min_samples_leaf=1
#     )
# myt1.fit(X=df_without_nan[["census_tract", "derived_msa.md", "county_code", "derived_loan_product_type",
#                      "derived_dwelling_category", "derived_ethnicity", "derived_race", "derived_sex",
#                       "purchaser_type", "loan_type", "loan_purpose", "reverse_mortgage", 
#                      "loan_amount", "loan_to_value_ratio", "property_value", "occupancy_type", "income",
#                      "debt_to_income_ratio", "applicant_credit_score_type", "applicant_age", 
#                      "tract_population", "tract_minority_population_percent", "ffiec_msa_md_median_family_income",
#                      "tract_to_msa_income_percentage", "tract_owner_occupied_units", "tract_median_age_of_housing_units",
#                      "Unemployment_Rate", "Median_HH_Income", "Vacant_Housing_Units_Percentage",
#                      "Tenure_Owner_Occupied_Percentage", "Median_Value_Owner_Occupied", "Median_Gross_Rent",
#                      "Total_Population_White_Percentage", "Total_Population_Black_Percentage"]], y = df_without_nan['action_taken'])

# # look at the decision_path - which samples end up in which leaf (via which path)
# mym1 = myt1.decision_path(df_without_nan[["census_tract", "derived_msa.md", "county_code", "derived_loan_product_type",
#                      "derived_dwelling_category", "derived_ethnicity", "derived_race", "derived_sex",
#                       "purchaser_type", "loan_type", "loan_purpose", "reverse_mortgage", 
#                      "loan_amount", "loan_to_value_ratio", "property_value", "occupancy_type", "income",
#                      "debt_to_income_ratio", "applicant_credit_score_type", "applicant_age", 
#                      "tract_population", "tract_minority_population_percent", "ffiec_msa_md_median_family_income",
#                      "tract_to_msa_income_percentage", "tract_owner_occupied_units", "tract_median_age_of_housing_units",
#                      "Unemployment_Rate", "Median_HH_Income", "Vacant_Housing_Units_Percentage",
#                      "Tenure_Owner_Occupied_Percentage", "Median_Value_Owner_Occupied", "Median_Gross_Rent",
#                      "Total_Population_White_Percentage", "Total_Population_Black_Percentage"]])
# print(mym1.toarray())

# # for extracting decision rules in a more humanly readable format, see e.g.:
# # https://mljar.com/blog/extract-rules-decision-tree/
# # also https://mljar.com/blog/visualize-decision-tree/

# # plot this
# from sklearn.tree import plot_tree
# plot_tree(myt1,filled = True)


#%% tree model, find the best model

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df_without_nan[["census_tract", "derived_msa.md", "county_code", "derived_loan_product_type",
                     "derived_dwelling_category", "derived_ethnicity", "derived_race", "derived_sex",
                     "purchaser_type", "loan_type", "loan_purpose", "reverse_mortgage", 
                     "loan_amount", "loan_to_value_ratio", "property_value", "occupancy_type", "income",
                     "debt_to_income_ratio", "applicant_credit_score_type", "applicant_age", 
                     "tract_population", "tract_minority_population_percent", "ffiec_msa_md_median_family_income",
                     "tract_to_msa_income_percentage", "tract_owner_occupied_units", "tract_median_age_of_housing_units",
                     "Unemployment_Rate", "Median_HH_Income", "Vacant_Housing_Units_Percentage",
                     "Tenure_Owner_Occupied_Percentage", "Median_Value_Owner_Occupied", "Median_Gross_Rent",
                     "Total_Population_White_Percentage", "Total_Population_Black_Percentage"]],  
    df_without_nan['action_taken'], 
    test_size=0.2, 
    random_state=42
)

# Define the parameter grid
param_grid = {'max_depth': [5, 10, 15, 20, None]}

# Create a decision tree classifier
tree_classifier = DecisionTreeClassifier(criterion='gini', min_samples_leaf=1)

# Perform grid search with cross-validation
grid_search = GridSearchCV(tree_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters:", grid_search.best_params_)

# Get the best model
best_tree = grid_search.best_estimator_


 