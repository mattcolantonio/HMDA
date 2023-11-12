#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 16:31:30 2023

@author: diegodearmas
"""

#%% Load and explore data

import pandas as pd


#%% Loading the data 

df = pd.read_csv('hmda_and_census.csv')

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

practice_df_clean = practice_df.dropna()

#%% Testing a simple model 



from sklearn.model_selection import train_test_split

# Other way ( This way we can choose both the independent and dependant variables)

'''
independent_v = [ ]

X = practice_df(independent_v)  
y = practice_df['income'] # Replace 'income' with the name of your target variable
 
'''

X = practice_df_clean.drop('income', axis=1)  # Replace income with the name of your target variable
y = practice_df_clean['income']


# Split the data - 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create linear regression object

model1 = LinearRegression()

# Train the model using the training sets

model1.fit(X_train, y_train)

# Make predictions using the testing set

y_pred = model1.predict(X_test)

# The coefficients
print('Coefficients:', model1.coef_)
# The mean squared error
print('Mean squared error:', mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination:', r2_score(y_test, y_pred))

# Better looking output 

coefficients = pd.DataFrame(model1.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# The model is really off, we need to do a better selections of our variables. 


