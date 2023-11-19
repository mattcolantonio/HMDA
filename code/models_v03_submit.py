#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 12:20:41 2023

@author: matthewcolantonio
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
import datetime

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
#%% 
# different data manipulations need to be done based on the type of object
cleaned_df = new_df.dropna() # a lot of data is categorical, imputing won't do us any good here
# action_taken is target. 8 values won't work for logit, so we are reducing to accepted, denied, or 'other'
action_taken_mapping = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3, 7:3, 8:3}
cleaned_df['action_taken'] = cleaned_df['action_taken'].replace(action_taken_mapping)
print(cleaned_df['action_taken'].value_counts()) # verify

cleaned_df['loan_to_value_ratio'] = pd.to_numeric(cleaned_df['loan_to_value_ratio'], errors='coerce')
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
bin_edges = [-float('inf'), 20, 30, 36, 43, 49, 60, float('inf')]
# Define the bin labels
bin_labels = ['<20', '20-30', '30-36', '37-43', '44-49', '50-60', '>60']
# Create a new column with the specified bins
cleaned_df['debt_to_income_ratio_range'] = pd.cut(cleaned_df['debt_to_income_ratio'], bins=bin_edges, labels=bin_labels)
# Verify the result
print(cleaned_df['debt_to_income_ratio_range'].value_counts())
# Drop the original 'debt_to_income_ratio' column if you want
cleaned_df = cleaned_df.drop(columns=['debt_to_income_ratio'])

# Label encode variables for each categorical variable
# from sklearn.preprocessing import LabelEncoder

categorical_vars = ["derived_ethnicity", "derived_race", "derived_sex", "derived_loan_product_type", "debt_to_income_ratio_range", "applicant_age", "derived_dwelling_category"  ] 

for var in categorical_vars:
    dummies = pd.get_dummies(cleaned_df[var], prefix=var, drop_first=True)
    cleaned_df = pd.concat([cleaned_df, dummies], axis=1)
    cleaned_df.drop(var, axis=1, inplace=True)  # Drop original column after creating dummies

# new columns names
print(cleaned_df.columns)
#the number of columns removed from the original df were added back with dummy columns
#%% Some plots
import seaborn as sns
import matplotlib.pyplot as plt
# Bar Plot for 'action_taken' across 'derived_race'
plt.figure(figsize=(10, 6))
sns.countplot(x='derived_race_Black or African American', hue='action_taken', data=cleaned_df)
plt.title('Distribution of Action Taken across Race')
plt.show()

# Box Plot for 'loan_amount' by 'action_taken'
plt.figure(figsize=(10, 6))
sns.boxplot(x='action_taken', y='loan_amount', data=cleaned_df)
plt.title('Distribution of Loan Amount by Action Taken')
plt.show()



#%% Correlation

t1 = datetime.datetime.now()

correlation_vars = ['loan_amount', 'income', 'action_taken', 'derived_race_Black or African American', 'derived_race_White', 'loan_to_value_ratio']
correlation_data = cleaned_df[correlation_vars] # Subset the DataFrame with the selected variables

# Create a pair plot for the selected variables
sns.pairplot(correlation_data, hue='action_taken', palette='viridis')
plt.show()

# Create a correlation heatmap
correlation_matrix = correlation_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

t2 = datetime.datetime.now()
cor_time = t2 - t1
print("Elapsed Time:", cor_time) # this takes forever


# Printing the correlation with 'action_taken' column is way faster, just not as pretty
correlation_with_action_taken = correlation_matrix['action_taken']

print("\nCorrelation with 'action_taken':")
print(correlation_with_action_taken)
#%% Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

t3 = datetime.datetime.now()

cleaned_df2 = cleaned_df.dropna()

X = cleaned_df2.drop(['action_taken'], axis=1)
y=cleaned_df2['action_taken']

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

t4 = datetime.datetime.now()
logit_time = t4 - t3
print("Elapsed Time:", logit_time)

#%% Determine best depth for Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import plot_tree

t5 = datetime.datetime.now()

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

# how long?
t6 = datetime.datetime.now()
tree_time = t6 - t5
print("Elapsed Time:", tree_time)

#%% Tree plot
# Assuming you already have the best_tree from your GridSearchCV
plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
plot_tree(best_tree, feature_names=X_train.columns.tolist(), class_names=['Accepted', 'Denied', 'Other'], filled=True, rounded=True)
plt.show()
