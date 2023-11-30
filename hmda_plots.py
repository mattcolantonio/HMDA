#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:39:35 2023

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

#csv_file_path = '/Users/matthewcolantonio/Documents/Research/HMDA/saveddata/cleaned_data.csv'
df = pd.read_csv(csv_file_path)
df = df[df['action_taken'] != 3] # remove 'others' 

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
correlation_data = df[correlation_vars] # Subset the DataFrame with the selected variables

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

#%% Spatial Analysis

# "/Users/matthewcolantonio/Documents/Research/HMDA/saveddata/shp_data.shp"