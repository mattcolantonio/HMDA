#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 16:25:23 2023

@author: diegodearmas
"""

#%% Importing pckgs 

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


#%% Reading the data 

data = pd.read_csv('hmda_and_census.csv')

print(data.columns)

#%% Creating a summary table 

data_summary = data.describe()
print(data_summary)


#%%

# Creating Correlation matrix 

correlation_matrix = data.corr()

# Creating a threshold for only relevant variables 

threshold = 0.5 

# Considering both negative and positive correlation 

filtered_corr = correlation_matrix.copy()
filtered_corr[np.abs(correlation_matrix) < threshold] = np.nan

# Creating a mask for the upper triangle since the table is simmetrical

mask = np.triu(np.ones_like(filtered_corr, dtype=bool))

# Creating the heatmap 

plt.figure(figsize=(24, 18))
sns.heatmap(filtered_corr, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Filtered Correlation Matrix')
plt.show()

#%% 

# Splitting the data into training and test dataset 

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

#%% 

# Creating a simple linear regression for our model 

# Selecting only the variables that we are using 

# Selecting our independent variables form our dataset 

selected_vars = [ 'Median_Gross_Rent', 
                 'Total_Population', 'loan_type', 'loan_amount']

X_train = train_data[selected_vars]

# Selecting our independent variable 

Y_train = train_data['Median_HH_Income']

#%% 

# Cleaning our data for N/a value 

print(X_train.isnull().sum())

print(Y_train.isnull().sum())

# 2 options delete rows or fill na with the mean of the column 

X_train = X_train.dropna()
Y_train = Y_train.loc[X_train.index]


# Checking the correlation and variance 

print(X_train.corr())

print(X_train.var())

# Creating the model 

X_train = sm.add_constant(X_train)

model1 = sm.OLS(Y_train, X_train).fit()

print(model1.summary())





