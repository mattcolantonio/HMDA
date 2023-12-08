#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:34:14 2023

@author: matthewcolantonio
"""

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import datetime
import seaborn as sns
import pandas as pd
import os
import sys

import hmda_fns as models


#%%
os.getcwd()
# path="/Users/matthewcolantonio/Documents/Research/HMDA"
# os.chdir(path)
# want to make sure the virtual env is activated to avoid package version confusion amongst the team

if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    # in a virtual environment
    print(f"You are in a virtual environment: {sys.prefix}")
else:
    # not in a virtual environment
    print("You are not in a virtual environment.")

# %% Load and explore data


# Load and Clean!
csv_file_path = '/Users/matthewcolantonio/Documents/Research/HMDA/saveddata/hmda_and_census.csv'
output_directory = "/Users/matthewcolantonio/Documents/Research/HMDA/saveddata/"
cleaned_df = models.cleaningday(csv_file_path, output_directory)


# %% Models

# Add this line to stratify cleaned_df into quintiles based on the 'income' column
cleaned_df['income_group'] = pd.qcut(cleaned_df['income'], q=5, labels=False)
# Now 'income_group' column represents the quintiles, 0 to 4

# we have geographic features- these are identifiers only
exclude_columns = ['census_tract', 'derived_msa.md', 'county_code']

# Split data within each income stratum
stratified_splits = {}
for income_group in cleaned_df['income_group'].unique():
    income_df = cleaned_df[cleaned_df['income_group'] == income_group]
    X_train, X_test, y_train, y_test = models.split_data(
        income_df, exclude_columns=exclude_columns)
    stratified_splits[income_group] = (X_train, X_test, y_train, y_test)
    
# stratified_splits is a dictionary-like structure where keys are income groups, 
# and values are tuples containing training and testing data.
# Check where splits occured
# what do the strata look like? 
for income_group in stratified_splits.keys():
    # Print the income range for each stratum
    income_range = cleaned_df.loc[cleaned_df['income_group']
                                  == income_group, 'income'].agg(['min', 'max'])
    print(
        f"Income Group: {income_group}, Income Range: {income_range['min']} to {income_range['max']}")

# Logistic Regression and Random Forest for each income stratum - Data Dictionary
logistic_models = {}
random_forest_models = {}

for income_group, (X_train, X_test, y_train, y_test) in stratified_splits.items():
    print(f"\nIncome Group: {income_group}")

    # Logistic Regression
    log_model = models.train_and_evaluate_logistic_regression(
        X_train, X_test, y_train, y_test)
    logistic_models[income_group] = log_model

    # Access and print coefficients
    coefficients = log_model.coef_
    coef_df = pd.DataFrame(coefficients, columns=X_train.columns)
    print("\nCoefficients (Log Odds):")
    print(coef_df)

    # Random Forest
    rf_model = models.train_and_evaluate_random_forest(
        X_train, X_test, y_train, y_test)
    random_forest_models[income_group] = rf_model

# Evaluate coefficients - logit models
# logistic_models is a dictionary where keys are income group labels, and values are logistic regression models
# Call the function that gives coeffieicents for each income stratum
log_odds_df = models.get_coefficients(logistic_models, X_train.columns)


# Visualize random forest tree and feature importance for one income group
# models.visualize_random_forest_tree(random_forest_models[cleaned_df['income_group'].unique()[0]], stratified_splits[cleaned_df['income_group'].unique()[0]][0])
models.visualize_feature_importance(
    random_forest_models[cleaned_df['income_group'].unique()[0]],
    stratified_splits[cleaned_df['income_group'].unique()[0]][0])

# Compute KL Divergence for each income stratum
for income_group, (X_train, X_test, y_train, y_test) in stratified_splits.items():
    models.kl_divergence(logistic_models[income_group],
                  random_forest_models[income_group], X_test)
# negative KL divergence outputs means RF is closer to true distribution of outcomes
# low values mean the models are pretty similar in their distributions
# Note: 'income_group' column represents quintiles; adjust visualization and analysis accordingly

# what is feature importance of ['x'] variabe in the RF?
models.get_feature_importance(random_forest_models, 'derived_race_Black or African American')


# ROC function Usage
models.plot_roc_curve(rf_model, X_test, y_test, 'Random Forest')
models.plot_roc_curve(log_model, X_test, y_test, 'Logistic Regression')


