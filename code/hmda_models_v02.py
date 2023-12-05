#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:34:14 2023

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

csv_file_path = '/Users/matthewcolantonio/Documents/Research/HMDA/saveddata/cleaned_data.csv'
df = pd.read_csv(csv_file_path)


#%% no information rate
count_accepted = df['action_taken'].eq(1).sum()
count_denied = df['action_taken'].eq(2).sum()
nir = count_accepted / (count_accepted + count_denied)
print("No Information Rate:", nir)


#%% Split data
from sklearn.model_selection import train_test_split

X = df.drop(['action_taken'], axis=1)
y=df['action_taken']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(y_train.value_counts())
print(y_test.value_counts())

#%% 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
#%% Functions

def train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test, solver='liblinear', penalty='l1', C=1.0): # default values
    start_time = datetime.datetime.now()
    log_model = LogisticRegression(solver=solver, penalty=penalty, C=C)
    log_model.fit(X_train, y_train)
    y_pred = log_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print("Elapsed Time:", elapsed_time)
    return log_model, elapsed_time

def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test):
    start_time = datetime.datetime.now()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print("Elapsed Time:", elapsed_time)
    return rf_model, elapsed_time

def visualize_random_forest_tree(rf_model, X_train):
    estimator = rf_model.estimators_[0]
    plt.figure(figsize=(20,10))
    plot_tree(estimator, 
              filled=True, 
              feature_names=X_train.columns, 
              class_names=['Class1', 'Class2', 'Class3'], 
              max_depth=3)  # Set max_depth to control the size of the tree
    plt.show()

def visualize_feature_importance(rf_model, X_train):
    feature_importances = pd.DataFrame(rf_model.feature_importances_,
                                       index=X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    plt.figure(figsize=(10,6))
    feature_importances[:10].plot(kind='barh')  # Top 10 features
    plt.title('Feature Importances in Random Forest')
    plt.show()

#%% Using functions

#Logistic Regression
# Train and evaluate logistic regression
log_model, log_elapsed_time = train_and_evaluate_logistic_regression(
    X_train, X_test, y_train, y_test,
    solver='liblinear',  # Change the solver
    penalty='l1',          # Change the penalty
    C= 1.0                 # Change the regularization strength
)

coefficients = log_model.coef_
# Print coefficients
coef_df = pd.DataFrame(coefficients, columns=X_train.columns)
print("\nCoefficients (Log Odds):")
print(coef_df)

# Random Forest 
# Train and evaluate random forest
rf_model, rf_elapsed_time = train_and_evaluate_random_forest(X_train, X_test, y_train, y_test)

# Visualize random forest tree
visualize_random_forest_tree(rf_model, X_train)

# Visualize feature importance
visualize_feature_importance(rf_model, X_train)

print("Logistic Regression Elapsed Time:", log_elapsed_time)
print("Random Forest Elapsed Time:", rf_elapsed_time)

