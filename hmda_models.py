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

#csv_file_path = '/Users/matthewcolantonio/Documents/Research/HMDA/saveddata/cleaned_data.csv'
df = pd.read_csv(csv_file_path)

#%% Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

t3 = datetime.datetime.now()

X = df.drop(['action_taken'], axis=1)
y=df['action_taken']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the model on the training set
log_model=LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l1', C=1.0)
# C: float, default=1.0 --> Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
# I'm suggesting L1/Lapalce since it has built-in feature selection, which can be helpful since we have so many predictors
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

#%% 


