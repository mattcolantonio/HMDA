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

df = df[df['action_taken'] != 3] # remove 'others' 

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

#%% Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
t3 = datetime.datetime.now()
#Train the model on the training set
log_model=LogisticRegression(solver='liblinear', penalty='l1', C=1.0)
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

#%% Random forest 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import datetime


# Creating the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight= 'balanced')

# Start time
t3 = datetime.datetime.now()

# Fitting the model
rf_model.fit(X_train, y_train)

# Making predictions
y_pred = rf_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Elapsed time
elapsed_time = datetime.datetime.now() - t3
print("Elapsed Time:", elapsed_time)

#%% Random Forest visualization 

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# This is just a single tree of our model. 

# Visualize one decision tree from the random forest
estimator = rf_model.estimators_[0]

plt.figure(figsize=(20,10))
plot_tree(estimator, 
          filled=True, 
          feature_names=X.columns, 
          class_names=['Class1', 'Class2', 'Class3'], 
          max_depth=3)  # Set max_depth to control the size of the tree
plt.show()

#%% Vizualization of feautre importance to understand the overall behavior of our forest

feature_importances = pd.DataFrame(rf_model.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

plt.figure(figsize=(10,6))
feature_importances[:10].plot(kind='barh')  # Top 10 features
plt.title('Feature Importances in Random Forest')
plt.show()


