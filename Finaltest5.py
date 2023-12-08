#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:05:42 2023

@author: diegodearmas
"""

# %% Load and explore data

import pandas as pd
import datetime
import os
import seaborn as sns
import matplotlib.pyplot as plt


# %%

os.getcwd()

# %% Loading the data

df = pd.read_csv('hmda_and_census.csv')

# what are we working with

# Creating a summary of statistics

info = df.info()
all_stats = df.describe(include='all')

# %%


def cleaningday(csv_file_path, output_directory):
    # Helper function to convert ranges to averages
    def convert_range_to_average(value):
        if '-' in str(value):
            lower, upper = value.split('-')
            return (float(lower) + float(upper)) / 2
        return value

    try:
        df = pd.read_csv(csv_file_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    try:
        # Select variables to keep
        variables_to_keep = [
            "census_tract", "derived_msa.md", "county_code", "derived_loan_product_type",
            "derived_dwelling_category", "derived_ethnicity", "derived_race", "derived_sex",
            "action_taken", "loan_type", "loan_purpose", "reverse_mortgage",
            "loan_amount", "loan_to_value_ratio", "property_value", "occupancy_type", "income",
            "debt_to_income_ratio", "applicant_credit_score_type", "applicant_age",
            "tract_population", "tract_minority_population_percent", "ffiec_msa_md_median_family_income",
            "tract_to_msa_income_percentage", "tract_owner_occupied_units", "tract_median_age_of_housing_units",
            "Unemployment_Rate", "Median_HH_Income", "Vacant_Housing_Units_Percentage",
            "Tenure_Owner_Occupied_Percentage", "Median_Value_Owner_Occupied", "Median_Gross_Rent",
            "Total_Population_White_Percentage", "Total_Population_Black_Percentage"
        ]
        new_df = df[variables_to_keep]
        print("Variables selected successfully.")
    except Exception as e:
        print(f"Error in variable selection: {e}")
        return

    try:
        # Map 'action_taken' values
        action_taken_mapping = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3}
        new_df['action_taken'] = new_df['action_taken'].replace(
            action_taken_mapping)

        # Recode 'action_taken' to 0 and 1
        new_df['action_taken'] = (new_df['action_taken'] == 1).astype(int)

        print("Action taken mapping and recoding applied successfully.")
    except Exception as e:
        print(f"Error in action taken mapping and recoding: {e}")
        return

    try:
        new_df['income'] = new_df['income'].where(new_df['income'] >= 0, pd.NA)
        print("Negative values in 'income' column replaced with NaN.")
    except Exception as e:
        print(f"Error in ensuring 'income' is nonnegative: {e}")
        return

    try:
        # Convert 'loan_to_value_ratio' and 'property_value' to numeric
        new_df['loan_to_value_ratio'] = pd.to_numeric(
            new_df['loan_to_value_ratio'], errors='coerce')
        new_df['property_value'] = pd.to_numeric(
            new_df['property_value'], errors='coerce')
        print(
            "Conversion to numeric types done for loan_to_value_ratio and property_value.")
    except Exception as e:
        print(
            f"Error in converting loan_to_value_ratio and property_value to numeric: {e}")
        return

    try:
        # Clean 'debt_to_income_ratio'
        new_df['debt_to_income_ratio'] = new_df['debt_to_income_ratio'].replace(
            '[%<>]', '', regex=True).replace('Exempt', pd.NA)
        new_df['debt_to_income_ratio'] = new_df['debt_to_income_ratio'].apply(
            convert_range_to_average)
        new_df['debt_to_income_ratio'] = pd.to_numeric(
            new_df['debt_to_income_ratio'], errors='coerce')
        print("debt_to_income_ratio cleaned and converted.")
    except Exception as e:
        print(f"Error in cleaning debt_to_income_ratio: {e}")
        return

    try:
        # Handle categorical variables and create dummies
        categorical_vars = ["derived_ethnicity", "derived_race", "derived_sex",
                            "derived_loan_product_type", "applicant_age", "derived_dwelling_category"]
        values_to_remove = ['Free Form Text Only',
                            'Sex Not Available', '8888', 'Joint', 'Race Not Available']
        new_df = new_df[~new_df[categorical_vars].isin(
            values_to_remove).any(axis=1)]

        for var in categorical_vars:
            dummies = pd.get_dummies(new_df[var], prefix=var, drop_first=True)
            new_df = pd.concat([new_df, dummies], axis=1).drop(var, axis=1)
        print("Categorical variables processed and dummies created.")
    except Exception as e:
        print(f"Error in processing categorical variables: {e}")
        return

    try:
        # Final data cleaning steps
        new_df = new_df.dropna()
        new_df = new_df[new_df['action_taken'] != 3]
        print("Final data cleaning steps completed.")
    except Exception as e:
        print(f"Error in final data cleaning steps: {e}")
        return

    try:
        # Export the cleaned data
        output_file_path = os.path.join(output_directory, "cleaned_data.csv")
        new_df.to_csv(output_file_path, index=False)
        print(f"Data exported to: {output_file_path}")
    except Exception as e:
        print(f"Error exporting data: {e}")

    return new_df

# %%


# Example usage
csv_file_path = "/Users/diegodearmas/Documents/ADEC7430/2023Fall_ADEC743001/hmda_and_census.csv"
output_directory = "/Users/diegodearmas/Documents/ADEC7430/2023Fall_ADEC743001/"
cleaned_df = cleaningday(csv_file_path, output_directory)


#%%

def data_summary(df):
    # Step 1: Generate summary statistics for numeric variables
    numeric_data = df.select_dtypes(include=[np.number])
    numeric_summary = numeric_data.describe()
    print("Summary of Numeric Variables:")
    print(numeric_summary)

    # Step 2: Calculate No Information Rate (NIR)
    count_accepted = df['action_taken'].eq(1).sum()
    count_denied = df['action_taken'].eq(0).sum()
    nir = count_accepted / (count_accepted + count_denied)
    print("No Information Rate:", nir)

    # Step 3: Correlation analysis and visualization
    correlation_vars = ['loan_amount', 'income', 'action_taken', 
                        'derived_race_Black or African American', 
                        'derived_race_White', 'loan_to_value_ratio']

    # Check if all correlation variables are in the DataFrame
    if not all(item in df.columns for item in correlation_vars):
        print("Not all specified variables are present in the DataFrame.")
        return

    correlation_data = df[correlation_vars]  # Subset the DataFrame with the selected variables

    # Create a pair plot for the selected variables
    sns.pairplot(correlation_data, hue='action_taken', palette='viridis')
    plt.show()

    # Create a correlation heatmap
    correlation_matrix = correlation_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

#%% 

data_summary(cleaned_df)

#%% 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mutual_info_score
from sklearn.tree import plot_tree
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

#%% Functions - Split data 

def split_data(df, target_column='action_taken', test_size=0.2, random_state=42):
    X = df.drop([target_column], axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


#%% Logistic regression model 

def train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test, solver='liblinear', penalty='l1', C=1.0):
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
    return log_model

#%% Random forest model 

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
    return rf_model

#%% random forest graphs and plots 

def visualize_random_forest_tree(rf_model, X_train):
    estimator = rf_model.estimators_[0]
    plt.figure(figsize=(20,10))
    plot_tree(estimator, 
              filled=True, 
              feature_names=X_train.columns, 
              class_names=['Class1', 'Class2', 'Class3'], 
              max_depth=3)
    plt.show()

def visualize_feature_importance(rf_model, X_train):
    feature_importances = pd.DataFrame(rf_model.feature_importances_,
                                       index=X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    plt.figure(figsize=(10,6))
    feature_importances[:10].plot(kind='barh')  # Top 10 features
    plt.title('Feature Importances in Random Forest')
    plt.show()

#%% Comapring our two models 

def kl_divergence(log_model, rf_model, X_test):
    log_probs = log_model.predict_proba(X_test)
    rf_probs = rf_model.predict_proba(X_test)

    # Binarize class labels
    y_true = label_binarize(y_test, classes=log_model.classes_)
    log_labels = label_binarize(log_model.predict(X_test), classes=log_model.classes_)
    rf_labels = label_binarize(rf_model.predict(X_test), classes=rf_model.classes_)

    # Compute KL Divergence
    kl_div = mutual_info_score(y_true.flatten(), log_labels.flatten()) - mutual_info_score(y_true.flatten(), rf_labels.flatten())
    print(f"KL Divergence between Logistic Regression and Random Forest: {kl_div}")

#%% Using functions

# Split data
X_train, X_test, y_train, y_test = split_data(cleaned_df)


#%% Logistic Regression
log_model = train_and_evaluate_logistic_regression(
    X_train, X_test, y_train, y_test,
    solver='liblinear', # edit solver
    penalty='l1', # edit penalty
    C=1.0 # edit regularization strength
)

#%% Random Forest

rf_model = train_and_evaluate_random_forest(X_train, X_test, y_train, y_test)

#%% Visualize random forest tree

visualize_random_forest_tree(rf_model, X_train)

#%% Visualize feature importance

visualize_feature_importance(rf_model, X_train)

#%% Compute KL Divergence

kl_divergence(log_model, rf_model, X_test)

#%%% Stratifying to account for Income

# Add this line to stratify cleaned_df into quintiles based on the 'income' column
cleaned_df['income_group'] = pd.qcut(cleaned_df['income'], q=5, labels=False)

# Now 'income_group' column represents the quintiles, 0 to 4

# Split data within each income stratum
stratified_splits = {}
for income_group in cleaned_df['income_group'].unique():
    income_df = cleaned_df[cleaned_df['income_group'] == income_group]
    X_train, X_test, y_train, y_test = split_data(income_df)
    stratified_splits[income_group] = (X_train, X_test, y_train, y_test)

# Logistic Regression and Random Forest for each income stratum
logistic_models = {}
random_forest_models = {}

for income_group, (X_train, X_test, y_train, y_test) in stratified_splits.items():
    print(f"\nIncome Group: {income_group}")
    
    # Logistic Regression
    log_model = train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test)
    logistic_models[income_group] = log_model

    # Access and print coefficients
    coefficients = log_model.coef_
    coef_df = pd.DataFrame(coefficients, columns=X_train.columns)
    print("\nCoefficients (Log Odds):")
    print(coef_df)

    # Random Forest
    rf_model = train_and_evaluate_random_forest(X_train, X_test, y_train, y_test)
    random_forest_models[income_group] = rf_model

# Evaluate coefficients
# logistic_models is a dictionary where keys are income group labels, and values are logistic regression models
# Create a DataFrame to store coefficients for each income group
coefficients_df = pd.DataFrame()

# Loop through each income group and extract coefficients
for income_group, log_model in logistic_models.items():
    # Extract coefficients and feature names
    coefficients = log_model.coef_[0]
    feature_names = X_train.columns

    # Create a DataFrame for the current income group
    income_group_df = pd.DataFrame({'Feature': feature_names, f'Coefficient_{income_group}': coefficients})

    # Set 'Feature' as the index for the first DataFrame
    if coefficients_df.empty:
        coefficients_df = income_group_df.set_index('Feature')
    else:
        # Merge with the overall coefficients DataFrame using the 'Feature' index
        coefficients_df = coefficients_df.merge(income_group_df.set_index('Feature'), left_index=True, right_index=True, how='outer')

# Reset the index to make 'Feature' a regular column
coefficients_df.reset_index(inplace=True)

# Print or visualize the coefficients DataFrame
print(coefficients_df)

# Visualize random forest tree and feature importance for one income group (you can customize this)
visualize_random_forest_tree(random_forest_models[cleaned_df['income_group'].unique()[0]], stratified_splits[cleaned_df['income_group'].unique()[0]][0])
visualize_feature_importance(random_forest_models[cleaned_df['income_group'].unique()[0]], stratified_splits[cleaned_df['income_group'].unique()[0]][0])

# Compute KL Divergence for each income stratum
for income_group, (X_train, X_test, y_train, y_test) in stratified_splits.items():
    kl_divergence(logistic_models[income_group], random_forest_models[income_group], X_test)
# negative KL divergence outputs means RF is closer to true distribution of outcomes
# low values mean the models are pretty similar in their distributions
# Note: 'income_group' column represents quintiles; adjust visualization and analysis accordingly

# what is feature importance of ['x'] variabe
for income_group, rf_model in random_forest_models.items():
    # Find the index of the feature in the columns
    feature_index = X_train.columns.get_loc('derived_race_Black or African American')
    
    # Print the feature importance for the specified column in each stratum
    feature_importance = rf_model.feature_importances_[feature_index]
    print(f"Income Group: {income_group}, Feature Importance ('derived_race_Black or African American'): {feature_importance}")

# what do the strata look like? looks good to me
for income_group in stratified_splits.keys():
    # Print the income range for each stratum
    income_range = cleaned_df.loc[cleaned_df['income_group'] == income_group, 'income'].agg(['min', 'max'])
    print(f"Income Group: {income_group}, Income Range: {income_range['min']} to {income_range['max']}")

#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def plot_roc_curve_direct(model, X_test, y_test, model_name):
    # Check the distribution in y_test
    unique, counts = np.unique(y_test, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print("Class distribution in y_test:", class_distribution)

    if len(class_distribution) < 2:
        print("Warning: Only one class present in y_test. ROC curve cannot be plotted.")
        return

    # Predict probabilities for the positive class
    y_prob_binary = model.predict_proba(X_test)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob_binary)
    roc_auc = roc_auc_score(y_test, y_prob_binary)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for ' + model_name)
    plt.legend(loc="lower right")
    plt.show()

# Usage example
# plot_roc_curve_direct(rf_model, X_test, y_test, 'Random Forest')
# plot_roc_curve_direct(log_model, X_test, y_test, 'Logistic Regression')


#%%  Usage

plot_roc_curve_direct(rf_model, X_test, y_test, 'Random Forest')

plot_roc_curve_direct(log_model, X_test, y_test, 'Logistic Regression')

#%%

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def plot_combined_roc_curve(models, model_names, X_test, y_test):
    plt.figure(figsize=(8, 6))

    for model, name in zip(models, model_names):
        # Predict probabilities for the positive class
        y_prob_binary = model.predict_proba(X_test)[:, 1]

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_prob_binary)
        roc_auc = roc_auc_score(y_test, y_prob_binary)

        # Plotting each model's ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'{name} ROC curve (area = {roc_auc:.2f})')

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Comparison')
    plt.legend(loc="lower right")
    plt.show()

#%% Usage example

modelstest = [rf_model, log_model]
model_names = ['Random Forest', 'Logistic Regression']
plot_combined_roc_curve(modelstest, model_names, X_test, y_test)

