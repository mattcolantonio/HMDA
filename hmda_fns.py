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
import geopandas as gpd
import os
import sys

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

# %% Load and explore data - don't need if only interested in cleaned_df

#csv_file_path = '/Users/matthewcolantonio/Documents/Research/HMDA/saveddata/hmda_and_census.csv'
#df = pd.read_csv(csv_file_path)

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
csv_file_path = '/Users/matthewcolantonio/Documents/Research/HMDA/saveddata/hmda_and_census.csv'
output_directory = "/Users/matthewcolantonio/Documents/Research/HMDA/saveddata/"
cleaned_df = cleaningday(csv_file_path, output_directory)

# %% Creating the function for the Summary statistics and correlation plot


def summarize_and_visualize(df, corr_threshold=0.3):
    try:
        # Calculating enhanced summary statistics
        summary = pd.DataFrame()
        summary['mean'] = df.mean()
        summary['std'] = df.std()
        summary['min'] = df.min()
        summary['25%'] = df.quantile(0.25)
        summary['50%'] = df.quantile(0.5)
        summary['75%'] = df.quantile(0.75)
        summary['max'] = df.max()
        summary['skew'] = df.skew()
        summary['kurtosis'] = df.kurtosis()
        summary['missing_values'] = df.isna().sum()
    except Exception as e:
        print(f"Error in calculating summary statistics: {e}")
        return

    try:
        # Displaying Summary Table
        print("Enhanced Summary Statistics:")
        # Converts the DataFrame to a string representation
        print(summary.to_string())
    except Exception as e:
        print(f"Error in displaying summary statistics: {e}")
        return

    try:
        # Calculating and plotting the improved correlation plot
        corr_matrix = df.corr()
        mask = abs(corr_matrix) < corr_threshold

        plt.figure(figsize=(20, 15))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f",
                    cmap='coolwarm', mask=mask)
        plt.title("Correlation Matrix")
        plt.show()
    except Exception as e:
        print(f"Error in generating correlation plot: {e}")


# Usage
# %%
summarize_and_visualize(cleaned_df)  # CHECK
# %% no information rate 
count_accepted = cleaned_df['action_taken'].eq(1).sum()
count_denied = cleaned_df['action_taken'].eq(0).sum()
nir = count_accepted / (count_accepted + count_denied)
print("No Information Rate:", nir)


# %% Functions


def split_data(df, target_column='action_taken', test_size=0.2, random_state=42, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []

    # Exclude specified columns from features
    X = df.drop([target_column] + exclude_columns, axis=1)

    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


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

def get_coefficients(logistic_models, feature_names):
    # Create a DataFrame to store coefficients for each income group
    coefficients_df = pd.DataFrame()

    # Loop through each income group and extract coefficients
    for income_group, log_model in logistic_models.items():
        # Extract coefficients
        coefficients = log_model.coef_[0]

        # Create a DataFrame for the current income group
        income_group_df = pd.DataFrame(
            {'Feature': feature_names, f'Coefficient_{income_group}': coefficients})

        # Set 'Feature' as the index for the first DataFrame
        if coefficients_df.empty:
            coefficients_df = income_group_df.set_index('Feature')
        else:
            # Merge with the overall coefficients DataFrame using the 'Feature' index
            coefficients_df = coefficients_df.merge(income_group_df.set_index(
                'Feature'), left_index=True, right_index=True, how='outer')

    # Reset the index to make 'Feature' a regular column
    coefficients_df.reset_index(inplace=True)

    # Print or visualize the coefficients DataFrame
    return coefficients_df


def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test):
    start_time = datetime.datetime.now()
    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced')
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


def visualize_random_forest_tree(rf_model, X_train):
    estimator = rf_model.estimators_[0]
    plt.figure(figsize=(20, 10))
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
    plt.figure(figsize=(10, 6))
    feature_importances[:10].plot(kind='barh')  # Top 10 features
    plt.title('Feature Importances in Random Forest')
    plt.show()

def get_feature_importance(random_forest_models, feature_name):
    # Iterate through each income group and print feature importance
    for income_group, rf_model in random_forest_models.items():
        # Find the index of the feature in the columns
        feature_index = X_train.columns.get_loc(feature_name)

        # Print the feature importance for the specified column in each stratum
        feature_importance = rf_model.feature_importances_[feature_index]
        print(
            f"Income Group: {income_group}, Feature Importance ('{feature_name}'): {feature_importance}")


def kl_divergence(log_model, rf_model, X_test):
    log_probs = log_model.predict_proba(X_test)
    rf_probs = rf_model.predict_proba(X_test)

    # Binarize class labels
    y_true = label_binarize(y_test, classes=log_model.classes_)
    log_labels = label_binarize(log_model.predict(
        X_test), classes=log_model.classes_)
    rf_labels = label_binarize(rf_model.predict(
        X_test), classes=rf_model.classes_)

    # Compute KL Divergence
    kl_div = mutual_info_score(y_true.flatten(), log_labels.flatten(
    )) - mutual_info_score(y_true.flatten(), rf_labels.flatten())
    print(
        f"KL Divergence between Logistic Regression and Random Forest: {kl_div}")


def plot_roc_curve(model, X_test, y_test, model_name):
    # Get the probability for the positive class
    y_prob_binary = model.predict_proba(X_test)[:, 1]

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_prob_binary)
    roc_auc = roc_auc_score(y_test, y_prob_binary)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(
        f'Receiver Operating Characteristic (ROC) Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.show()
    



# %% Stratifying to account for Income variable's potential confounding effects on other vars

# Add this line to stratify cleaned_df into quintiles based on the 'income' column
cleaned_df['income_group'] = pd.qcut(cleaned_df['income'], q=5, labels=False)
# Now 'income_group' column represents the quintiles, 0 to 4

# we have geographic features- these are identifiers only
exclude_columns = ['census_tract', 'derived_msa.md', 'county_code']

# Split data within each income stratum
stratified_splits = {}
for income_group in cleaned_df['income_group'].unique():
    income_df = cleaned_df[cleaned_df['income_group'] == income_group]
    X_train, X_test, y_train, y_test = split_data(
        income_df, exclude_columns=exclude_columns)
    stratified_splits[income_group] = (X_train, X_test, y_train, y_test)
    
# stratified_splits is a dictionary-like structure where keys are income groups, 
# and values are tuples containing training and testing data.

# Logistic Regression and Random Forest for each income stratum - Data Dictionary
logistic_models = {}
random_forest_models = {}

for income_group, (X_train, X_test, y_train, y_test) in stratified_splits.items():
    print(f"\nIncome Group: {income_group}")

    # Logistic Regression
    log_model = train_and_evaluate_logistic_regression(
        X_train, X_test, y_train, y_test)
    logistic_models[income_group] = log_model

    # Access and print coefficients
    coefficients = log_model.coef_
    coef_df = pd.DataFrame(coefficients, columns=X_train.columns)
    print("\nCoefficients (Log Odds):")
    print(coef_df)

    # Random Forest
    rf_model = train_and_evaluate_random_forest(
        X_train, X_test, y_train, y_test)
    random_forest_models[income_group] = rf_model

# Evaluate coefficients - logit models
# logistic_models is a dictionary where keys are income group labels, and values are logistic regression models
# Call the function that gives coeffieicents for each income stratum
log_odds_df = get_coefficients(logistic_models, X_train.columns)




# Visualize random forest tree and feature importance for one income group
# visualize_random_forest_tree(random_forest_models[cleaned_df['income_group'].unique()[0]], stratified_splits[cleaned_df['income_group'].unique()[0]][0])
visualize_feature_importance(
    random_forest_models[cleaned_df['income_group'].unique()[0]],
    stratified_splits[cleaned_df['income_group'].unique()[0]][0])

# Compute KL Divergence for each income stratum
for income_group, (X_train, X_test, y_train, y_test) in stratified_splits.items():
    kl_divergence(logistic_models[income_group],
                  random_forest_models[income_group], X_test)
# negative KL divergence outputs means RF is closer to true distribution of outcomes
# low values mean the models are pretty similar in their distributions
# Note: 'income_group' column represents quintiles; adjust visualization and analysis accordingly

# what is feature importance of ['x'] variabe
get_feature_importance(random_forest_models, 'derived_race_Black or African American')



# what do the strata look like? looks good to me
for income_group in stratified_splits.keys():
    # Print the income range for each stratum
    income_range = cleaned_df.loc[cleaned_df['income_group']
                                  == income_group, 'income'].agg(['min', 'max'])
    print(
        f"Income Group: {income_group}, Income Range: {income_range['min']} to {income_range['max']}")


# ROC function Usage
plot_roc_curve(rf_model, X_test, y_test, 'Random Forest')
plot_roc_curve(log_model, X_test, y_test, 'Logistic Regression')

#%% Shapefile - make function

path_to_shape = "/Users/matthewcolantonio/Documents/Research/HMDA/rawdata/tl_2020_25_tract.zip" 
# yes, upload the entire zipfile
census_shape = gpd.read_file(path_to_shape)

print(cleaned_df['census_tract'].dtype)
print(census_shape['GEOID'].dtype)
census_shape['GEOID'] = pd.to_numeric(census_shape['GEOID'], errors='coerce')


shp_data = census_shape.merge(cleaned_df, left_on='GEOID', right_on='census_tract', how='inner')
# remove unecessary columns resulting from the merge
columns_to_remove = ['STATEFP', 'COUNTYFP', 'TRACTCE', 'NAME', 'NAMELSAD', 'MTFCC', 'FUNCSTAT', 'ALAND', 'AWATER', 'census_tract']
shp_data = shp_data.drop(columns=columns_to_remove)

#%% Make a map - make function 


import folium
from folium.plugins import DualMap

# Create a GeoDataFrame for the acceptance rate
acceptance_rate_gdf = shp_data[['GEOID', 'action_taken', 'geometry']].set_index('GEOID')

# Calculate NIR-style acceptance rate
count_accepted = shp_data['action_taken'].eq(1).sum()
count_denied = shp_data['action_taken'].eq(0).sum()
nir = count_accepted / (count_accepted + count_denied)

# Assuming 'GEOID' is the identifier column in shp_data
acceptance_rate_by_geoid = shp_data.groupby('GEOID')['action_taken'].mean().fillna(nir)

# Merge the calculated acceptance rate back to the shp_data GeoDataFrame
shp_data = shp_data.merge(acceptance_rate_by_geoid, how='left', left_on='GEOID', right_index=True, suffixes=('_orig', ''))

# Create an interactive map using folium
m = folium.Map(location=[shp_data.geometry.centroid.y.mean(), shp_data.geometry.centroid.x.mean()], zoom_start=10)

# Layer for acceptance rate
folium.Choropleth(geo_data=acceptance_rate_gdf.__geo_interface__,
                  name='Acceptance Rate',
                  data=acceptance_rate_gdf['action_taken'],
                  key_on='feature.id',
                  fill_color='YlGnBu',
                  fill_opacity=0.7,
                  line_opacity=0.2,
                  legend_name='Acceptance Rate').add_to(m)

# Add layer control to switch between layers
folium.LayerControl(collapsed=False).add_to(m)

# Save the interactive map as an HTML file
output_file_path = os.path.join(output_directory, "interactive_acceptance_rate_map.html")
m.save(output_file_path)
print(f"Interactive acceptance rate map saved to: {output_file_path}")





