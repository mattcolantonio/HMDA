#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:40:39 2023

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
        print("Action taken mapping applied successfully.")
    except Exception as e:
        print(f"Error in action taken mapping: {e}")
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
        print(summary.to_string())  # Converts the DataFrame to a string representation
    except Exception as e:
        print(f"Error in displaying summary statistics: {e}")
        return

    try:
        # Calculating and plotting the improved correlation plot
        corr_matrix = df.corr()
        mask = abs(corr_matrix) < corr_threshold

        plt.figure(figsize=(20, 15))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', mask=mask)
        plt.title("Correlation Matrix")
        plt.show()
    except Exception as e:
        print(f"Error in generating correlation plot: {e}")
    
# %% 

# Usage 

summarize_and_visualize(cleaned_df)

