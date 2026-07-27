"""Data cleaning for injury risk training dataset."""

import pandas as pd
from pathlib import Path


def clean_injury_data(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    TODO: Implement clean_injury_data function:
    - Load CSV from input_path
    - Remove duplicates based on User_ID (keep first occurrence)
    - Handle missing values for numeric columns: Age, BMI, Flexibility_Score, Strength_Imbalance_Score, Training_Frequency_Hours, Overtraining_Risk_Score, Has_Health_Conditions, Previous_Injury
      * Convert to numeric (errors='coerce')
      * Fill NaN with median value
    - Remove outliers using IQR method (1.5 * IQR) for numeric columns (exclude binary flags: Has_Health_Conditions, Previous_Injury)
    - Validate categorical columns:
      * Gender must be in [Male, Female, Other]
      * Fitness_Level must be in [Beginner, Intermediate, Advanced, Athlete]
      * Fitness_Experience must be in [Never Exercised, Beginner, Some Experience, Advanced]
      * Injury_Risk_Class must be in [Low Risk, Moderate Risk, High Risk, Very High Risk]
    - Validate binary columns: Has_Health_Conditions and Previous_Injury must be in [0, 1]
    - Remove rows with invalid values
    - Reset index
    - If output_path provided: create parent directories, save cleaned CSV, print confirmation message
    - Return: cleaned DataFrame
    """
    
    # Load CSV from input_path
    df = pd.read_csv(input_path)

    # Erase data profile records containing repeating primary identifier markers
    if "User_ID" in df.columns:
        df = df.drop_duplicates(subset=["User_ID"], keep="first")

    # Differentiate skewed continuous values from unskewed category labels
    continuous_metrics = [
        "Age", "BMI", "Flexibility_Score", "Strength_Imbalance_Score", 
        "Training_Frequency_Hours", "Overtraining_Risk_Score"
    ]
    binary_indicator_flags = ["Has_Health_Conditions", "Previous_Injury"]

    # Coerce all fields in a single optimized loop configuration sequence
    for variable_header in (continuous_metrics + binary_indicator_flags):
        if variable_header in df.columns:
            df[variable_header] = pd.to_numeric(df[variable_header], errors="coerce")
            df[variable_header] = df[variable_header].fillna(df[variable_header].median())

    # Purge distribution outliers safely by creating dynamic lower and upper bounds
    for continuous_key in continuous_metrics:
        if continuous_key in df.columns:
            quartile_low, quartile_high = df[continuous_key].quantile([0.25, 0.75])
            variance_span = 1.5 * (quartile_high - quartile_low)
            
            # Combine condition profiles into a single expression statement to alter block structures
            df = df[
                df[continuous_key].between(quartile_low - variance_span, quartile_high + variance_span)
            ]

    # Structure discrete state definitions using compressed rule sets
    categorical_criteria = {
        "Gender": {"Male", "Female", "Other"},
        "Fitness_Level": {"Beginner", "Intermediate", "Advanced", "Athlete"},
        "Fitness_Level_Class": {"Beginner", "Intermediate", "Advanced", "Athlete"},
        "Fitness_Experience": {"Never Exercised", "Beginner", "Some Experience", "Advanced"},
        "Injury_Risk_Class": {"Low Risk", "Moderate Risk", "High Risk", "Very High Risk"},
        "Has_Health_Conditions": {0, 1},
        "Previous_Injury": {0, 1}
    }

    # Evaluate all target constraints cleanly
    for target_column_name, accepted_states in categorical_criteria.items():
        if target_column_name in df.columns:
            df = df[df[target_column_name].isin(accepted_states)]

    # Flattens index profiles
    df = df.reset_index(drop=True)

    # Delegate target export processing safely
    if output_path:
        storage_location = Path(output_path)
        storage_location.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(storage_location, index=False)
        print(f"Success: Cleaned data successfully saved to '{output_path}'")

    return df
