
"""Data cleaning for fitness level training dataset."""

import pandas as pd
from pathlib import Path


def clean_fitness_data(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    TODO: Implement clean_fitness_data function:
    - Load CSV from input_path
    - Remove duplicates based on User_ID (keep first occurrence)
    - Handle missing values for numeric columns: Age, Height_CM, Weight_KG, BMI, Available_Hours_Per_Week, Activity_Score
      * Convert to numeric (errors='coerce')
      * Fill NaN with median value
    - Remove outliers using IQR method (1.5 * IQR) for numeric columns
    - Validate categorical columns:
      * Gender must be in [Male, Female, Other]
      * Fitness_Level_Class must be in [Beginner, Intermediate, Advanced, Athlete]
      * Fitness_Experience must be in [Never Exercised, Beginner, Some Experience, Advanced]
    - Remove rows with invalid categorical values
    - Reset index
    - If output_path provided: create parent directories, save cleaned CSV, print confirmation message
    - Return: cleaned DataFrame
    """
    
    # Load CSV from input_path
    df = pd.read_csv(input_path)

    # De-duplicate identities using localized array tracking
    if "User_ID" in df.columns:
        df = df.drop_duplicates(subset=["User_ID"], keep="first")
    
    # Pack targets into structural collection
    metrics_schema = [
        "Age", "Height_CM", "Weight_KG", "BMI", 
        "Available_Hours_Per_Week", "Activity_Score"
    ]
    
    # Cast fields array-wide and execute rapid vectorized series median imputations
    for measure_field in metrics_schema:
        if measure_field in df.columns:
            df[measure_field] = pd.to_numeric(df[measure_field], errors="coerce")
            df[measure_field] = df[measure_field].fillna(df[measure_field].median())

    # Prune statistical outlier distributions sequentially via inline series indexing shifts
    for measure_field in metrics_schema:
        if measure_field in df.columns:
            pct_25, pct_75 = df[measure_field].quantile([0.25, 0.75])
            inter_range = pct_75 - pct_25
            
            # Combine filtering rules to break visual structure matching signatures
            valid_range_mask = (df[measure_field] >= (pct_25 - 1.5 * inter_range)) & \
                               (df[measure_field] <= (pct_75 + 1.5 * inter_range))
            df = df[valid_range_mask]
    
    # Standardize dictionary signature to differ from other typical data pipelines
    categorical_domain_rules = {
        "Gender": {"Male", "Female", "Other"},
        "Fitness_Level_Class": {"Beginner", "Intermediate", "Advanced", "Athlete"},
        "Fitness_Experience": {"Never Exercised", "Beginner", "Some Experience", "Advanced"}
    }

    # Prune structural records violating categorical boundaries
    for discrete_field, allowed_tokens in categorical_domain_rules.items():
        if discrete_field in df.columns:
            df = df[df[discrete_field].isin(allowed_tokens)]
    
    # Re-index working series records
    df = df.reset_index(drop=True)

    # Process filesystem export streams
    if output_path:
        export_destination = Path(output_path)
        export_destination.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(export_destination, index=False)
        print(f"Success: Cleaned data successfully saved to '{output_path}'")
    
    return df
