"""Train injury risk classifier using RandomForest."""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train_injury_risk_model(
    training_data_path: str,
    model_output_path: str,
    scaler_output_path: str,
    encoder_output_path: str
) -> dict:
    """
    TODO: Implement train_injury_risk_model function:
    - Load cleaned CSV from training_data_path
    - Define 12 feature columns: Age, BMI, Fitness_Level, Gender, Fitness_Experience, Age_Category, BMI_Category, Has_Health_Conditions, Previous_Injury, Flexibility_Score, Strength_Imbalance_Score, Training_Frequency_Hours
    - Target: Injury_Risk_Class (Low Risk|Moderate Risk|High Risk|Very High Risk)
    - Encode categorical features (Fitness_Level, Gender, Fitness_Experience, Age_Category, BMI_Category) using LabelEncoder
    - Encode target variable using LabelEncoder
    - Split 80/20 with stratification and random_state=42
    - Scale features using StandardScaler (fit on train, transform on test)
    - Train RandomForest: n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, class_weight='balanced', n_jobs=-1
    - Calculate test set accuracy and classification_report (dict format)
    - Save model (pickle) to model_output_path
    - Save scaler (pickle) to scaler_output_path
    - Save encoder_data dict with target_encoder, feature_encoders, feature_columns to encoder_output_path
    - Calculate and sort feature importance
    - Return: {model_accuracy, total_samples, train_samples, test_samples, classes[], feature_importance{}, classification_report{}, confusion_matrix[]}
    """
    
    # Load cleaned CSV from training_data_path
    df = pd.read_csv(training_data_path)

    # Define 12 feature columns and target
    feature_columns = [
        "Age",
        "BMI",
        "Fitness_Level",
        "Gender",
        "Fitness_Experience",
        "Age_Category",
        "BMI_Category",
        "Has_Health_Conditions",
        "Previous_Injury",
        "Flexibility_Score",
        "Strength_Imbalance_Score",
        "Training_Frequency_Hours",
    ]
    categorical_features = [
        "Fitness_Level",
        "Gender",
        "Fitness_Experience",
        "Age_Category",
        "BMI_Category",
    ]
    target_column = "Injury_Risk_Class"

    x = df[feature_columns].copy()
    y = df[target_column].copy()

    # Encode categorical features using LabelEncoder
    feature_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col].astype(str))
        feature_encoders[col] = le
    
    # Encode target variable using LabelEncoder
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y.astype(str))

    # Split 80/20 with stratification and random_state=42
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=42
    )

    # Scale features using StandardScaler (fit on train, transform on test)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train RandomForest with the specified hyperparameters
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        max_features="sqrt",
    )
    model.fit(x_train_scaled, y_train)

    # Evaluate the model on test set
    y_pred = model.predict(x_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    # Calculate classification report (dict format)
    report_dict = classification_report(
        y_test, y_pred, target_names=target_encoder.classes_, output_dict=True
    )

    # Calculate confusion matrix converted to native python nested lists
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Ensure parent directories exist for all output files
    for path_str in [model_output_path, scaler_output_path, encoder_output_path]:
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model (pickle) to model_output_path
    with open(model_output_path, "wb") as f:
        pickle.dump(model, f)
    
    # Save scaler (pickle) to scaler_output_path
    with open(scaler_output_path, "wb") as f:
        pickle.dump(scaler, f)
    
    # Save encoder_data dict to encoder_output_path
    encoder_data = {
        "target_encoder": target_encoder,
        "feature_encoders": feature_encoders,
        "feature_columns": feature_columns,
    }
    with open(encoder_output_path, "wb") as f:
        pickle.dump(encoder_data, f)
    
    # Calculate and sort feature importance descending
    importances = model.feature_importances_
    feature_importance_dict = dict(zip(feature_columns, importances))
    sorted_feature_importance = dict(
        sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)
    )

    # Construct and return performance summary dictionary
    return {
        "model_accuracy": float(acc),
        "total_samples": int(len(df)),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "classes": target_encoder.classes_.tolist(),
        "feature_importance": sorted_feature_importance,
        "classification_report": report_dict,
        "confusion_matrix": cm,
    }
