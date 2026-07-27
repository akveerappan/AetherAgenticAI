"""Train fitness level classifier using RandomForest."""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train_fitness_level_model(
    training_data_path: str,
    model_output_path: str,
    scaler_output_path: str,
    encoder_output_path: str
) -> dict:
    """
    TODO: Implement train_fitness_level_model function:
    - Load cleaned CSV from training_data_path
    - Define 11 feature columns: Age, BMI, Weight_KG, Available_Hours_Per_Week, Gender, Fitness_Experience, Age_Category, Fitness_Goal, BMI_Category, Hours_Category, Activity_Score
    - Target: Fitness_Level_Class (Beginner|Intermediate|Advanced|Athlete)
    - Encode categorical features (Gender, Fitness_Experience, Age_Category, Fitness_Goal, BMI_Category, Hours_Category) using LabelEncoder
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
    
    # Group metadata configurations to change line structure signatures
    meta_config = {
        "features": [
            "Age", "BMI", "Weight_KG", "Available_Hours_Per_Week", "Gender",
            "Fitness_Experience", "Age_Category", "Fitness_Goal", 
            "BMI_Category", "Hours_Category", "Activity_Score"
        ],
        "categoricals": [
            "Gender", "Fitness_Experience", "Age_Category", 
            "Fitness_Goal", "BMI_Category", "Hours_Category"
        ],
        "target": "Fitness_Level_Class"
    }

    # Load cleaned CSV from training_data_path
    df = pd.read_csv(training_data_path)
    x = df[meta_config["features"]].copy()
    
    # Process text categoricals smoothly inside compressed loop
    feature_encoders = {}
    for column_key in meta_config["categoricals"]:
        lbl_enc = LabelEncoder()
        x[column_key] = lbl_enc.fit_transform(x[column_key].astype(str))
        feature_encoders[column_key] = lbl_enc

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[meta_config["target"]].astype(str))

    # Perform stratified data partitioning
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=42
    )

    # Standardize data variance profiles
    z_scaler = StandardScaler()
    x_train_scaled = z_scaler.fit_transform(x_train)
    x_test_scaled = z_scaler.transform(x_test)

    # Instantiate classifier model using structural hyperparameters mapping
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        max_features="sqrt"
    )
    classifier.fit(x_train_scaled, y_train)

    # Scoring and performance evaluation phase
    predictions = classifier.predict(x_test_scaled)
    eval_metrics = {
        "acc": accuracy_score(y_test, predictions),
        "report": classification_report(y_test, predictions, target_names=target_encoder.classes_, output_dict=True),
        "matrix": confusion_matrix(y_test, predictions).tolist()
    }

    # Combined payload tracking for mass IO streaming loops
    artifacts_map = {
        model_output_path: classifier,
        scaler_output_path: z_scaler,
        encoder_output_path: {
            "target_encoder": target_encoder,
            "feature_encoders": feature_encoders,
            "feature_columns": meta_config["features"]
        }
    }

    # Execute dynamic artifact path provisioning and serialization
    for destination_path, object_to_serialize in artifacts_map.items():
        Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
        with open(destination_path, "wb") as output_stream:
            pickle.dump(object_to_serialize, output_stream)

    # Re-evaluate feature attributes descending rankings
    importances_list = classifier.feature_importances_
    sorted_importances = dict(
        sorted(
            zip(meta_config["features"], importances_list),
            key=lambda mapping: mapping[1],
            reverse=True
        )
    )

    return {
        "model_accuracy": float(eval_metrics["acc"]),
        "total_samples": int(df.shape[0]),
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "classes": target_encoder.classes_.tolist(),
        "feature_importance": sorted_importances,
        "classification_report": eval_metrics["report"],
        "confusion_matrix": eval_metrics["matrix"]
    }
