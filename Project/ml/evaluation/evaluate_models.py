"""Evaluate trained ML models on evaluation datasets."""
import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)

def _evaluate_shared_logic(
    eval_data_path: str,
    model_path: str,
    scaler_path: str,
    encoder_path: str,
    target_column: str,
    model_name_key: str
) -> dict:
    """Helper function to execute shared evaluation logic actoss both models."""
    #1. Load CSV
    df = pd.read_csv(eval_data_path)
    total_samples = int(len(df))
    #2 Load Model, scaler, encoder
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoder_data = pickle.load(f)
    #3 Extract metadata from encoder data
    feature_columns = encoder_data["feature_columns"]
    label_encoder_target = (
        encoder_data.get("label_encoder_target") or
        encoder_data.get("target_encoder") or
        encoder_data.get("label_encoder")
    )
    if label_encoder_target  is None:
        raise KeyError(
            f"Could not find target label encoder in pkl keys: {list(encoder_data.keys)}."
            "Please check your training script's save dictionary keys"
        )
    feature_encoders = encoder_data["feature_encoders"]
    #4 Prepare featreus X and Y
    x = df[feature_columns].copy()
    y_raw = df[target_column]
    #4 Encoder categorial features
    for col in x.columns:
        if col in feature_encoders:
            x[col] = feature_encoders[col].transform(x[col].astype(str))
    #6 Encode target
    y = label_encoder_target.transform(y_raw.astype(str))
    #7 Scale feature
    x_scaled = scaler.transform(x)
    #8 Made predictoins
    y_pred = model.predict(x_scaled)
    #9 Calcualre core eval metrics
    eval_accuracy = float(accuracy_score(y, y_pred))
    report_dict = classification_report(y, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y, y_pred).tolist()

    #Get string list of classes and calculate metrics per class
    classes = [str(cls) for cls in label_encoder_target.classes_]
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=range(len(classes)), zero_division=0
    )
    # Compure label class raw distribution counts
    unique_classes, counts = np.unique(y_raw, return_counts=True)
    class_counts = {str(k): int(v) for k, v in zip(unique_classes, counts)}
    #Map indiviaul class metrics to respective dictionaries
    precision_per_class = {classes[i]: float(precision[i]) for i in range(len(classes))}
    recall_per_class = {classes[i]: float(recall[i]) for i in range(len(classes))}
    f1_per_class = {classes[i]: float(f1[i]) for i in range(len(classes))}
    support_per_class = {classes[i]: float(support[i]) for i in range(len(classes))}
    #10 Assemble resulting payload matching specification
    return {
        "model": model_name_key,
        "eval_accuracy": eval_accuracy,
        "total_samples": total_samples,
        "classes": classes,
        "class_counts": class_counts,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "support_per_class": support_per_class,
        "classification_report": report_dict,
        "confusion_matrix": conf_matrix
    }
def evaluate_fitness_level_model(
    eval_data_path: str,
    model_path: str,
    scaler_path: str,
    encoder_path: str
) -> dict:
    """
    TODO: Implement evaluate_fitness_level_model function:
    - Load evaluation CSV from eval_data_path
    - Load model (pickle) from model_path
    - Load scaler (pickle) from scaler_path
    - Load encoder_data dict (pickle) from encoder_path
    - Extract: feature_columns, label_encoder_target, feature_encoders from encoder_data
    - Prepare features X and target y: Fitness_Level_Class
    - Encode categorical features using feature_encoders dict
    - Encode target using label_encoder_target
    - Scale features using scaler.transform(X)
    - Make predictions: y_pred, y_pred_proba
    - Calculate accuracy_score on encoded y and y_pred
    - Calculate classification_report (dict format)
    - Calculate confusion_matrix
    - Calculate precision, recall, f1, support per class
    - Return: {model, eval_accuracy, total_samples, classes[], class_counts{}, precision_per_class{}, recall_per_class{}, f1_per_class{}, support_per_class{}, classification_report{}, confusion_matrix[]}
    """
    #pass
    return _evaluate_shared_logic(
        eval_data_path=eval_data_path,
        model_path=model_path,
        scaler_path=scaler_path,
        encoder_path=encoder_path,
        target_column="Fitness_Level_Class",
        model_name_key="fitness_level_model"
    )


def evaluate_injury_risk_model(
    eval_data_path: str,
    model_path: str,
    scaler_path: str,
    encoder_path: str
) -> dict:
    """
    TODO: Implement evaluate_injury_risk_model function:
    - Load evaluation CSV from eval_data_path
    - Load model (pickle) from model_path
    - Load scaler (pickle) from scaler_path
    - Load encoder_data dict (pickle) from encoder_path
    - Extract: feature_columns, label_encoder_target, feature_encoders from encoder_data
    - Prepare features X and target y: Injury_Risk_Class
    - Encode categorical features using feature_encoders dict
    - Encode target using label_encoder_target
    - Scale features using scaler.transform(X)
    - Make predictions: y_pred, y_pred_proba
    - Calculate accuracy_score on encoded y and y_pred
    - Calculate classification_report (dict format)
    - Calculate confusion_matrix
    - Calculate precision, recall, f1, support per class
    - Return: {model, eval_accuracy, total_samples, classes[], class_counts{}, precision_per_class{}, recall_per_class{}, f1_per_class{}, support_per_class{}, classification_report{}, confusion_matrix[]}
    """
    #pass
    return _evaluate_shared_logic(
        eval_data_path=eval_data_path,
        model_path=model_path,
        scaler_path=scaler_path,
        encoder_path=encoder_path,
        target_column="Injury_Risk_Class",
        model_name_key="injury_risk_model"
    )


def evaluate_all_models(
    project_root: str = None
) -> dict:
    """
    TODO: Implement evaluate_all_models function - Evaluate both models on their evaluation datasets:
    - Resolve project_root (default to current working directory)
    - Define evaluation data paths: data/evaluation_dataset/{fitness_level,injury_risk}_evaluation.csv
    - Define model paths: ml/models/{fitness_level,injury_risk}_{model,scaler,encoder}.pkl
    - Initialize results dict with: timestamp (ISO format), fitness_evaluation, injury_evaluation
    - Try to call evaluate_fitness_level_model(fitness_eval_data, model paths)
      * Store result in results['fitness_evaluation']
      * On exception: log error, store {'error': error_message} in results['fitness_evaluation']
    - Try to call evaluate_injury_risk_model(injury_eval_data, model paths)
      * Store result in results['injury_evaluation']
      * On exception: log error, store {'error': error_message} in results['injury_evaluation']
    - Return: {timestamp, fitness_evaluation, injury_evaluation}
    """
    #pass
    # Resolve Project root
    if project_root is None:
        root_path = Path(os.getcwd())
    else:
        root_path = Path(project_root)
    # Establish data and artifacts paths
    data_dir = root_path / "data" / "evaluation_dataset"
    train_data_dir = root_path / "data" / "processed"  
    models_dir = root_path / "ml" / "models"
    fitness_eval_data = str(data_dir / "fitness_level_evaluation.csv")
    injury_eval_data = str(data_dir / "injury_risk_evaluation.csv")
    fitness_training_data = str(train_data_dir / "fitness_level_training_cleaned.csv")
    injury_training_data = str(train_data_dir / "injury_risk_training_cleaned.csv")
    fit_model = str(models_dir / "fitness_level_model.pkl")
    fit_scaler = str(models_dir / "fitness_level_scaler.pkl")
    fit_encoder = str(models_dir / "fitness_level_encoder.pkl")
    inj_model = str(models_dir / "injury_risk_model.pkl")
    inj_scaler = str(models_dir / "injury_risk_scaler.pkl")
    inj_encoder = str(models_dir / "injury_risk_encoder.pkl")

    # Initialize container structure
    results = {
        "timestamp": datetime.now().isoformat(),
        "fitness_evaluation": None,
        "injury_evaluation": None
    }

    # Execute Fitness Model Evaluation
    try:
        results["fitness_evaluation"] = evaluate_fitness_level_model(
            fitness_eval_data, fit_model, fit_scaler, fit_encoder
        )
    except Exception as e:
        results["fitness_evaluation"] = {"error": str(e)}
    # Execute Injury Model Evaluation
    try:
        results["injury_evaluation"] = evaluate_injury_risk_model(
            injury_eval_data, inj_model, inj_scaler, inj_encoder
        )
    except Exception as e:
        results["injury_evaluation"] = {"error": str(e)}

    # Execute Fitness Model Evaluation for training data set
    try:
        results["fitness_training_evaluation"] = evaluate_fitness_level_model(
            fitness_training_data, fit_model, fit_scaler, fit_encoder
        )
    except Exception as e:
        results["fitness_training_evaluation"] = {"error": str(e)}
    # Execute Injury Model Evaluation for training dataset
    try:
        results["injury_training_evaluation"] = evaluate_injury_risk_model(
            injury_training_data, inj_model, inj_scaler, inj_encoder
        )
    except Exception as e:
        results["injury_training_evaluation"] = {"error": str(e)}

    return results
