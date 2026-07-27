"""
Injury Assessor ML Agent module

TODO: Implement InjuryAssessorMLAgent class:
  - Constants: CLASS_NAMES = [Low Risk, Moderate Risk, High Risk, Very High Risk]

  - Constructor __init__(model_dir="ml/models"):
    * Load 3 pickle files: injury_risk_model.pkl, injury_risk_scaler.pkl, injury_risk_encoder.pkl
    * Store as self.model, self.scaler, self.label_encoder

  - Method predict_injury_risk(user_profile):
    * Extract fields: age, bmi, fitness_level_class, gender, fitness_experience_level, health_conditions, etc.
    * Build 12-feature vector:
      - Encode all categorical features (fitness_level, gender, experience, age_category, bmi_category)
      - Extract binary flags: has_health_conditions, previous_injury
      - Calculate derived scores: flexibility_score, strength_imbalance, training_frequency
    * Scale features using self.scaler.transform()
    * Predict using self.model.predict() and predict_proba()
    * Map prediction index to CLASS_NAMES
    * Extract max probability as confidence percentage
    * Extract risk_factors[] based on age>50, BMI>30, health_conditions, injury_history
    * Return: {injury_risk_score, injury_risk_class, injury_confidence, injury_risk_factors}
"""
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

class InjuryAssessorMLAgent:
    """Predicts injury risk using trained RandomForest model."""
    CLASS_NAMES = ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]

    def __init__(self, model_dir: str = "ml/models"):
        """Load trained model and preprocessors from disk."""
        self.model_dir = model_dir

        from ml.train_pipeline import load_ml_artifacts
        artifacts = load_ml_artifacts(
            model_dir,
            {
                "model": "injury_risk_model.pkl",
                "scaler": "injury_risk_scaler.pkl",
                "encoder": "injury_risk_encoder.pkl",
            },
            agent_name="Injury Assessor",
        )
        self.model = artifacts["model"]
        self.scaler = artifacts["scaler"]
        self.encoder = artifacts["encoder"]

    def _parse_binary_flags(self, user_profile: Dict[str, Any]) -> Tuple[str, int, str, int]:
        """Parses health conditions and injury history to return labels and binary flags."""
        health_conditions = user_profile.get("health_conditions", "None")
        has_health_conditions = 0 if health_conditions.lower() in ["none", "no", "", "n/a"] else 1

        injury_history = user_profile.get("past_injuries", "None")
        previous_injury = 0 if injury_history.lower() in ["none", "no", "", "n/a"] else 1
        
        return health_conditions, has_health_conditions, injury_history, previous_injury

    def _encode_categorical_features(self, user_profile: Dict[str, Any]) -> Dict[str, int]:
        """Maps categorical text values into machine readable integers using the encoder."""
        cats = {
            "Gender": user_profile.get("gender", "Male"),
            "Fitness_Experience": user_profile.get("fitness_experience", "Beginner"),
            "Age_Category": user_profile.get("age_category", "Adult"),
            "Fitness_level": user_profile.get("fitness_level_class", "Beginner"),
            "BMI_Category": user_profile.get("bmi_category", "Normal")
        }

        encoded_cats = {}
        for col_name, val in cats.items():
            if isinstance(self.encoder, dict) and col_name in self.encoder:
                try:
                    encoded_cats[col_name] = self.encoder[col_name].transform([val])[0]
                except (ValueError, KeyError):
                    encoded_cats[col_name] = 0
            else:
                encoded_cats[col_name] = 0
        return encoded_cats

    def _extract_risk_factors(self, age: float, bmi: float, has_health: int, health_str: str, prev_injury: int, injury_str: str) -> List[str]:
        """Identifies specific physiological and medical injury risk flags."""
        risk_factors: List[str] = []
        if age > 50:
            risk_factors.append("Age over 50 increases baseline injury risk")
        if bmi > 30:
            risk_factors.append("BMI over 30 put additional stress on joints.")
        if has_health == 1:
            risk_factors.append(f"Pre-existing health condition:{health_str}")
        if prev_injury == 1:
            risk_factors.append(f"History of previous injury:{injury_str}")

        if not risk_factors:
            risk_factors.append("No major physiological risk factors identified.")
        return risk_factors

    def predict_injury_risk(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Predict injury risk from user profile features."""
        if self.model is None or self.scaler is None:
            return {
                "injury_risk_score": 0,
                "injury_risk_class": "Unknown (Model Not Loaded)",
                "injury_confidence": 0.0,
                "injury_risk_factors": ["Model initialization failed"]
            }
        
        try:
            # 1. Extract standard numeric and frequency fields
            age = float(user_profile.get("age", 25))
            bmi = float(user_profile.get("bmi", 22.0))
            flexibility_score = float(user_profile.get("flexibility_score", 5.0))
            strength_imbalance = float(user_profile.get("strength_imbalance", 0.0))

            hours_str = str(user_profile.get("available_hours_per_week", "3"))
            training_frequency = float(''.join([x for x in hours_str if x.isdigit() or x == '.']) or 3.0)

            # 2. Extract Flags, Categorical, and Risk values using helpers
            health_conditions, has_health_conditions, injury_history, previous_injury = self._parse_binary_flags(user_profile)
            encoded_cats = self._encode_categorical_features(user_profile)
            risk_factors = self._extract_risk_factors(age, bmi, has_health_conditions, health_conditions, previous_injury, injury_history)

            # 3. Build the 12-feature matrix
            features = [
                age, bmi,
                encoded_cats["Gender"],
                encoded_cats["Fitness_Experience"],
                encoded_cats["Age_Category"],
                encoded_cats["Fitness_level"],
                encoded_cats["BMI_Category"],
                has_health_conditions,
                previous_injury,
                flexibility_score,
                strength_imbalance,
                training_frequency
            ]
            # Wrap in a DataFrame using the scaler's recorded feature names so
            # the names/order match training and sklearn does not warn.
            feature_vector = pd.DataFrame([features], columns=self.scaler.feature_names_in_)

            # 4. Scale and Predict
            scaled_features = self.scaler.transform(feature_vector)
            prediction_idx = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]

            # 5. Map to class and confidence score
            if isinstance(prediction_idx, str):
                predicted_class = prediction_idx
                score = self.CLASS_NAMES.index(predicted_class) if predicted_class in self.CLASS_NAMES else 0
            else:
                score = int(prediction_idx)
                predicted_class = self.CLASS_NAMES[score] if score < len(self.CLASS_NAMES) else "Unknown"

            confidence = float(np.max(probabilities) * 100)

            return {
                "injury_risk_score": score,
                "injury_risk_class": predicted_class,
                "injury_confidence": round(confidence, 2),
                "injury_risk_factors": risk_factors
            }

        except Exception as e:
            print(f"Error during injury prediction:{e}")
            return {
                "injury_risk_score": -1,
                "injury_risk_class": "Error processing data",
                "injury_confidence": 0.0,
                "injury_risk_factors": [str(e)]
            }
