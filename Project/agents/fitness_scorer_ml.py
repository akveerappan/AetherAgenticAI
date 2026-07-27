"""Fitness Scorer ML Agent — classifies fitness level using a trained RandomForest model."""
"""
TODO: Implement FitnessScorerMLAgent class:
  - Constants: CLASS_NAMES = [Beginner, Intermediate, Advanced, Athlete]

  - Constructor __init__(model_dir="ml/models"):
    * Load 3 pickle files: fitness_level_model.pkl, fitness_level_scaler.pkl, fitness_level_encoder.pkl
    * Store as self.model, self.scaler, self.label_encoder

  - Method predict_fitness_level(user_profile):
    * Extract fields: age, bmi, weight_kg, gender, fitness_experience_level, age_category, fitness_goal, etc.
    * Build 11-feature vector with encoded categorical features
    * Handle available_hours as string parsing if needed
    * Calculate activity_score from available_hours and other factors
    * Scale features using self.scaler.transform()
    * Predict using self.model.predict() and predict_proba()
    * Map prediction index to CLASS_NAMES
    * Extract max probability as confidence percentage
    * Return: {fitness_level_score, fitness_level_class, fitness_confidence}
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


class FitnessScorerMLAgent:
    """Classifies fitness level using trained RandomForest model."""
    CLASS_NAMES = ["Beginner", "Intermediate", "Advanced", "Athlete"]

    def __init__(self, model_dir: str = "ml/models"):
        """Load trained model and preprocessors from disk."""
        self.model_dir = model_dir

        from ml.train_pipeline import load_ml_artifacts
        artifacts = load_ml_artifacts(
            model_dir,
            {
                "model": "fitness_level_model.pkl",
                "scaler": "fitness_level_scaler.pkl",
                "encoder": "fitness_level_encoder.pkl",
            },
            agent_name="Fitness Scorer",
        )
        self.model = artifacts["model"]
        self.scaler = artifacts["scaler"]
        self.encoder = artifacts["encoder"]

    def _parse_hours_and_score(self, user_profile: Dict[str, Any], weight_kg: float) -> Tuple[float, float]:
        """Extracts available weekly hours and computes the resulting activity score metrics."""
        hours_str = str(user_profile.get("available_hours_per_week") or user_profile.get("available_hours"))
        digits = [x for x in hours_str if x.isdigit() or x == '.']
        available_hours = float(''.join(digits) if digits else 3.0)
        activity_score = (available_hours * 1.5) - (weight_kg * 0.01)
        return available_hours, activity_score

    def _get_fallback_bmi_category(self, bmi: float) -> str:
        """Determines the physiological category ranking matching the derived BMI metrics."""
        bmi_bounds = [18.5, 25.0, 30.0]
        bmi_labels = ["Underweight", "Normal", "Overweight", "Obese"]
        label_index = sum(bmi >= threshold for threshold in bmi_bounds)
        return bmi_labels[label_index]

    def _get_fallback_hours_category(self, hours: float) -> str:
        """Determines the engagement frequency intensity rating based on training duration hours."""
        if hours < 3.0:
            return "Low"
        return "Medium" if hours <= 5.0 else "High"

    def _encode_categorical_features(self, user_profile: Dict[str, Any], bmi: float, hours: float) -> Dict[str, int]:
        """Maps alphanumeric categorization parameters to numerical label vector indices."""
        bmi_cat = user_profile.get("bmi_category") or self._get_fallback_bmi_category(bmi)
        hours_cat = user_profile.get("hours_category") or self._get_fallback_hours_category(hours)

        raw_categorical_inputs = [
            ("Gender", user_profile.get("gender")),
            ("Fitness_Experience", user_profile.get("experience")),
            ("Age_Category", user_profile.get("age_category")),
            ("Fitness_Goal", user_profile.get("fitness_goal")),
            ("BMI_Category", bmi_cat),
            ("Hours_Category", hours_cat)
        ]

        encoded_cats = {}
        for attribute_key, textual_token in raw_categorical_inputs:
            resolved_index_id = 0
            if isinstance(self.encoder, dict) and attribute_key in self.encoder:
                try:
                    resolved_index_id = self.encoder[attribute_key].transform([textual_token])[0]
                except (ValueError, KeyError):
                    pass
            encoded_cats[attribute_key] = resolved_index_id
        return encoded_cats

    def predict_fitness_level(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Predict fitness level from user profile features."""
        # Fallback if models aren't loaded
        if self.model is None or self.scaler is None:
            return {
                "fitness_level_score": 0,
                "fitness_level_class": "Unknown (Model Not Loaded)",
                "fitness_confidence": 0.0
            }
        
        try:
            # 1. Extract standard numeric fields
            age = float(user_profile.get("age", 25))
            bmi = float(user_profile.get("bmi", 22.0))
            weight_kg = float(user_profile.get("weight_kg", 70.0))

            # 2. Extract hours and track classifications vectors
            available_hours, activity_score = self._parse_hours_and_score(user_profile, weight_kg)
            encoded_cats = self._encode_categorical_features(user_profile, bmi, available_hours)

            # 3. Build the 11-features
            ordered_keys = ["Gender", "Fitness_Experience", "Age_Category", "Fitness_Goal", "BMI_Category", "Hours_Category"]
            transformed_categorical_list = [encoded_cats.get(schema_key, 0) for schema_key in ordered_keys]
            
            # Combine components dynamically via mathematical array joins
            features = [age, bmi, weight_kg, available_hours] + transformed_categorical_list + [activity_score]
            # Wrap in a DataFrame using the scaler's recorded feature names so
            # the names/order match training and sklearn does not warn.
            feature_vector = pd.DataFrame([features], columns=self.scaler.feature_names_in_)

            # 4. Scale and Predict
            scaled_features = self.scaler.transform(feature_vector)
            prediction_idx = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]

            # 5. Map to class and confidence score
            prediction_matches_text = isinstance(prediction_idx, str)
            if prediction_matches_text:
                score = self.CLASS_NAMES.index(prediction_idx) if prediction_idx in self.CLASS_NAMES else 0
                predicted_class = prediction_idx
            else:
                score = int(prediction_idx)
                predicted_class = self.CLASS_NAMES[score] if score < len(self.CLASS_NAMES) else "Unknown"

            confidence = float(np.max(probabilities) * 100)

            return {
                "fitness_level_score": score,
                "fitness_level_class": predicted_class,
                "fitness_confidence": round(confidence, 2)
            }

        except Exception as e:
            print(f"Error during fitness prediction: {e}")
            return {
                "fitness_level_score": -1,
                "fitness_level_class": "Error parsing data",
                "fitness_confidence": 0.0
            }
