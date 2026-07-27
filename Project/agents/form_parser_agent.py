"""Form Parser Agent — validates raw form inputs and derives BMI / age categories."""
"""
TODO: Implement FormParserAgent class:
  - Constants: VALID_GENDERS = [Male, Female, Other]
              VALID_FITNESS_GOALS = [Weight Loss, Muscle Building, Endurance/Cardio, General Fitness]

  - Method __init__(): Empty constructor

  - Method validate_and_parse(form_data: Dict[str, Any]) -> Dict[str, Any]:
    * Validate age: 18-100, is numeric
    * Validate height_cm: 100-250, is numeric
    * Validate weight_kg: 30-300, is numeric
    * Validate gender: Must be in VALID_GENDERS
    * Validate fitness_goal: Must be in VALID_FITNESS_GOALS
    * Validate fitness_experience: Required non-empty string
    * Validate health_conditions: Required string (can be "None")
    * Validate available_hours_per_week: Required non-empty string
    * Collect all errors in list, append error messages
    * Calculate BMI = weight_kg / (height_m^2) if numeric inputs valid
    * Categorize age: Young Adult (<30), Adult (30-44), Middle Aged (45-59), Senior (60+)
    * Categorize BMI: Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (30+)
    * Return dict with: parsed_profile (or None), bmi, age_category, bmi_category,
                        validation_errors[], parsing_complete (bool), error_occurred (bool)
"""
from typing import Dict, List, Any, Tuple

class FormParserAgent:
    """Validates user form inputs and calculates derived metrics."""
    VALID_GENDERS = ["Male", "Female", "Other"]
    VALID_FITNESS_GOALS = ["Weight Loss", "Muscle Building", "Endurance/Cardio", "General Fitness"]

    def __init__(self):
        """Initialize FormParserAgent."""
        pass

    def _validate_numeric(self, value: Any, min_val: float, max_val: float) -> Tuple[bool, float]:
        """Helper to safely parse and range-check numeric strings/values."""
        try:
            if value is None:
                return False, 0.0
            num = float(value)
            if min_val <= num <= max_val:
                return True, num
            return False, 0.0
        except (ValueError, TypeError):
            return False, 0.0

    def _validate_metrics(self, form_data: Dict[str, Any], parsed_data: Dict[str, Any], errors: List[str]) -> Tuple[bool, bool, bool]:
        """Validates age, height, and weight metrics."""
        is_valid_age, age = self._validate_numeric(form_data.get("age"), 18, 100)
        if not is_valid_age:
            errors.append("Age must be a numeric value between 18 and 100.")
        else:
            parsed_data["age"] = int(age)

        is_valid_height, height_cm = self._validate_numeric(form_data.get("height_cm"), 100, 250)
        if not is_valid_height:
            errors.append("Height must be a numeric value between 100 and 250 cm.")
        else:
            parsed_data["height_cm"] = height_cm

        is_valid_weight, weight_kg = self._validate_numeric(form_data.get("weight_kg"), 30, 300)
        if not is_valid_weight:
            errors.append("Weight must be a numeric value between 30 and 300 kg.")
        else:
            parsed_data["weight_kg"] = weight_kg

        return is_valid_age, is_valid_height, is_valid_weight

    def _validate_text_fields(self, form_data: Dict[str, Any], parsed_data: Dict[str, Any], errors: List[str]):
        """Validates text-based profile input values and categorical settings."""
        gender = form_data.get("gender")
        if gender not in self.VALID_GENDERS:
            errors.append(f"Gender must be one of: {','.join(self.VALID_GENDERS)}")
        else:
            parsed_data["gender"] = gender

        goal = form_data.get("fitness_goal")
        if goal not in self.VALID_FITNESS_GOALS:
            errors.append(f"Fitness goal must be one of: {','.join(self.VALID_FITNESS_GOALS)}")
        else:
            parsed_data["fitness_goal"] = goal

        exp = form_data.get("fitness_experience")
        if not isinstance(exp, str) or not exp.strip():
            errors.append("Fitness experience is required and must be a non-empty string.")
        else:
            parsed_data["fitness_experience"] = exp.strip()

        health = form_data.get("health_conditions")
        if not isinstance(health, str):
            errors.append("Health conditions must be a string (can be 'None').")
        else:
            parsed_data["health_conditions"] = health.strip() or "None"

        hours = form_data.get("available_hours_per_week")
        if not isinstance(hours, str) or not hours.strip():
            errors.append("Available hours per week is required and must be a non-empty string.")
        else:
            parsed_data["available_hours_per_week"] = hours.strip()

    def _validate_inputs(self, form_data: Dict[str, Any], errors: List[str]) -> Tuple[Dict[str, Any], bool, bool, bool]:
        """Validates all raw profile input values and aggregates errors."""
        parsed_data: Dict[str, Any] = {}
        
        is_valid_age, is_valid_height, is_valid_weight = self._validate_metrics(form_data, parsed_data, errors)
        self._validate_text_fields(form_data, parsed_data, errors)

        return parsed_data, is_valid_age, is_valid_height, is_valid_weight

    def _get_age_category(self, age: int) -> str:
        """Categorizes age into a specific range label."""
        if age < 30:
            return "Young Adult"
        if age < 45:
            return "Adult"
        if age < 60:
            return "Middle Aged"
        return "Senior"

    def _get_bmi_category(self, bmi: float) -> str:
        """Categorizes calculated BMI into health classifications."""
        if bmi < 18.5:
            return "Underweight"
        if bmi < 25.0:
            return "Normal"
        if bmi < 30.0:
            return "Overweight"
        return "Obese"

    def validate_and_parse(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate form inputs and return validation results with derived metrics."""
        errors: List[str] = []
        bmi = None
        age_category = None
        bmi_category = None

        parsed_data, is_valid_age, is_valid_height, is_valid_weight = self._validate_inputs(form_data, errors)

        # Process Age Category if valid
        if is_valid_age:
            age_category = self._get_age_category(parsed_data["age"])
            parsed_data["age_category"] = age_category

        # Process BMI and BMI Category if valid
        if is_valid_height and is_valid_weight:
            height_m = parsed_data["height_cm"] / 100.0
            bmi = round(parsed_data["weight_kg"] / (height_m ** 2), 2)
            parsed_data["bmi"] = bmi
            
            bmi_category = self._get_bmi_category(bmi)
            parsed_data["bmi_category"] = bmi_category

        error_occurred = len(errors) > 0

        # Only return a parsed_profile when all validations pass
        return {
            "parsed_profile": None if error_occurred else parsed_data,
            "bmi": bmi,
            "age_category": age_category,
            "bmi_category": bmi_category,
            "validation_errors": errors,
            "parsing_complete": not error_occurred,
            "error_occurred": error_occurred
        }
