from typing import Dict, Any

from agents.form_parser_agent import FormParserAgent
from state import FitnessAssessmentState


def form_parser_node(state: FitnessAssessmentState, client=None) -> Dict[str, Any]:
    """
    TODO: Implement form_parser_node function:
    - Extract form inputs from state: age, height_cm, weight_kg, gender, fitness_goal, fitness_experience, health_conditions, available_hours_per_week
    - Create FormParserAgent instance
    - Call agent.validate_and_parse(form_data)
    - Return dict with: parsed_profile (if valid), bmi (calculated), age_category (Young Adult|Adult|Middle Aged|Senior),
                        bmi_category (Underweight|Normal|Overweight|Obese), validation_errors[], parsing_complete (bool), error_occurred (bool)
    - Handle exceptions: return null parsed_profile with error message and error_occurred=True
    """
    try:
        _ = client

        # Collect raw user inputs from workflow state
        form_data = {
            "age" : state.get("age"),
            "height_cm" : state.get("height_cm"),
            "weight_kg" : state.get("weight_kg"),
            "gender" : state.get("gender"),
            "fitness_goal" : state.get("fitness_goal"),
            "fitness_experience" : state.get("fitness_experience"),
            "health_conditions" : state.get("health_conditions"),
            "available_hours_per_week" : state.get("available_hours_per_week")
        }

        # Validate inputs and compute derived metrics (BMI, age/BMI categories)
        agent = FormParserAgent()
        parsed_form = agent.validate_and_parse(form_data)

        # Return validated profile and derived metrics to state
        return {
                "parsed_profile" : parsed_form.get("parsed_profile", None),
                "bmi" : parsed_form.get("bmi", None),
                "age_category"  : parsed_form.get("age_category", None),
                "bmi_category"  : parsed_form.get("bmi_category", None),
                "validation_errors"  : parsed_form.get("validation_errors", []),
                "parsing_complete" : parsed_form.get("parsing_complete", False),
                "error_occurred" : parsed_form.get("error_occurred", False)
        }

    except Exception as e:
        print(f"Error executing form_parser_node : {str(e)}")
        # Return safe defaults so the workflow can continue
        return {
                "parsed_profile" : None,
                "bmi" : None,
                "age_category"  : None,
                "bmi_category"  : None,
                "validation_errors"  : [],
                "parsing_complete" : False,
                "error_occurred" : True
        }