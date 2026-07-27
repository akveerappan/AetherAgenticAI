from typing import Dict, Any

from agents.injury_assessor_ml import InjuryAssessorMLAgent
from state import FitnessAssessmentState


def injury_assessor_node(state: FitnessAssessmentState, client=None) -> Dict[str, Any]:
    """
    TODO: Implement injury_assessor_node function:
    - Extract user profile from state: age, bmi, gender, fitness_goal, age_category, bmi_category, health_conditions
    - If parsed_profile exists in state, use it; otherwise, construct from individual fields
    - Add fitness_level_class from parallel fitness_scorer node output
    - Extract normalized_fitness_experience and add fitness_experience_level if available
    - Add available_hours_per_week from state
    - Create InjuryAssessorMLAgent instance
    - Call agent.predict_injury_risk(user_profile) with 12-feature vector
    - Extract injury_risk_factors[] from response (based on age>50, BMI>30, health_conditions, injury_history)
    - Return dict with: injury_risk_score, injury_risk_class (Low Risk|Moderate Risk|High Risk|Very High Risk),
                        injury_confidence (percentage 0-100), injury_risk_factors[], injury_assessment_complete=True
    - Handle exceptions: return zero score, "Unknown" class, 0.0 confidence, empty factors[], injury_assessment_complete=False
    """
    
    try:
        # Prefer the fully validated parsed_profile; fall back to raw state fields
        if state.get("parsed_profile"):
            user_profile = dict(state["parsed_profile"])
        else:
            user_profile = {
                "age" : state.get("age"),
                "bmi" : state.get("bmi"),
                "gender" : state.get("gender"),
                "fitness_goal" : state.get("fitness_goal"),
                "age_category" : state.get("age_category"),
                "bmi_category" : state.get("bmi_category"),
                "health_conditions" : state.get("health_conditions"),
                }

        # Resolve fitness_level_class from parallel fitness_scorer output or state
        if isinstance(client, dict) and "fitness_level_class" in client:
            fitness_class = client.get("fitness_level_class", "Unknown")
        else:
            fitness_class = state.get("fitness_level_class", "Unknown")

        user_profile["fitness_level_class"] = fitness_class

        # Enrich with LLM-normalised experience data if available
        if state.get("normalized_fitness_experience") is not None:
            user_profile["normalized_fitness_experience"] = state.get("normalized_fitness_experience")

        if state.get("fitness_experience_level") is not None:
            user_profile["fitness_experience_level"] = state.get("fitness_experience_level")

        # Add training frequency needed for the 12-feature vector
        user_profile["available_hours_per_week"] = state.get("available_hours_per_week")

        # Run ML prediction
        agent = InjuryAssessorMLAgent()
        prediction = agent.predict_injury_risk(user_profile)

        return{
            "injury_risk_score" : prediction.get("injury_risk_score", 0),
            "injury_risk_class" : prediction.get("injury_risk_class", "Unknown"),
            "injury_confidence"  : prediction.get("injury_confidence", 0.0),
            "injury_risk_factors" : prediction.get("injury_risk_factors", []),
            "injury_assessment_complete" : True
        }
    except Exception as e:
        print(f"Error executing injury_assessor_node : {str(e)}")
        return{
            "injury_risk_score" : 0,
            "injury_risk_class" : "Unknown",
            "injury_confidence"  : 0.0,
            "injury_risk_factors" : [],
            "injury_assessment_complete" : False
        }

