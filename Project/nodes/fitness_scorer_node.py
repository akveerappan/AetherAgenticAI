from typing import Dict, Any

from agents.fitness_scorer_ml import FitnessScorerMLAgent
from state import FitnessAssessmentState


def fitness_scorer_node(state: FitnessAssessmentState, client=None) -> Dict[str, Any]:
    """
    TODO: Implement fitness_scorer_node function:
    - Extract user profile from state: age, bmi, weight_kg, gender, fitness_goal, age_category, bmi_category
    - If parsed_profile exists in state, use it; otherwise, construct from individual fields
    - Extract normalized_fitness_experience from state and add fitness_experience_level if available
    - Create FitnessScorerMLAgent instance
    - Call agent.predict_fitness_level(user_profile) with 11-feature vector
    - Return dict with: fitness_level_score, fitness_level_class (Beginner|Intermediate|Advanced|Athlete),
                        fitness_confidence (percentage 0-100), fitness_analysis_complete=True
    - Handle exceptions: return zero score, "Unknown" class, 0.0 confidence, fitness_analysis_complete=False
    """
    try:
        _ = client

        # Prefer the fully validated parsed_profile; fall back to raw state fields
        if "parsed_profile" in state and state.get("parsed_profile"):
            user_profile = dict(state["parsed_profile"])
        else:
            user_profile = {
                "age" : state.get("age"),
                "bmi" : state.get("bmi"),
                "weight_kg" : state.get("weight_kg"),
                "gender" : state.get("gender"),
                "fitness_goal" : state.get("fitness_goal"),
                "age_category" : state.get("age_category"),
                "bmi_category" : state.get("bmi_category")
                }

        # Enrich profile with LLM-normalised experience level if available
        if state.get("normalized_fitness_experience") is not None:
            user_profile["fitness_experience_level"] = state.get("fitness_experience_level")

        # Run ML prediction
        agent = FitnessScorerMLAgent()
        prediction = agent.predict_fitness_level(user_profile)

        return {
            "fitness_level_score" : prediction.get("fitness_level_score", 0.0),
            "fitness_level_class" : prediction.get("fitness_level_class", "Unknown"),
            "fitness_confidence"  : prediction.get("fitness_confidence", 0.0),
            "fitness_analysis_complete" : True
        }
    except Exception as e:
        print(f"Error executing fitness_scorer_node : {str(e)}")
        return {
            "fitness_level_score" : 0.0,
            "fitness_level_class" : "Unknown",
            "fitness_confidence"  : 0.0,
            "fitness_analysis_complete" : False
        }
