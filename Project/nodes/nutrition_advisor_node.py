from typing import Dict, Any

from agents.nutrition_plan_generator_llm import NutritionPlanGeneratorLLMAgent
from state import FitnessAssessmentState


def nutrition_advisor_node(state: FitnessAssessmentState, client) -> Dict[str, Any]:
    """
    TODO: Implement nutrition_advisor_node function:
    - Extract profile data from state: age, weight_kg, height_cm, gender, fitness_goal, fitness_level_class, bmi, workout_frequency_per_week
    - Create NutritionPlanGeneratorLLMAgent instance with client
    - Call agent.generate_nutrition_plan(profile) to create LLM-generated nutrition plan
    - Extract from LLM response:
      * daily_calorie_target (integer kcal, calculated using Harris-Benedict equation with activity multiplier)
      * macro_targets: {protein_g (int), carbs_g (int), fat_g (int)}
      * meal_suggestions[]: {meal_name, foods[], protein_g, carbs_g, fat_g, calories} (3-5 meals with Indian cuisine options)
      * hydration_recommendation (string describing daily water intake)
      * nutrition_timing_guidance (string about pre/post workout nutrition)
    - Return dict with: nutrition_plan, daily_calorie_target, macro_targets, meal_suggestions[], hydration_recommendation,
                        nutrition_timing_guidance, nutrition_analysis_complete=True
    - Handle exceptions: return None plan and null values with empty meal list, nutrition_analysis_complete=False
    """
    try :
      # Build profile with fields required by the Harris-Benedict calorie calculation
      profile = {
        "age" : state.get("age"),
        "weight_kg" : state.get("weight_kg"),
        "height_cm" : state.get("height_cm"),
        "gender" : state.get("gender"),
        "fitness_goal" : state.get("fitness_goal"),
        "fitness_level_class" : state.get("fitness_level_class"),
        "bmi" : state.get("bmi"),
        "workout_frequency_per_week" : state.get("workout_frequency_per_week")
      }

      # Generate LLM-based nutrition plan
      agent = NutritionPlanGeneratorLLMAgent(client) 
      agent_response = agent.generate_nutrition_plan(profile)

      # Guard against non-dict or empty LLM responses
      if not agent_response or not isinstance(agent_response,dict):
          agent_response = {}

      # Ensure macro_targets is always a dict with numeric defaults
      macro_targets = agent_response.get("macro_targets",{})
      if not isinstance(macro_targets, dict):
        macro_targets = {"protein_g" : 0, "carbs_g" : 0, "fat_g" : 0}

      # Ensure meal_suggestions is always a list
      meal_suggestions = agent_response.get("meal_suggestions",[])
      if not isinstance(meal_suggestions, list):
        meal_suggestions = []

      return {
        "nutrition_plan" : agent_response if agent_response else {},
        "daily_calorie_target" : agent_response.get("daily_calorie_target"),
        "macro_targets" : {
          "protein_g" : macro_targets.get("protein_g",0),
          "carbs_g" : macro_targets.get("carbs_g",0),
          "fat_g" : macro_targets.get("fat_g",0)
        },
        "meal_suggestions" : meal_suggestions,
        "hydration_recommendation" : agent_response.get("hydration_recommendation"),
        "nutrition_timing_guidance" : agent_response.get("nutrition_timing_guidance"),
        "nutrition_analysis_complete" : True
      }
      
    except Exception as e:
      print(f"Error executing nutrition_advisor_node : {str(e)}")
      return {
        "nutrition_plan" : None,
        "daily_calorie_target" : None,
        "macro_targets" : None,
        "meal_suggestions" : [],
        "hydration_recommendation" : None,
        "nutrition_timing_guidance" : None,
        "nutrition_analysis_complete" : False
      }
      

