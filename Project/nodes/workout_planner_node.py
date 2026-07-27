from typing import Dict, Any

from agents.workout_plan_generator_llm import WorkoutPlanGeneratorLLMAgent
from state import FitnessAssessmentState


def workout_planner_node(state: FitnessAssessmentState, client) -> Dict[str, Any]:
    """
    TODO: Implement workout_planner_node function:
    - Merge output from parallel fitness_scorer and injury_assessor nodes
    - Extract profile data from state: fitness_level_class, injury_risk_class, fitness_goal, available_hours_per_week, health_conditions, gender, age
    - Extract schedule preferences from normalized_schedule if available: preferred_days[], preferred_times[]
    - Create WorkoutPlanGeneratorLLMAgent instance with client
    - Call agent.generate_workout_plan(profile) to create LLM-generated workout plan
    - Extract from LLM response:
      * weekly_schedule: {day: [{exercise_name, sets, reps, rest_period}]}
      * workout_intensity_level (Light|Moderate|Vigorous)
      * workout_duration_per_session (integer minutes)
      * workout_frequency_per_week (integer)
      * workout_progression_timeline (4-6 weeks or 8-12 weeks)
      * workout_safety_notes[] (accounting for injury_risk and health_conditions)
      * workout_equipment_needed[] (or empty for bodyweight only)
    - Return dict with: workout_plan, weekly_schedule, workout_intensity_level, workout_duration_per_session,
                        workout_frequency_per_week, workout_progression_timeline, workout_safety_notes[], workout_equipment_needed[], workout_analysis_complete=True
    - Handle exceptions: return None plan and null values with empty lists, workout_analysis_complete=False
    """
    try :
      # Build profile dict from merged fitness_scorer + injury_assessor state outputs
      profile = {
        "age" : state.get("age"),
        "fitness_level_class" : state.get("fitness_level_class", "Unknown"),
        "injury_risk_class" : state.get("injury_risk_class", "Unknown"),
        "health_conditions" : state.get("health_conditions"),
        "fitness_goal" : state.get("fitness_goal"),
        "gender" : state.get("gender"),
        "available_hours_per_week" : state.get("available_hours_per_week")
      }

      # Attach schedule preferences from the normalizer if present
      normalized_schedule = state.get("normalized_schedule")
      if isinstance(normalized_schedule, dict):
        profile["preferred_days"] = normalized_schedule.get("preferred_days",[])
        profile["preferred_times"] = normalized_schedule.get("preferred_times",[])
      else:
        profile["preferred_days"] = []
        profile["preferred_times"] = []

      # Generate LLM-based workout plan
      agent = WorkoutPlanGeneratorLLMAgent(client)
      agent_response = agent.generate_workout_plan(profile)

      # Guard against non-dict or empty LLM responses
      if not agent_response or not isinstance(agent_response,dict):
        agent_response = {}

      # Ensure weekly_schedule is always a dict
      weekly_schedule = agent_response.get("weekly_schedule",{})
      if not isinstance(weekly_schedule, dict):
        weekly_schedule = {}

      return {
        "workout_plan" : agent_response if agent_response else {},
        "weekly_schedule" : weekly_schedule,
        "workout_intensity_level" : agent_response.get("workout_intensity_level"),
        "workout_duration_per_session" : agent_response.get("workout_duration_per_session"),
        "workout_frequency_per_week" : agent_response.get("workout_frequency_per_week"),
        "workout_progression_timeline" : agent_response.get("workout_progression_timeline"),
        "workout_safety_notes" : list(agent_response.get("workout_safety_notes",[])),
        "workout_equipment_needed" : list(agent_response.get("workout_equipment_needed",[])),
        "workout_analysis_complete" : True
      }

    except Exception as e:
      print(f"Error executing workout_planner_node : {str(e)}")
      return {
        "workout_plan" : None,
        "weekly_schedule" : {},
        "workout_intensity_level" : None,
        "workout_duration_per_session" : None,
        "workout_frequency_per_week" : None,
        "workout_progression_timeline" : None,
        "workout_safety_notes" : [],
        "workout_equipment_needed" : [],
        "workout_analysis_complete" : False
      }