from typing import Dict, Any

from agents.recovery_lifestyle_optimizer_llm import RecoveryLifestyleOptimizerLLMAgent
from state import FitnessAssessmentState


def recovery_lifestyle_optimizer_node(state: FitnessAssessmentState, client) -> Dict[str, Any]:
    """
    TODO: Implement recovery_lifestyle_optimizer_node function:
    - Extract profile data from state: age, fitness_level_class, injury_risk_class, health_conditions, fitness_goal,
                                       workout_frequency_per_week, available_hours_per_week
    - Extract schedule preferences from normalized_schedule if available: preferred_days[], preferred_times[]
    - Create RecoveryLifestyleOptimizerLLMAgent instance with client
    - Call agent.generate_recovery_lifestyle_plan(profile) to create LLM-generated recovery/lifestyle plan
    - Extract from LLM response:
      * sleep_recommendations: {hours_per_night (int), sleep_quality_tips[]}
      * rest_day_activities[] (active recovery options specific to injury_risk)
      * mobility_work[] (stretching/mobility routines)
      * stress_management_techniques[]
      * recovery_techniques[] (foam rolling, massage, etc.)
      * deload_strategy (string describing when/how to reduce intensity)
      * schedule_integration: {best_days[], best_times[], weekly_schedule_tips}
      * time_management_tips[] (for available_hours_per_week)
      * habit_formation_strategies[]
      * adherence_tips[] (staying motivated, overcoming obstacles)
    - Return dict with: sleep_recommendations, rest_day_activities[], mobility_work[], stress_management_techniques[],
                        recovery_techniques[], deload_strategy, schedule_integration, time_management_tips[], habit_formation_strategies[],
                        adherence_tips[], recovery_lifestyle_analysis_complete=True
    - Handle exceptions: return None objects/values with empty lists, recovery_lifestyle_analysis_complete=False
    """
    try :
      # Build profile combining workout results and user context
      profile = {
        "age" : state.get("age"),
        "fitness_level_class" : state.get("fitness_level_class"),
        "injury_risk_class" : state.get("injury_risk_class"),
        "health_conditions" : state.get("health_conditions"),
        "fitness_goal" : state.get("fitness_goal"),
        "workout_frequency_per_week" : state.get("workout_frequency_per_week"),
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

      # Generate LLM-based recovery and lifestyle plan
      agent = RecoveryLifestyleOptimizerLLMAgent(client) 
      agent_response = agent.generate_recovery_lifestyle_plan(profile)

      # Guard against non-dict or empty LLM responses
      if not agent_response or not isinstance(agent_response,dict):
        agent_response = {}

      # Ensure sleep_recommendations is always a dict
      sleep_rec = agent_response.get("sleep_recommendations",{})
      if not isinstance(sleep_rec, dict):
        sleep_rec = {"hours_per_night" : 0, "sleep_quality_tips" : []}

      # Ensure schedule_integration is always a dict
      schedule_int = agent_response.get("schedule_integration",{})
      if not isinstance(schedule_int, dict):
        schedule_int = {"best_days" : [],"best_times" : [], "weekly_schedule_tips" : ""}

      def safe_list(key_name : str) -> list:
        lst =agent_response.get("key_name",[])
        return list(lst) if isinstance(lst, list) else []
      
      return {
        "sleep_recommendations" : {
          "hours_per_night" : sleep_rec.get("hours_per_night", 0),
          "sleep_quality_tips" : sleep_rec.get("sleep_quality_tips", [])
        },
        "rest_day_activities" : safe_list("rest_day_activities"),
        "mobility_work" : safe_list("mobility_work"),
        "stress_management_techniques" : safe_list("stress_management_techniques"),
        "recovery_techniques" : safe_list("recovery_techniques"),
        "deload_strategy" : agent_response.get("deload_strategy"),
        "schedule_integration" :{
          "best_days" : schedule_int.get("best_days",[]),
          "best_times" : schedule_int.get("best_times",[]),
          "weekly_schedule_tips" : schedule_int.get("weekly_schedule_tips","")
        },
        "time_management_tips" : safe_list("time_management_tips"),
        "habit_formation_strategies" : safe_list("habit_formation_strategies"),
        "adherence_tips" : safe_list("adherence_tips"),
        "recovery_lifestyle_analysis_complete" : True
      }

    except Exception as e:
        print(f"Error executing recovery_lifestyle_optimizer_node : {str(e)}")
        return {
        "sleep_recommendations" :None,
        "rest_day_activities" : [],
        "mobility_work" : [],
        "stress_management_techniques" : [],
        "recovery_techniques" : [],
        "deload_strategy" : None,
        "schedule_integration" : None,
        "time_management_tips" : [],
        "habit_formation_strategies" : [],
        "adherence_tips" : [],
        "recovery_lifestyle_analysis_complete" : False
      }