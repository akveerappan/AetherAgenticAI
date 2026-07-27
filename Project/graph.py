"""
Main Workflow Orchestration Module

TODO: Import required modules:
  - datetime: For generating analysis_timestamp
  - uuid4: For generating unique plan_id
  - Dict, Any from typing: For type hints
  - build_fitness_assessment_graph, get_workflow_structure from workflow module
  - FitnessAssessmentState, get_initial_state from state module

TODO: Implement assess_fitness() as main entry point:
  - Accept parameters: age, height_cm, weight_kg, gender, fitness_goal,
                       fitness_experience, health_conditions, available_hours_per_week,
                       client, user_name
  - Create form_data dict from all parameters
  - Call get_initial_state(form_data) to initialize state
  - Generate plan_id using uuid4()
  - Set analysis_timestamp to current datetime in ISO format
  - Call build_fitness_assessment_graph(client) to build LangGraph
  - Execute workflow using graph.invoke(state)
  - Handle exceptions gracefully - set error flags and return partial state
  - Return complete FitnessAssessmentState dict

TODO: Implement helper functions:
  - get_assessment_summary(assessment_result): Extract overview of assessment
  - get_workout_plan_details(assessment_result): Extract workout plan specifics
  - get_nutrition_plan_details(assessment_result): Extract nutrition plan specifics
  - get_recovery_lifestyle_details(assessment_result): Extract recovery & lifestyle details
  - get_workflow_info(): Return workflow structure metadata
  Each function should extract relevant fields from assessment_result and return organized dict
"""

from datetime import datetime, timezone
from typing import Dict, Any
from uuid import uuid4
from workflow import build_fitness_assessment_graph, get_workflow_structure
from state import FitnessAssessmentState, get_initial_state


""" TODO: Implement assess_fitness function with signature below"""
def assess_fitness(
    age: int,
    height_cm: float,
    weight_kg: float,
    gender: str,
    fitness_goal: str,
    fitness_experience: str,
    health_conditions: str,
    available_hours_per_week: str,
    client=None,
    user_name: str = None
) -> Dict[str, Any]:
    """
    Run complete fitness assessment workflow.

    Main entry point executing the 7-node LangGraph workflow pipeline.
    """
    # Pack all form inputs into a single dict for state initialisation
    form_data = {
        "age" : age,
        "height_cm" : height_cm,
        "weight_kg" : weight_kg,
        "gender" : gender,
        "fitness_goal" : fitness_goal,
        "fitness_experience" : fitness_experience,
        "health_conditions" : health_conditions,
        "available_hours_per_week" : available_hours_per_week,
        "user_name" : user_name
    }

    # Initialise state and attach unique plan identifiers
    state = get_initial_state(form_data)
    state["plan_id"] = str(uuid4())
    state["analysis_timestamp"] = datetime.now(timezone.utc).isoformat()

    try:
        # Build and execute the 7-node LangGraph pipeline
        graph = build_fitness_assessment_graph(client)
        assessment_result = graph.invoke(state)
        return dict(assessment_result)

    except Exception as e:
        # Return partial state with error flags so the caller can handle failures gracefully
        state["error_occurred"] = True
        state["validation_errors"] = [f"Critical Workflow Crash : {str(e)}"]
        state["parsing_complete"] = False
        return dict(state)

        
"""TODO: Implement get_assessment_summary(assessment_result)"""
def get_assessment_summary(assessment_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract overview with user_profile, derived_metrics, assessments, plans, system metadata."""
    return {
        "age" : assessment_result.get("age"),
        "bmi" : assessment_result.get("bmi"), 
        "weight_kg" : assessment_result.get("weight_kg"),
        "gender" : assessment_result.get("gender"),
        "age_category" : assessment_result.get("age_category"),
        "bmi_category" : assessment_result.get("bmi_category"),
        "parsing_complete" : assessment_result.get("parsing_complete"),
        "validation_errors" : assessment_result.get("validation_errors"),
        "error_occurred" : assessment_result.get("error_occurred"),
        "fitness_level_score" : assessment_result.get("fitness_level_score"),
        "fitness_level_class" : assessment_result.get("fitness_level_class"),
        "fitness_confidence" : assessment_result.get("fitness_confidence"),
        "fitness_analysis_complete" : assessment_result.get("fitness_analysis_complete"),
        "injury_risk_score" : assessment_result.get("injury_risk_score"),
        "injury_risk_class" : assessment_result.get("injury_risk_class"),
        "injury_confidence" : assessment_result.get("injury_confidence"),
        "injury_risk_factors" : assessment_result.get("injury_risk_factors"),
        "injury_assessment_complete" : assessment_result.get("injury_assessment_complete"),
        "workout_plan" : assessment_result.get("workout_plan"),
        "weekly_schedule" : assessment_result.get("weekly_schedule"),
        "workout_intensity_level" : assessment_result.get("workout_intensity_level"),
        "workout_duration_per_session" : assessment_result.get("workout_duration_per_session"),
        "workout_frequency_per_week" : assessment_result.get("workout_frequency_per_week"),
        "workout_progression_timeline" : assessment_result.get("workout_progression_timeline"),
        "workout_safety_notes" : assessment_result.get("workout_safety_notes"),
        "workout_equipment_needed" : assessment_result.get("workout_equipment_needed"),
        "workout_analysis_complete" : assessment_result.get("workout_analysis_complete"),
        "nutrition_plan" : assessment_result.get("nutrition_plan"),
        "daily_calorie_target" : assessment_result.get("daily_calorie_target"),
        "macro_targets" : assessment_result.get("macro_targets"),
        "meal_suggestions" : assessment_result.get("meal_suggestions"),
        "hydration_recommendation" : assessment_result.get("hydration_recommendation"),
        "nutrition_timing_guidance" : assessment_result.get("nutrition_timing_guidance"),
        "nutrition_analysis_complete" : assessment_result.get("nutrition_analysis_complete"),
        "sleep_recommendations" : assessment_result.get("sleep_recommendations"),
        "rest_day_activities" : assessment_result.get("rest_day_activities"),
        "mobility_work" : assessment_result.get("mobility_work"),
        "stress_management_techniques" : assessment_result.get("stress_management_techniques"),
        "recovery_techniques" : assessment_result.get("recovery_techniques"),
        "deload_strategy" : assessment_result.get("deload_strategy"),
        "schedule_integration" : assessment_result.get("schedule_integration"),
        "time_management_tips" : assessment_result.get("time_management_tips"),
        "habit_formation_strategies" : assessment_result.get("habit_formation_strategies"),
        "adherence_tips" : assessment_result.get("adherence_tips"),
        "recovery_lifestyle_analysis_complete" : assessment_result.get("recovery_lifestyle_analysis_complete"),
        "plan_id" : assessment_result.get("plan_id"),
        "plan_generated" : assessment_result.get("plan_generated"),
        "analysis_timestamp" : assessment_result.get("analysis_timestamp"),
        "user_name" : assessment_result.get("user_name"),
        "error_messages" : assessment_result.get("error_messages")        
    }


"""TODO: Implement get_workout_plan_details(assessment_result)"""
def get_workout_plan_details(assessment_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract: weekly_schedule, intensity_level, duration, frequency, progression, safety_notes, equipment."""
    return {
        "weekly_schedule" : assessment_result.get("weekly_schedule"),
        "workout_intensity_level" : assessment_result.get("workout_intensity_level"),
        "workout_duration_per_session" : assessment_result.get("workout_duration_per_session"),
        "workout_frequency_per_week" : assessment_result.get("workout_frequency_per_week"),
        "workout_progression_timeline" : assessment_result.get("workout_progression_timeline"),
        "workout_safety_notes" : assessment_result.get("workout_safety_notes"),
        "workout_equipment_needed" : assessment_result.get("workout_equipment_needed"),
    }


"""TODO: Implement get_nutrition_plan_details(assessment_result)"""
def get_nutrition_plan_details(assessment_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract: daily_calorie_target, macro_targets, meal_suggestions, hydration, timing_guidance."""
    return {
        "daily_calorie_target" : assessment_result.get("daily_calorie_target"),
        "macro_targets" : assessment_result.get("macro_targets"),
        "meal_suggestions" : assessment_result.get("meal_suggestions"),
        "hydration_recommendation" : assessment_result.get("hydration_recommendation"),
        "nutrition_timing_guidance" : assessment_result.get("nutrition_timing_guidance")
    }


"""TODO: Implement get_recovery_lifestyle_details(assessment_result)"""
def get_recovery_lifestyle_details(assessment_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract: sleep, rest_activities, mobility, stress_mgmt, recovery, deload, schedule, time_mgmt, habits, adherence."""
    return {
        "sleep_recommendations" : assessment_result.get("sleep_recommendations"),
        "rest_day_activities" : assessment_result.get("rest_day_activities"),
        "mobility_work" : assessment_result.get("mobility_work"),
        "stress_management_techniques" : assessment_result.get("stress_management_techniques"),
        "recovery_techniques" : assessment_result.get("recovery_techniques"),
        "deload_strategy" : assessment_result.get("deload_strategy"),
        "schedule_integration" : assessment_result.get("schedule_integration"),
        "time_management_tips" : assessment_result.get("time_management_tips"),
        "habit_formation_strategies" : assessment_result.get("habit_formation_strategies"),
        "adherence_tips" : assessment_result.get("adherence_tips")
    }


"""TODO: Implement get_workflow_info()"""
def get_workflow_info() -> Dict[str, Any]:
    """Return workflow structure metadata from build_fitness_assessment_graph."""
    return get_workflow_structure()
