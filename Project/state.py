"""
Central State Management Module

TODO: Define FitnessAssessmentState TypedDict with:
  - 8 INPUT FIELDS: age, height_cm, weight_kg, gender, fitness_experience,
                    health_conditions, fitness_goal, available_hours_per_week
  - 7 DERIVED FIELDS: bmi, age_category, bmi_category, parsed_profile,
                      validation_errors, parsing_complete, error_occurred
  - 3 NORMALIZER OUTPUTS: normalized_fitness_experience, normalized_health_conditions,
                          normalized_schedule
  - 4 FITNESS LEVEL OUTPUTS: fitness_level_score, fitness_level_class,
                             fitness_confidence, fitness_analysis_complete
  - 5 INJURY RISK OUTPUTS: injury_risk_score, injury_risk_class, injury_confidence,
                           injury_risk_factors, injury_assessment_complete
  - 9 WORKOUT PLAN OUTPUTS: workout_plan, weekly_schedule, workout_intensity_level,
                            workout_duration_per_session, workout_frequency_per_week,
                            workout_progression_timeline, workout_safety_notes,
                            workout_equipment_needed, workout_analysis_complete
  - 7 NUTRITION PLAN OUTPUTS: nutrition_plan, daily_calorie_target, macro_targets,
                              meal_suggestions, hydration_recommendation,
                              nutrition_timing_guidance, nutrition_analysis_complete
  - 11 RECOVERY & LIFESTYLE OUTPUTS: sleep_recommendations, rest_day_activities,
                                     mobility_work, stress_management_techniques,
                                     recovery_techniques, deload_strategy,
                                     schedule_integration, time_management_tips,
                                     habit_formation_strategies, adherence_tips,
                                     recovery_lifestyle_analysis_complete
  - 5 SYSTEM FIELDS: error_messages, analysis_timestamp, plan_generated, plan_id, user_name
  Total: ~85 fields organized in 10 categories
  Use TypedDict with total=False for optional fields

TODO: Implement get_initial_state() function:
  - Accept form_data: Dict[str, Any] parameter
  - Return FitnessAssessmentState TypedDict instance
  - Initialize all INPUT fields from form_data using .get() method
  - Initialize all OUTPUT fields to None, empty lists, or False as appropriate
  - Initialize error tracking fields (error_messages=[], parsing_complete=False, error_occurred=False)
  - Initialize system fields (plan_id=None, analysis_timestamp=None, plan_generated=False)
"""

from typing import TypedDict, List, Dict, Any, Optional


"""TODO: Define FitnessAssessmentState TypedDict class here"""
# total=False makes every field optional so nodes can update state incrementally
class FitnessAssessmentState(TypedDict, total=False):
    # 1. INPUT FIELDS (8 fields)
    age : Optional[int]
    height_cm : Optional[float]
    weight_kg : Optional[float]
    gender : Optional[str]
    fitness_experience : Optional[str]
    health_conditions : Optional[List[str]]
    fitness_goal : Optional[str]
    available_hours_per_week : Optional[float]

    #2. DERIVED FIELDS (7 Fields)
    bmi : Optional[float]
    age_category : Optional[str]
    bmi_category : Optional[str]
    parsed_profile : Optional[Dict[str,Any]]
    validation_errors : Optional[List[str]]
    parsing_complete : bool
    error_occurred : bool

    #3. NORMALIZER OUTPUTS(3 fields)
    normalized_fitness_experience : Optional[Dict[str,Any]]
    normalized_health_conditions : Optional[Dict[str,Any]]
    normalized_schedule : Optional[Dict[str,Any]]

    #4. FITNESS LEVEL OUTPUTS(4 fields)
    fitness_level_score : Optional[float]
    fitness_level_class : Optional[str]
    fitness_confidence : Optional[float]
    fitness_analysis_complete : bool

    #5. INJURY RISK OUTPUTS(5 fields)
    injury_risk_score : Optional[float]
    injury_risk_class : Optional[str]
    injury_confidence : Optional[float]
    injury_risk_factors : Optional[List[str]]
    injury_assessment_complete : bool

    #6. WORKOUT PLAN OUTPUTS (9 fields)
    workout_plan : Optional[Dict[str,any]]
    weekly_schedule : Optional[Dict[str,List[Dict[str,any]]]]
    workout_intensity_level : Optional[str]
    workout_duration_per_session : Optional[int]
    workout_frequency_per_week : Optional[int]
    workout_progression_timeline : Optional[str]
    workout_safety_notes : Optional[List[str]]
    workout_equipment_needed : Optional[List[str]]
    workout_analysis_complete : bool

    #7. NUTRITION PLAN OUTPUTS(7 fields)
    nutrition_plan : Optional[Dict[str,any]]
    daily_calorie_target : Optional[int]
    macro_targets : Optional[Dict[str,any]]
    meal_suggestions : Optional[List[Dict[str,any]]]
    hydration_recommendation : Optional[str]
    nutrition_timing_guidance : Optional[str]
    nutrition_analysis_complete : bool

    #8. RECOVERY & LIFESTYLE OUTPUTS(11 fields)
    sleep_recommendations : Optional[Dict[str,any]]
    rest_day_activities : Optional[List[str]]
    mobility_work : Optional[List[str]]
    stress_management_techniques : Optional[List[str]]
    recovery_techniques : Optional[List[str]]
    deload_strategy : Optional[str]
    schedule_integration : Optional[Dict[str,any]]
    time_management_tips : Optional[List[str]]
    habit_formation_strategies : Optional[List[str]]
    adherence_tips : Optional[List[str]]
    recovery_lifestyle_analysis_complete : bool

    #9. SYSTEM FIELDS(5 fields)
    error_messages : List[str]
    analysis_timestamp : Optional[str]
    plan_generated : bool
    plan_id : Optional[str]
    user_name : Optional[str]



"""TODO: Implement get_initial_state function here"""
def get_initial_state(form_data: Dict[str, Any]) -> FitnessAssessmentState:
    """
    Initialize FitnessAssessmentState with form input data.

    Args:
        form_data: Dictionary containing user form inputs

    Returns:
        FitnessAssessmentState with all fields initialized
    """
    state = FitnessAssessmentState()

    # 1. Map raw user inputs from the submitted form
    state["age"] = form_data.get("age")
    state["height_cm"] = form_data.get("height_cm")
    state["weight_kg"] = form_data.get("weight_kg")
    state["gender"] = form_data.get("gender")
    state["fitness_experience"] = form_data.get("fitness_experience")
    state["health_conditions"] = form_data.get("health_conditions")
    state["fitness_goal"] = form_data.get("fitness_goal")
    state["available_hours_per_week"] = form_data.get("available_hours_per_week")

    # 2. Derived fields — computed by form_parser_node; start as None/False
    state["bmi"] = None
    state["age_category"] = None
    state["bmi_category"] = None
    state["parsed_profile"] = None
    state["validation_errors"] = []
    state["parsing_complete"] = False
    state["error_occurred"] = False

    # 3. Normalizer outputs — populated by input_normalizer_node
    state["normalized_fitness_experience"] = None
    state["normalized_health_conditions"] = None
    state["normalized_schedule"] = None

    # 4. Fitness level outputs — populated by fitness_scorer_node
    state["fitness_level_score"] = None
    state["fitness_level_class"] = None
    state["fitness_confidence"] = None
    state["fitness_analysis_complete"] = False

    # 5. Injury risk outputs — populated by injury_assessor_node
    state["injury_risk_score"] = None
    state["injury_risk_class"] = None
    state["injury_confidence"] = None
    state["injury_risk_factors"] = []
    state["injury_assessment_complete"] = False

    # 6. Workout plan outputs — populated by workout_planner_node
    state["workout_plan"] = None
    state["weekly_schedule"] = None
    state["workout_intensity_level"] = None
    state["workout_duration_per_session"] = None
    state["workout_frequency_per_week"] = None

    # 7. Nutrition plan outputs — populated by nutrition_advisor_node
    state["nutrition_plan"] = None
    state["daily_calorie_target"] = None
    state["macro_targets"] = None
    state["meal_suggestions"] = []
    state["hydration_recommendation"] = None
    state["nutrition_timing_guidance"] = None
    state["nutrition_analysis_complete"] = False

    # 8. Recovery & lifestyle outputs — populated by recovery_lifestyle_optimizer_node
    state["sleep_recommendations"] = None
    state["rest_day_activities"] = []
    state["mobility_work"] = []
    state["stress_management_techniques"] = []
    state["recovery_techniques"] = []
    state["deload_strategy"] = None
    state["schedule_integration"] = None
    state["time_management_tips"] = []
    state["habit_formation_strategies"] = []
    state["adherence_tips"] = []
    state["recovery_lifestyle_analysis_complete"] = False

    # 9. System metadata — assigned in graph.py before workflow execution
    state["error_messages"] = []
    state["analysis_timestamp"] = None
    state["plan_generated"] = False
    state["plan_id"] = None
    state["user_name"] = form_data.get("user_name")

    return state
