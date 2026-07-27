from typing import Dict, Any

from agents.input_normalizer_llm import InputNormalizerLLMAgent
from state import FitnessAssessmentState

import logging
import sys
      
logger = logging.getLogger(__name__)

def input_normalizer_node(state: FitnessAssessmentState, client) -> Dict[str, Any]:
    """
    TODO: Implement input_normalizer_node function:
    - Extract inputs from state: fitness_experience, health_conditions, available_hours_per_week (free-text strings)
    - Create InputNormalizerLLMAgent instance with client
    - Call agent.normalize_inputs(fitness_experience, health_conditions, available_hours_per_week)
    - Extracts from LLM response:
      * normalized_fitness_experience: {experience_level (enum), years_active (int), activity_description (str)}
      * normalized_health_conditions: {conditions[] (list), severity_assessment (enum), exercise_limitations[] (list), cleared_for_exercise (bool)}
      * normalized_schedule: {estimated_hours_per_week (float), preferred_days[] (list), preferred_times[] (list), schedule_constraints (str)}
    - Return dict with: normalized_fitness_experience, normalized_health_conditions, normalized_schedule
    - Handle exceptions: return None values for all fields, log error
    """
    try:
      # Extract free-text fields that need LLM-based structuring
      fitness_experience = state.get("fitness_experience") 
      health_conditions = state.get("health_conditions")    
      available_hours_per_week = state.get("available_hours_per_week")

      # Call LLM agent to parse all three free-text fields in one request
      agent = InputNormalizerLLMAgent(client) 
      llm_response = agent.normalize_inputs(
        fitness_experience=fitness_experience, 
        health_conditions=health_conditions, 
        available_hours_per_week=available_hours_per_week)
      
      # Extract each structured block with safe fallback defaults
      normalized_fitness_experience = llm_response.get("normalized_fitness_experience",
                                      {"experience_level":"beginner","years_active": 0,"activity_description": ""})
      normalized_health_conditions = llm_response.get("normalized_health_conditions",
                                      {"conditions":[],"severity_assessment":None, "exercise_limitations":[],"cleared_for_exercise": False})
      normalized_schedule = llm_response.get("normalized_schedule",
                                      {"estimated_hours_per_week" : 0.0,"preferred_days":[],"preferred_times":[],"schedule_constraints": ""})
      return {
        "normalized_fitness_experience":normalized_fitness_experience,
        "normalized_health_conditions" : normalized_health_conditions,
        "normalized_schedule" : normalized_schedule
      }
    except Exception as e:
      logger.error(f"Error executing input_normalizer_node : {str(e)}", exc_info = True )
      # Return None values so downstream nodes can handle missing normalization gracefully
      return {
        "normalized_fitness_experience":None,
        "normalized_health_conditions" : None,
        "normalized_schedule" : None,
        "error_occured" : True,
        "error_messages" : [f"Input Normalizer Exception : {str(e)}"]
      }