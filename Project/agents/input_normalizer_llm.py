"""Input Normalizer LLM Agent — converts free-text inputs into structured JSON via LLM."""
"""
TODO: Implement InputNormalizerLLMAgent class:
  - Constructor __init__(client=None):
    * Require client parameter, raise ValueError if None
    * Store self.client = client

  - Method normalize_inputs(fitness_experience, health_conditions, available_hours_per_week):
    * Accept 3 free-text string parameters
    * Create LLM prompt requesting JSON with structured extraction:
      - From fitness_experience: experience_level (enum), years_active (int), activity_description (str)
      - From health_conditions: conditions[] (list), severity_assessment (enum), exercise_limitations[] (list),
      cleared_for_exercise (bool)
      - From available_hours_per_week: estimated_hours_per_week (float), preferred_days[] (list), preferred_times[] (list),
      schedule_constraints (str)
    * Call self.client.generate_structured_json() with required fields validation
    * Return dict with: normalized_fitness_experience{}, normalized_health_conditions{}, normalized_schedule{}
"""
import json
from typing import Dict, Any

from utils.gemini_client import invoke_and_parse_json


class InputNormalizerLLMAgent:
    """Parse natural language inputs using LLM to extract structured information."""

    def __init__(self, client=None):
        """Initialize with LLM client."""
        if client is None:
          raise ValueError("LLM client must be provided.")
        self.client = client

        

    def normalize_inputs(self, fitness_experience: str, health_conditions: str, available_hours_per_week: str) -> Dict[str, Any]:
        """Parse natural language inputs and extract structured normalized information."""

        default_structure = {
            "normalized_fitness_experience":{
              "experience_level":"Beginner",
              "years_active":0,
              "activity_description":str(fitness_experience)
            },
            "normalized_health_conditions":{
              "conditions":[str(health_conditions)] if health_conditions else[],
              "severity_assessment":"Unknown",
              "exercise_limitation":[],
              "cleared_for_exercise":True
            },
            "normalized_schedule":{
              "estimated_hours_per_week":3.0,
              "preferred_days":[],
              "preferred_times":[],
              "schedule_constraints":str(available_hours_per_week)
            }
          }

        schema_json = json.dumps(default_structure,indent=2)

        #Craft a highly specific prompt to enforce JSON structure
        prompt = f"""
        You are a clinical fitness AI assistant, Your task is to extract structured data from free-text user inputs.

        INPUTS:
        1.Fitness Experience:"{fitness_experience}"
        2.Health Conditions:"{health_conditions}"
        3.Available Schedule:"{available_hours_per_week}"

        REQUIREMENT:
        Extract the information into the exact JSON structure below. Make reasonable inferences based on the text.
        If information is missing, use logical defaults (e.g., 0 for years active, [] for empty lists.)

        {schema_json}

        OUTPUT ONLY VALID JSON. Do not include markdown code blocks, explanations, or backticks.
        """
        # Invoke the LLM, strip markdown fences, and parse JSON. Falls back to
        # safe defaults so the workflow can continue on any error.
        return invoke_and_parse_json(self.client, prompt, default=default_structure)

      
        
