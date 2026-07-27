"""Recovery & Lifestyle Optimizer LLM Agent — generates sleep, recovery, and adherence plans via LLM."""
"""
TODO: Implement RecoveryLifestyleOptimizerLLMAgent class:
  - Constructor __init__(client=None):
    * Require client parameter, raise ValueError if None
    * Store self.client = client

  - Method generate_recovery_lifestyle_plan(profile):
    * Extract fields: age, fitness_level_class, injury_risk_class, health_conditions, fitness_goal,
    workout_frequency_per_week, available_hours_per_week
    * Extract schedule preferences: preferred_days[], preferred_times[]
    * Create LLM prompt requesting JSON with:
      - Sleep recommendations: hours_per_night (integer), sleep_quality_tips[]
      - Rest day activities[] (active recovery options specific to injury_risk)
      - Mobility work[] (stretching/mobility routines)
      - Stress management techniques[]
      - Recovery techniques[] (foam rolling, massage, etc.)
      - Deload strategy: when/how to reduce intensity
      - Schedule integration: best_days[], best_times[], weekly_schedule_tips
      - Time management tips[] (for available_hours_per_week)
      - Habit formation strategies[]
      - Adherence tips[] (staying motivated, overcoming obstacles)
    * Call self.client.generate_structured_json() with required fields validation
    * Return: {sleep_recommendations{}, rest_day_activities[], mobility_work[], stress_management_techniques[],
    recovery_techniques[], deload_strategy, schedule_integration{}, time_management_tips[],
    habit_formation_strategies[], adherence_tips[]}
"""

from typing import Dict, Any
import json

from utils.gemini_client import invoke_and_parse_json


class RecoveryLifestyleOptimizerLLMAgent:
    """Generate recovery and lifestyle integration plans using LLM."""

    def __init__(self, client):
        """Initialize with LLM client."""
        if client is None:
          raise ValueError("LLM client must be provided.")
        self.client = client
        

    def generate_recovery_lifestyle_plan(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recovery and lifestyle integration plan."""
        
        default_structure = {
            "sleep_recommendations":{
            "hours_per_night":8,
            "sleep_quality_tips":["Keep room cool and dark","Avoid screens 1 hour before bed"]
          
            },
          "rest_day_activities":["Light walking","Yoga"],
          "mobility_work":["Dynamic stretching before workouts","Static stretching after workouts"],
          "stress_management_techniques":["Deep breathing exercise","Meditation"],
          "recovery_techniques":["Hydration","Adequate protein intake"],
          "deload_strategy":"Reduce training volume by 40% every 4th to 6th week to allow nervous system recovery.",
          "schedule_integration":{
            "best_days":["Monday","Wednesday","Friday"],
            "best_times":["Early morning","Evening"],
            "weekly_schedule_tips":"Space out your workouts to avoid training the same muscle groups on consecutive days."
            },
          "time_management_tips":["Prep meals in advance","Schedule workouts in your calender like meetings"],
          "habit_formation_strategies":["Start with small, manageable changes","Pair new habits with existing routines"],
          "adherence_tips":["Focus on how you feel, not just the mirror","Find a workout partner for accountability"]

          }
        
        schema_json = json.dumps(default_structure,indent=2)

        #1. Extract fields with safe defaults
        age = profile.get("age",25)
        injury_risk = profile.get("injury_risk_class","Moderate Risk")
        fitness_goal = profile.get("fitness_goal","General Fitness")
        fitness_level = profile.get("fitness_level_class", "Beginner")
        health_conditions = profile.get("health_conditions","None")

        # Extract schedule and availablility
        available_hours = profile.get("available_hours_per_week","3")
        workout_frequency = profile.get("workout_frequency_per_week", 3)

        # Safety extract nested schedule preferences if they exist from the normalizer
        normalized_schedule = profile.get("normalized_schedule",{})
        preferred_days = normalized_schedule.get("preferred_days",["Any day"])
        preferred_times = normalized_schedule.get("preferred_times",["Any time"])

        #2.Create the LLM prompt
        prompt = f"""
        You are an expert recovery and lifestyle plan generator. Create a customized daily Generate recovery and lifestyle integration plan on the  user's profile.

        USER PROFILE:
        - Age:{age}
        - Injury Risk:{injury_risk}
        - Fitness Level:{fitness_level}
        - Fitness Goal:{fitness_goal}
        - Health Conditions:{health_conditions}
        - Available Hours/Week: {available_hours}
        - Workout Frequency:{workout_frequency} times per week
        - Preferred Training Days:{','.join(preferred_days)}
        - Preferred Training Times:{','.join(preferred_times)}

        INSTRUCTIONS:
        Provide actionable advice tailored specifically to their injury risk profile and time constraints.

        OUTPUT FORMAT:
        Return ONLY a valid JSON object with the exact structure below. Do not include markdown code blocks, explanations, or  backticks.

        {schema_json} 
        """
        # Invoke the LLM, strip markdown fences, and parse JSON. Falls back to
        # safe defaults so the workflow can continue on any error.
        return invoke_and_parse_json(self.client, prompt, default=default_structure)