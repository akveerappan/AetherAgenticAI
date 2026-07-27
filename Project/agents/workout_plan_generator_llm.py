"""Workout Plan Generator LLM Agent — generates a personalised weekly training plan via LLM."""
"""
TODO: Implement WorkoutPlanGeneratorLLMAgent class:
  - Constructor __init__(client=None):
    * Require client parameter, raise ValueError if None
    * Store self.client = client
  - Method generate_workout_plan(profile):
    * Extract fields: fitness_level_class, injury_risk_class, fitness_goal, available_hours_per_week, health_conditions
    * Extract schedule preferences: preferred_days[], preferred_times[]
    * Create LLM prompt requesting JSON with:
      - Weekly schedule: {day: [{exercise_name, sets, reps, rest_period}]}
      - Workout intensity level (Light|Moderate|Vigorous)
      - Duration per session (minutes)
      - Frequency per week (integer)
      - Progression timeline (4-6 weeks or 8-12 weeks)
      - Safety notes[] accounting for injury_risk and health_conditions
      - Equipment needed[] (or empty for bodyweight only)
    * Call self.client.generate_structured_json() with required fields validation
    * Return: {weekly_schedule, workout_intensity_level, workout_duration_per_session, workout_frequency_per_week,
    workout_progression_timeline, workout_safety_notes[], workout_equipment_needed[]}
"""
from typing import Dict, Any

from utils.gemini_client import invoke_and_parse_json

class WorkoutPlanGeneratorLLMAgent:
    """Generate personalized workout plans using LLM."""

    def __init__(self, client=None):
        """Initialize with LLM client."""
        if client is None:
            raise ValueError("LLM client must be provided.")
        self.client = client

    def generate_workout_plan(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate customized workout plan using structured client interfaces."""
        #1. Extract core fields with safe defaults
        injury_risk = profile.get("injury_risk_class", "Moderate Risk")
        fitness_goal = profile.get("fitness_goal", "General Fitness")
        fitness_level = profile.get("fitness_level_class", "Beginner")
        available_hours_per_week = profile.get("available_hours_per_week", "3")
        health_conditions = profile.get("health_conditions", "None")

        #2.# Extract schedule preferences
        normalized_schedule = profile.get("normalized_schedule", {})
        preferred_days = normalized_schedule.get("preferred_days", ["Any day"])
        preferred_times = normalized_schedule.get("preferred_times", ["Any time"])

        #3.Create the LLM prompt
        prompt = f"""
        You are an expert personal trainer and physical therapist. Create a customized workout plan based on the user's profile.

        USER PROFILE:
        - Injury Risk:{injury_risk}
        - Fitness Level:{fitness_level}
        - Fitness Goal:{fitness_goal}
        - Health Conditions:{health_conditions}
        - Available Hours/Week: {available_hours_per_week}
        - Preferred Training Days:{','.join(preferred_days)}
        - Preferred Training Times:{','.join(preferred_times)}

        INSTRUCTIONS:
        1.Weekly schedule: 
        2.Workout intensity level (Light|Moderate|Vigorous)
        3.Duration per session (minutes)
        4.Frequency per week (integer)
        5.Progression timeline (4-6 weeks or 8-12 weeks).
        6.Safety notes[] accounting for injury_risk and health_conditions.
        7.Equipment needed[] (or empty for bodyweight only)

        OUTPUT FORMAT:
        Return ONLY a valid JSON object with the exact structure below. Do not include markdown code blocks, explanations, or  backticks.
        
        {{
          "weekly_schedule":{{
            "Monday":[
              {{"exercise_name":"<String>","sets":<Interger>."reps":"<String>","rest_period":"<String>"}}
            ],
            "Wednesday":[
              
               {{"exercise_name":"<String>","sets":<Interger>."reps":"<String>","rest_period":"<String>"}} 
              
            ]
          }},
          "workout_intensity_level":"<String:Light,Moderate, or Vigorous>",
          "workout_duration_per_session":<Integer:minutes>,
          "workout_frequency_per_week":<Integer>,
          "workout_progression_timeline":"<String>",
          "workout_safety_notes":["<String>"],
          "workout_equipment_needed":["<String>"]
        }}
        """

        # Fallback plan returned if the LLM call or JSON parsing fails.
        fallback_plan = {
            "weekly_schedule": {
                "Day 1": [
                    {"exercise_name": "Bodyweight Squats", "sets": 3, "reps": "10-15", "rest_period": "60s"},
                    {"exercise_name": "Push-ups (or knee push-ups)", "sets": 3, "reps": "8-12", "rest_period": "60s"},
                    {"exercise_name": "Plank", "sets": 3, "reps": "30s", "rest_period": "60s"}
                ]
            },
            "workout_intensity_level": "Light",
            "workout_duration_per_session": 30,
            "workout_frequency_per_week": 3,
            "workout_progression_timeline": "4-6 weeks",
            "workout_safety_notes": ["Stop immediately if you feel sharp pain."],
            "workout_equipment_needed": ["Yoga Mat"]
        }

        # Invoke the LLM, strip markdown fences, and parse JSON.
        return invoke_and_parse_json(self.client, prompt, default=fallback_plan)
