"""Nutrition Plan Generator LLM Agent — generates a personalised daily nutrition plan via LLM."""
"""
TODO: Implement NutritionPlanGeneratorLLMAgent class:
  - Constructor __init__(client=None):
    * Require client parameter, raise ValueError if None
    * Store self.client = client

  - Method generate_nutrition_plan(profile):
    * Extract fields: age, weight_kg, height_cm, gender, fitness_goal, fitness_level_class, bmi, workout_frequency_per_week
    * Create LLM prompt requesting JSON with:
      - Use Harris-Benedict equation to calculate daily calorie target
      - Apply activity multiplier based on workout_frequency
      - Provide macro targets: protein_g, carbs_g, fat_g
      - Include 3-5 sample daily meal suggestions with Indian cuisine options
      - Each meal: meal_name, foods[], protein_g, carbs_g, fat_g, calories
      - Pre- and post-workout nutrition timing guidance
      - Hydration recommendations (daily water intake)
    * Call self.client.generate_structured_json() with required fields validation
    * Return: {daily_calorie_target, macro_targets{}, meal_suggestions[], hydration_recommendation, nutrition_timing_guidance}
"""
from typing import Dict, Any

from utils.gemini_client import invoke_and_parse_json

class NutritionPlanGeneratorLLMAgent:
    """Generate personalized nutrition plans using LLM."""

    def __init__(self, client):
        """Initialize with LLM client."""
        if client is None:
            raise ValueError("LLM client must be provided.")
        self.client = client

    def generate_nutrition_plan(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate customized nutrition plan matching physiological constraints."""
        #1.Extract fields with safe defaults
        demographic_metrics = {
            "age_val": profile.get("age", 25),
            "weight_val": profile.get("weight_kg", 70.0),
            "height_val": profile.get("height_cm", 170.0),
            "gender_val": profile.get("gender", "Male"),
            "bmi_val": profile.get("bmi", 22.0)
        }
        
        goal_classification = profile.get("fitness_goal", "General Fitness")
        experience_class = profile.get("fitness_level_class", "Beginner")

        weekly_frequency = profile.get("available_hours_per_week", 3)
        if isinstance(weekly_frequency, str):
            numeric_characters = [char for char in weekly_frequency if char.isdigit() or char == '.']
            weekly_frequency = float(''.join(numeric_characters) if numeric_characters else 3.0)

        #2. Create the LLM prompt
        prompt = f"""
        You are an expert sports nutritionist. Create a customized daily nutrition plan based on the user's profile.

        USER PROFILE:
        - Age: {demographic_metrics['age_val']}
        - Gender: {demographic_metrics['gender_val']}
        - Weight: {demographic_metrics['weight_val']} kg
        - Height: {demographic_metrics['height_val']} cm
        - BMI: {demographic_metrics['bmi_val']}
        - Fitness Level: {experience_class}
        - Fitness Goal: {goal_classification}
        - Workout Frequency: {weekly_frequency} times per week

        INSTRUCTIONS:
        1. Use Harris-Benedict equation to calculate daily calorie target.
        2. Apply activity multiplier based on workout_frequency.
        3. Provide macro targets: protein_g, carbs_g, fat_g.
        4. Include 3-5 sample daily meal suggestions with Indian cuisine options.
        5. Each meal: meal_name, foods[], protein_g, carbs_g, fat_g, calories.
        6. Pre- and post-workout nutrition timing guidance.
        7. Hydration recommendations (daily water intake).

        OUTPUT FORMAT:
        Return ONLY a valid JSON object with the exact structure below. Do not include markdown code blocks, explanations, or backticks.
        {{
          "daily_calorie_target":<Integer>
          "macro_targets":{{
            "protein_g":<Integer>
            "carbs_g":<Integer>
            "fat_g":<Integer>
          }},
          "meal_suggestions":[
            {{
              "meal_name":"<String, e.g., Breakfast>",
              "foods":["<List of Strings, e.g., 2 Moong Dal Chilla, 1 cup Curd>"],
              "protein_g":<Integer>,
              "carbs_g":<Integer>,
              "fat_g":<Integer>,
              "calories":<Integer>
            }}
          ],
          "hydration_recommendation":"<String>"
          "nutrition_timing_guidance":"<String>"
        }} 
        """

        # Fallback plan returned if the LLM call or JSON parsing fails.
        fallback_plan = {
            "daily_calorie_target": 2000,
            "macro_targets": {
                "protein_g": 100,
                "carbs_g": 200,
                "fat_g": 60
            },
            "meal_suggestions": [
                {
                    "meal_name": "Standard Indian Breakfast",
                    "foods": ["Poha with peanuts", "1 boiled egg"],
                    "protein_g": 15,
                    "carbs_g": 45,
                    "fat_g": 10,
                    "calories": 330
                }
            ],
            "hydration_recommendation": "Drink at least 3 liters of water per day.",
            "nutrition_timing_guidance": "Eat a light carb-heavy snack 1 hour before workout, and a protein-rich meal within 2 hours after."
        }

        # Invoke the LLM, strip markdown fences, and parse JSON.
        return invoke_and_parse_json(self.client, prompt, default=fallback_plan)
