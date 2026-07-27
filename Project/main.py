"""
Streamlit Web Application Module

TODO: Import required modules:
  - streamlit as st: Web framework
  - json: For exporting assessment results
  - datetime: For timestamps
  - assess_fitness from graph: Main orchestration function
  - build_gemini_client from utils: API client setup

TODO: Implement initialize_session_state():
  - Initialize st.session_state fields: assessment_result, client, eval_results, run_evaluation
  - Called once at app startup to persist state across reruns

TODO: Implement setup_api_client():
  - Build Gemini client if not already in session_state
  - Handle API key missing errors gracefully
  - Return bool indicating success/failure

TODO: Implement display_overview_tab(assessment):
  - Show 3-column layout with fitness level, injury risk, BMI metrics
  - Display confidence scores for ML predictions
  - Show personal info: age, gender, height, weight
  - Include assessment completion status

TODO: Implement display_workout_tab(assessment):
  - Show workout frequency, intensity, duration per session
  - Display progression timeline and weekly schedule (expandable)
  - List equipment needed and safety guidelines
  - Include exercise breakdown by day

TODO: Implement display_nutrition_tab(assessment):
  - Show daily calorie target and macro targets (protein/carbs/fat)
  - Display meal suggestions with nutritional breakdown (expandable)
  - Include hydration recommendations
  - Show pre/post-workout nutrition timing guidance

TODO: Implement display_recovery_tab(assessment):
  - Show sleep recommendations and hours per night
  - Display rest day activities and mobility routines
  - Include stress management techniques
  - Show recovery methods and deload strategy
  - Display schedule integration tips
  - Show time management and habit formation strategies
  - Include adherence/motivation tips

TODO: Implement export_assessment(assessment):
  - Convert assessment dict to formatted JSON
  - Include metadata: plan_id, timestamp, user_name
  - Return JSON string for download

TODO: Implement display_model_evaluation_section(eval_results):
  - Show fitness model evaluation: accuracy, precision, recall, F1 per class
  - Show injury risk model evaluation: same metrics
  - Include expandable per-class metrics
  - Display confusion matrices if available

TODO: Implement main() entry point:
  - Configure Streamlit page: wide layout, expanded sidebar
  - Call initialize_session_state()
  - Display app header and description
  - Build form in left sidebar:
    * Name input (optional, text)
    * Age slider (18-100, default 30)
    * Height number input (100-250 cm, default 170, 1 decimal)
    * Weight number input (30-300 kg, default 70, 1 decimal)
    * Gender dropdown (Male/Female/Other)
    * Fitness goal dropdown (Weight Loss/Muscle Building/Endurance/General Fitness)
    * Fitness experience text area (free text)
    * Health conditions text area (free text)
    * Available hours text area (free text schedule)
  - Add "Generate Assessment" button:
    * Setup API client
    * Show loading spinner
    * Call assess_fitness() with form data
    * Store result in session_state
    * Display success/error message
  - If assessment_result exists:
    * Display "New Assessment" button to reset
    * Show result header with user name
    * Add "Download JSON" button to export
    * Create 4 tabs: Overview, Workout, Nutrition, Recovery & Lifestyle
    * Call appropriate display functions for each tab
  - Optional model evaluation section:
    * "Run Model Evaluation" button in Tools subsection
    * Display evaluation metrics if available
    * Show training vs evaluation accuracy comparison
"""

import streamlit as st
import os
import json
import urllib.parse
import pandas as pd
from datetime import datetime
from graph import assess_fitness
from utils.gemini_client import build_gemini_client
import sys
from pathlib import Path

def initialize_session_state():
    """Initialize Streamlit session state variables."""    
    if "assessment_result" not in st.session_state:
      st.session_state.assessment_result = None
    if "client" not in st.session_state:
      st.session_state.client = None
    if "eval_results" not in st.session_state:
      st.session_state.eval_results = None
    if "run_evaluation" not in st.session_state:
      st.session_state.run_evaluation = False
    
    # Resolve project root path
    root_path = Path(__file__).parent.resolve()

    # Track essential pipeline artifacts
    required_files = [
      # Processed clean dataset
      root_path / "data" / "processed" / "fitness_level_training_cleaned.csv",
      root_path / "data" / "processed" / "injury_risk_training_cleaned.csv",
      # Saved machine learning model artifacts
      root_path / "ml" / "models" / "fitness_level_model.pkl",
      root_path / "ml" / "models" / "fitness_level_scaler.pkl",
      root_path / "ml" / "models" / "fitness_level_encoder.pkl",
      root_path / "ml" / "models" / "injury_risk_model.pkl",
      root_path / "ml" / "models" / "injury_risk_scaler.pkl",
      root_path / "ml" / "models" / "injury_risk_encoder.pkl",
    ]

    # Check for structural completeness
    artifacts_exist = all(file_path.exists() for file_path in required_files)

    if not artifacts_exist:
      # Display a subtle layout notification message to the user inside Streamlit dashboard
      with st.spinner("Initilazing first-time system stratup: Running ML Training Pipeline..."):
        # Ensure project root is in sys.Path
        if str(root_path) not in sys.path:
          sys.path.insert(0, str(root_path))
        
        #Dynamically import and run training setup
        from ml.train_pipeline import run_training_pipeline

        pipeline_res = run_training_pipeline(project_root=str(root_path))

        # Error handeling
        if pipeline_res.get("pipeline_status", "").startswith("failed"):
          st.error(
            f"Critical initilization failure: Pipeline build fell apart with error:"
            f"{pipeline_res.get('pipeline_status')}"
          )

def setup_api_client():
    """Setup and verify Gemini API client."""
    # Reuse existing client if already initialised in this session
    if "gemini_client" in st.session_state and st.session_state.gemini_client is not None:
      return True
    try:
      client = build_gemini_client()
      st.session_state.gemini_client = client
      return True
    except (ValueError, ImportError) as e:
      st.sidebar.error(f"Failed to initalize Gemini API Client: {str(e)}")
      st.session_state.gemini_client = None
      return False

def display_overview_tab(assessment):
    """Display assessment overview with fitness level, injury risk, and metrics."""
    st.subheader("Executive Fitness Assessment Overview", anchor=False)
    st.write("---")
    col1, col2, col3 = st.columns(3)
    with col1:
      st.metric(
        label=" Fitness Level",
        value=assessment.get("fitness_level_class", "Pending")
      )
      conf = assessment.get("fitness_confidence")
      st.caption(f"Prediction Confidence: {f'{conf:.1f}%' if conf else 'N/A'}")

    with col2:
      st.metric(
        label="Injury Risk",
        value=assessment.get("injury_risk_class", "Pending")
      )
      inj_conf = assessment.get("injury_confidence")
      st.caption(f"Prediction Confidence: {f'{inj_conf:.1f}%' if inj_conf else 'N/A'}")

    with col3:
      bmi_val = assessment.get("bmi")
      st.metric(
        label="BMI",
        value=f"{bmi_val:.1f}" if bmi_val else "N/A"
      )
      st.caption(f"Category: {assessment.get('bmi_category', 'N/A')}")
    st.write("---")

    st.markdown("### User Demographics & Background Profile")
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
      st.markdown(f"**Name:** {assessment.get('user_name', 'Guest')}")
      st.markdown(f"**Age:** {assessment.get('age')} years ({assessment.get('age_category', 'N/A')})")
      st.markdown(f"**Gender Identity:** {assessment.get('gender')}")
    with sub_col2:
      st.markdown(f"**Height:** {assessment.get('height_cm')} cm")
      st.markdown(f"**Weight:** {assessment.get('weight_kg')} kg")
      st.markdown(f"**Target Objective:** {assessment.get('fitness_goal')}")

    st.write("---")

    risk_factors = assessment.get("injury_risk_factors", [])
    if risk_factors:
      st.warning("### Recognized Injury Risk Factors")
      for factor in risk_factors:
        st.markdown(f"* {factor}")
    else:
      st.success("No structural injury risks identified from personal health entries")

#HELPR METHODS
def _parse_workout_plan(raw_plan):
  """Parse a raw workout plan value into a dict, handling both dict and JSON/Python string formats."""
  if isinstance(raw_plan, dict):
    return raw_plan
  if not isinstance(raw_plan, str) and raw_plan.strip().startswith("{"):
    return None
  import ast, json
  cleaned = raw_plan.strip()
  try:
    return ast.literal_eval(cleaned)
  except Exception:
    try:
      return json.loads(cleaned)
    except Exception:
      return None    

def _sync_assessment_keys(assessment, parsed_plan):
  """Copy missing workout keys from parsed_plan into the assessment dict."""
  if not (parsed_plan and isinstance(parsed_plan, dict)):
    return
  keys = [
      "workout_intensity_level", "workout_duration_per_session",
      "workout_frequency_per_week", "workout_progression_timeline",
      "workout_safety_notes", "workout_equipment_needed"
    ]
  for key in keys:
      if not assessment.get(key) and key in parsed_plan:
        assessment[key] = parsed_plan[key]

def _youtube_search_url(exercise_name: str) -> str:
  """Build a YouTube search URL for how-to/form videos of an exercise.

  Uses a search query (not a specific video ID) so the link always resolves —
  no API key and no risk of dead/hallucinated links.
  """
  query = urllib.parse.quote_plus(f"{exercise_name} proper form technique")
  return f"https://www.youtube.com/results?search_query={query}"

def _render_exercise(ex):
  """Render a single exercise entry with sets, reps, rest period, and a form-video link."""
  if isinstance(ex, dict):
    name = ex.get('exercise_name', 'Exercise')
    video_url = _youtube_search_url(name)
    st.markdown(
      f"**{name}** -"
      f"{ex.get('sets','?')} sets x  {ex.get('reps','?')} reps "
      f"| Rest: {ex.get('rest_period', 'N/A')} "
      f"| [📺 Watch form]({video_url})"
    )
  else:
    st.write(ex)
def _render_single_day_schedule(excercises):
  """Render all exercises for a single training day."""
  if isinstance(excercises, list):
    for ex in excercises:
      _render_exercise(ex)
  else:
    st.write(excercises)

def _render_weekly_schedule(assessment,parsed_plan):
  """Render the full weekly exercise schedule, falling back to parsed_plan if needed."""
  st.markdown("###  Weekly Schedule")
  schedule = assessment.get("weekly_schedule") or (
    parsed_plan.get("weekly_schedule", {}) if parsed_plan else {}
  )

  if not isinstance(schedule, dict) or not schedule:
    st.info("No structured weekly schedule available.")
    return

  for day, excercises in schedule.items():
    with st.expander(f"{day.upper()}", expanded=True):
      _render_single_day_schedule(excercises)

def _render_safety_equipment(assessment):
  """Render equipment requirements, safety notes, and progression timeline."""
  st.markdown("### Safety Constraints & Equipment Guidelines")
  w_col1, w_col2 = st.columns(2)
  with w_col1:
    st.markdown("**Equipment Requirements:**")
    equip = assessment.get("workout_equipment_needed", []) or ["Bodyweight/Minimalist Equipment"]
    for item in equip or ["Bodyweight/Minimalist Equipment"]:
        st.markdown(f"- {item}")

  with w_col2:
    st.markdown("**Safety constraints:**")
    safety = assessment.get("workout_safety_notes", []) or ["- Maintain safe form."]
    safety_list = safety if isinstance(safety, list) else [safety]
    for note in safety_list:
        st.markdown(f"- {note}" if isinstance(safety,list) else note)

  st.markdown(
    f"**Timeline Progression Directives:**" 
    f"{assessment.get('workout_progression_timeline', 'Gradual incremental overload.')}"
  )

def display_workout_tab(assessment):
    """Display workout plan details including schedule and intensity."""
          
    #MAIN RENDERING LOGIC
    st.subheader("Tailored Exercise & Workout Strategy", anchor=False)
    st.write("---")

    raw_plan = assessment.get("workout_plan", "")
    parsed_plan = _parse_workout_plan(raw_plan)
    
    if parsed_plan and isinstance(raw_plan,str):
      raw_plan = ""

    _sync_assessment_keys(assessment,parsed_plan)

    col1, col2, col3 =st.columns(3)
    with col1:
      st.metric("Intensity Level", assessment.get("workout_intensity_level", "N/A"))
    with col2:
      st.metric("Frequency", f"{assessment.get('workout_frequency_per_week', 0)} Days / Wk")  
    with col3:
      st.metric("Session Duration", f"{assessment.get('workout_duration_per_session', 0)} Mins")

    st.write("---")
    _render_weekly_schedule(assessment, parsed_plan)

    st.write("---")
    _render_safety_equipment(assessment)

# Helper Methods
def _render_metrics(assessment):
  """Render teh top level calorie and macro neutrient metrices"""
  col1, col2 = st.columns(2)
  with col1:
    calorie_target = assessment.get('daily_calorie_target', 0)
    st.metric("Daily calorie Target", f"{calorie_target} kcal")
  with col2:
    macros =assessment.get("macro_targets")
    if isinstance(macros, dict):
      macro_str = f"P: {macros.get('protein_g',0)}g | C: {macros.get('carbs_g', 0)}g | F: {macros.get('fat_g', 0)}g"
    else:
      macro_str = "N/A"
    st.metric("Target Macro-nutrients Distribution", macro_str)

def _render_food_suggestions(options):
  """Renders the list of food suggestions"""
  if isinstance(options, list) and options:
    food_lines = ["- **Suggestions:**"] + [f"  - {food}" for food in options]
    st.markdown("\n".join(food_lines))
  else:
    st.markdown("- **Suggestions:**\n  - N/A")

def _render_nutrition_rationale(meal):
  """Renders the nutritional breakdown for a meal"""
  if isinstance(meal, dict):
    nutrition_lines = [
      "- **Nutritional Rationale:**",
      f"  - **Protein:** {meal.get('protein_g', 0)}g",
      f"  - **Carbs:** {meal.get('carbs_g', 0)}g",
      f"  - **Fat:** {meal.get('fat_g', 0)}g",
      f"  - **Calories:** {meal.get('calories', 0)} kcal"
    ]
    st.markdown("\n".join(nutrition_lines))
  else:
    st.markdown("- **Nutritional Rationale:**\n  - N/A")

def _render_meals(assessment):
  """Renders the expandable meals list section"""
  st.markdown("**Structured Meal Planning**")
  meals = assessment.get("meal_suggestions", [])

  if not meals:
    st.info("Review textual meal framework block below for the daily breakdown details")
    return
  
  for meal in meals:
    meal_name = meal.get('meal_name','Meal Assignment') if isinstance(meal, dict) else 'Meal Assignment'
    with st.expander(meal_name, expanded=True):
      options = meal.get('foods', []) if isinstance(meal, dict) else []
      _render_food_suggestions(options)
      _render_nutrition_rationale(meal)

def _render_hydration_and_timing(assessment):
  """Renders the ebottom hydration and timing guideline"""
  st.markdown("### Hydration & Window Timing Guidelines")
  n_col1, n_col2 = st.columns(2)
  with n_col1:
    hydration = assessment.get('hydration_recommendation', '2-3 Liters')
    st.markdown(f"**Optimal Hydration Volumn:** {hydration}")
  with n_col2:
    timing = assessment.get('nutrition_timing_guidance', 'Standard Distribution.')
    st.markdown(f"**Nutrient Timing Guidance:** {timing}")

# Main Rendering Logic
def display_nutrition_tab(assessment):
    """Display nutrition plan with calorie targets and meal suggestions."""
    st.subheader(" Culturally-Appropriate (Indian Diet) Nutrition Advisory", anchor=False)
    st.write("---")

    _render_metrics(assessment)
    st.write("---")

    _render_meals(assessment)
    st.write("---")

    _render_hydration_and_timing(assessment)

# Helper Methods
def _as_list(value) -> list:
  """Normalize an LLM field to a list.

  The model occasionally returns a scalar string where a list is expected;
  returning it as-is would let ', '.join() split it character-by-character.
  """
  if isinstance(value, list):
    return value
  if value in (None, ""):
    return []
  return [value]

def _render_markdown_list(title: str, items:list) -> None:
  """Helper to render a clean markdown bullet list or an N/A fallback"""
  if isinstance(items, list) and items:
    lines = [f"- **{title}:**"] + [f"  - {item}" for item in items]
    st.markdown("\n".join(lines))
  else:  
    st.markdown(f"- **{title}:**\n  - N/A")

def _render_sleep_section(assessment: dict) -> None:
  """Render sleep hour targets and quality tips inside an expander."""
  with st.expander("Sleep Architecture Directive", expanded=True):
    sleep_rec = assessment.get("sleep_recommendations", {})
    hours = sleep_rec.get('hours_per_night', 8) if isinstance(sleep_rec, dict) else 8
    st.markdown(f"- **Sleep Target:** {hours} hrs")

    tips = sleep_rec.get('sleep_quality_tips', []) if isinstance(sleep_rec, dict) else []
    _render_markdown_list("Sleep Quality Tips", tips)

def _render_recovery_section(assessment: dict) -> None:
  """Render active recovery options and mobility routines inside an expander."""
  with st.expander("Active Recovery & Mobility Operations", expanded=True):
    _render_markdown_list("Mobility Target Protocols", assessment.get("mobility_work", []))
    _render_markdown_list("Rest Day Active Operations", assessment.get("rest_day_activities", []))

def _render_stress_section(assessment: dict) -> None:
  """Render stress management techniques and deload strategy inside an expander."""
  with st.expander("Stress Mitigation & System Deload Strategies", expanded=True):
    _render_markdown_list("Stress Alleviation Modalities", assessment.get("stress_management_techniques", []))
    deload = assessment.get('deload_strategy','Every 6-8 weeks.')
    st.markdown(f"- **Progression Deload Protocol:**\n  - {deload}")

def _render_schedule_section(assessment: dict) -> None:
  """Render schedule integration details and time management tips inside an expander."""
  with st.expander("Schedule Integration & Time-Management", expanded=True):
    schedule_integration = assessment.get("schedule_integration", {})
    if isinstance(schedule_integration, dict):
      routine_lines = ["- **Routine Integration Alignment:**"]
      # Coerce to a list first: the LLM sometimes returns a plain string, and
      # ', '.join(<str>) would otherwise split it character-by-character.
      best_days = _as_list(schedule_integration.get("best_days", []))
      if best_days:
        routine_lines.append(f"  - **Best Days:** {', '.join(best_days)}")
      best_times = _as_list(schedule_integration.get("best_times", []))
      if best_times:
        routine_lines.append(f"  - **Best Times:** {', '.join(best_times)}")
      weekly_tips = schedule_integration.get("weekly_schedule_tips", [])
      if weekly_tips:
        routine_lines.append(f"  - **Weekly Schedule Tips:** {weekly_tips}")
      st.markdown("\n".join(routine_lines))
    
    _render_markdown_list("Time Optimization Tips", assessment.get("time_management_tips", []))

def _render_adherence_section(assessment: dict) -> None:
  """Render habit formation strategies and adherence tips inside an expander."""
  with st.expander("Behavioural Habit Formation & Long-Term Adherence", expanded=True):
    _render_markdown_list("Atomic Habit Anchoring Routines", assessment.get("habit_formation_strategies", []))
    _render_markdown_list("Consistency Retention Directives", assessment.get("adherence_tips", []))
  
# Main Rendering Logic
def display_recovery_tab(assessment: dict) -> None:
    """Display recovery and lifestyle optimization plan."""
    st.subheader("Recovery, Stress, Adherence Optimization", anchor=False)
    st.write("---")
    _render_sleep_section(assessment)
    _render_recovery_section(assessment)
    _render_stress_section(assessment)
    _render_schedule_section(assessment)
    _render_adherence_section(assessment)

def export_assessment(assessment):
    """Export assessment as JSON string."""
    # Strip any private/internal keys before serialising
    clean_dict = {k: v for k, v in assessment.items() if not k.startswith('_')}
    return json.dumps(clean_dict, indent=4)

def get_metric_value(metric_obj, index, class_name):
  """Helper tool to extract data cross structural data maps safely."""
  # Handle both dict (keyed by class name or index) and list/tuple formats
  if isinstance(metric_obj, dict):
    return metric_obj.get(class_name, metric_obj.get(index, metric_obj.get(str(index))))
  elif isinstance(metric_obj, (list, tuple)):
    return metric_obj[index] if index < len(metric_obj) else None

def render_metrics_table(model_eval):
  """Render a per-class precision, recall, and F1-score table for a model evaluation result."""
  classes = model_eval.get("classes") or []
  precision = model_eval.get("precision_per_class") or []
  recall = model_eval.get("recall_per_class") or []
  f1 = model_eval.get("f1_per_class") or []

  rows = []
  for i, class_name in enumerate(classes):
    rows.append({
      "Fitness Class" : class_name,
      "precision" : get_metric_value(precision, i , class_name),
      "recall" : get_metric_value(recall, i, class_name),
      "F1-Score" : get_metric_value(f1, i , class_name)
    })

  if rows:
    df=pd.DataFrame(rows).set_index("Fitness Class")
    st.dataframe(df.style.format("{:.2f}",na_rep="_"), width='stretch')
  else:
    st.warning("No Valid Evaluation class available")

def render_confusion_matrix(model_eval):
  """Render the confusion matrix from a model evaluation result, if available."""
  cm = model_eval.get("confusion_matrix")
  classes = model_eval.get("classes") or []
  if cm is not None:
    st.markdown("** Confusion matrix array representation")
    st.caption("Columns = Predicted Classes | Rows = Actual(True) Classes")
    try:
      df_cm = pd.DataFrame(cm, index = classes, columns = classes)
      st.dataframe(df_cm, width = 'stretch')
    except Exception:
      st.write(cm)

def display_model_evaluation_section(eval_results):
    """Display model evaluation metrics and comparison."""
    h_col1, h_col2 = st.columns([0.8, 0.2])
    with h_col1:
      st.markdown("## Model Evaluation Metrics")
      st.caption(f"Model Evaluated At: {eval_results.get('timestamp', 'N/A')}")
      st.write("---")
    with h_col2:
      if st.button("<- Back to Assessment  Report", width = 'stretch', key='butn_back_to_assessment'):
        st.session_state.run_evaluation = False
        st.rerun()
    

    #1 FITNESS MODEL EVALUATION
    fit_eval = eval_results.get("fitness_evaluation", {})
    if "error" in fit_eval:
      st.error(f"Fitness Model Validation Error: {fit_eval['error']}")
    else:
      st.markdown("Fitness Model Evaluation Results")
      fit_eval_accuracy = fit_eval.get('eval_accuracy', 0.0)
      st.metric("Fitness Level Accurancy", f"{fit_eval_accuracy:.2%}")
      
      with st.expander("  Per-Class Metrics Matrix (Fitness Level)", expanded=True):
        render_metrics_table(fit_eval)
      render_confusion_matrix(fit_eval)
    st.write("---")
    # 2 INJURY RISK MODEL EVALUATION
    inj_eval = eval_results.get("injury_evaluation",{})
    if "error" in inj_eval:
      st.error(f"Injury Model Validation Error: {inj_eval['error']}")
    else:
      st.markdown("Injury Prediction Model Evaluation Results")
      inj_eval_accuracy = inj_eval.get('eval_accuracy', 0.0)
      st.metric("Injury Assesment model Accurancy", f"{inj_eval_accuracy:.2%}")

      with st.expander("Pre-Class Metrics Matrix (Injury Risk)", expanded=True):
        render_metrics_table(inj_eval)
      render_confusion_matrix(inj_eval)

    #3 FITNESS MODEL EVALUATION FOR TRAINING DATA
    fit_train = eval_results.get("fitness_training_evaluation", {})
    if "error" in fit_train:
      st.error(f"Fitness Model Validation Error for training dataset: {fit_train['error']}")
    else:
      fit_train_accuracy = fit_train.get('eval_accuracy', 0.0)

    #4 INJURY RISK MODEL EVALUATION FOR TRAINING DATASET
    inj_train = eval_results.get("injury_training_evaluation",{})
    if "error" in inj_train:
      st.error(f"Injury Model Validation Error for training dataset: {inj_train['error']}")
    else:
      inj_train_accuracy = inj_train.get('eval_accuracy', 0.0)


    #Accuracy Comparision
    fit_delta = fit_eval_accuracy - fit_train_accuracy
    inj_delta = inj_eval_accuracy - inj_train_accuracy

    st.markdown("### Training Vs Evaluation Accuracy Comparision")

    model_col1, model_col2 = st.columns(2)

    with model_col1:
      st.markdown("### Fitness level Model")
      f_col1, f_col2 = st.columns(2)
      f_col1.metric(
        label = "Training Accuracy",
        value = f"{fit_train_accuracy:.2%}"
      )
      f_col2.metric(
        label = "Evaluation Accuracy",
        value = f"{fit_eval_accuracy:.2%}",
        delta = f"{fit_delta:.2%} vs train",
        delta_color = "normal" if fit_delta >= -0.05 else "inverse"
      )

    with model_col2:
      st.markdown("### Injury Prediction Model")
      f_col1, f_col2 = st.columns(2)
      f_col1.metric(
        label = "Training Accuracy",
        value = f"{inj_train_accuracy:.2%}"
      )
      f_col2.metric(
        label = "Evaluation Accuracy",
        value = f"{inj_eval_accuracy:.2%}",
        delta = f"{inj_delta:.2%} vs train",
        delta_color = "normal" if inj_delta >= -0.05 else "inverse"
      )

def render_assessment_tabs(assessment_data):
  """Render tabs for the current assessment."""
  tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Workout Plan",
        "Nutrition Plan",
        "Recovery and Lifestyle"
      ])

  with tab1:
    display_overview_tab(assessment_data)
  with tab2:
    display_workout_tab(assessment_data)
  with tab3:
    display_nutrition_tab(assessment_data)
  with tab4:
    display_recovery_tab(assessment_data)

def main():
    """Main Streamlit application entry point."""
    st.set_page_config(page_title="AetherFit AI Dashboard", page_icon="💪", layout="wide", initial_sidebar_state="expanded")
    initialize_session_state()
    
    st.markdown("""
      <style>
      button[data-testid="sidebar-toggle"] {
        overflow: hidden !important;
        color: transparent !important;
        background: transparent !important;
      }
      button[data-testid="sidebar-toggle"]::after {
        content: "->" !important;
        color: var(--text-color, #31333F) !important;
        font-size: 16px !important;
        line-height: 1 !important;
        display: block !important;
        text-align: center !important;
      }
      div[data-testid="stSidebarCollapseNav"]{
        display: none !important
      }
      </style>
    """, unsafe_allow_html=True)
    # Application Header Title
    st.title("💪🏃🩺 AetherFit: AI-Powered Personalized Fitness Assessment System")
    st.markdown("*Intelligent State Orchestration pipeline for Fitness Assessment.*")
    st.write("---")
    # SIDEBAR FORM DESIGN
    st.sidebar.header("User Profile")

    user_name = st.sidebar.text_input("Full Name", value="Guest")
    age = st.sidebar.slider("Age", min_value=18, max_value=100, value=30)
    height_cm = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=.1, format="%.1f")
    weight_kg = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0, step=.1, format="%.1f")
    gender = st.sidebar.selectbox("Gender", options=["Male", "Female", "Other"])

    fitness_goal = st.sidebar.selectbox(
      "Fitness Goal",
      options=["Weight Loss","Muscle Building","Endurance","General Fitness"]
    )

    fitness_experience = st.sidebar.text_area(
      "Fitness Experience",
      placeholder="Describe your historic training background, activity depth, etc"
    )
    health_conditions = st.sidebar.text_area(
      "Health Conditions",
      placeholder="Describe previous injuries, cardiovascular parameters, metabolic constraints, etc."
    )
    available_hours = st.sidebar.text_area(
      "Available Hours",
      placeholder="e.g., 4 hours total: Mon/Wed/Fri 1 hour evening, Sat morning 1 hour"
    )    
    # Submission Action Controls
    if st.sidebar.button("Generate Assessment", width='stretch'):
      if setup_api_client():
        # Collect sidebar form values into a single payload dict
        form_payload = {
          "user_name": user_name,
          "age": int(age),
          "height_cm": float(height_cm),
          "weight_kg":float(weight_kg),
          "gender": gender,
          "fitness_goal": fitness_goal,
          "fitness_experience": fitness_experience,
          "health_conditions": health_conditions,
          "available_hours_per_week": available_hours
        }

        with st.spinner("Orchestrating multi-agent network execution for fitness assessment..."):
          try:
            # Invoke the full 7-node LangGraph pipeline
            output_state = assess_fitness(**form_payload, client=st.session_state.gemini_client)
            st.session_state.assessment_result = output_state
            st.session_state.run_evaluation = False
            st.sidebar.success("Assessment completed Successfully!!!")
          except Exception as e:
            st.sidebar.error(f"Assessment Failed: {str(e)}")
    # Admin/Tools Area Section
    st.sidebar.write("---")
    st.sidebar.markdown("### Tools")
    
    if st.sidebar.button("Run Model Evaluation", width='stretch'):
      try:
        from ml.evaluation.evaluate_models import evaluate_all_models
        with st.spinner("Processing test splits..."):
          st.session_state.eval_results = evaluate_all_models()
          st.session_state.run_evaluation = True
          st.rerun()
      except Exception as e:
        st.sidebar.error(f"Could not pull evaluation scripts: {str(e)}")
    # MAIN ENTRANCE VIEWPORTS
    if st.session_state.run_evaluation and st.session_state.eval_results:
      display_model_evaluation_section(st.session_state.eval_results)

    elif st.session_state.assessment_result:
      h_col1, h_col2 = st.columns([0.8, 0.2])
      with h_col1:
        st.markdown(f"## Health Report & Prescription Card: {st.session_state.assessment_result.get('user_name')}")
      with h_col2:
        # Allow the user to clear the current result and start a new assessment
        if st.button("New Assessment", width='stretch', key='butn_new_assessment'):
          st.session_state.assessment_result = None
          st.rerun()
      
      json_string = export_assessment(st.session_state.assessment_result)
      st.download_button(
        label="Download JSON",
        data=json_string,
        file_name=f"AetherFit_Blueprint_{st.session_state.assessment_result.get('plan_id', 'export')}.json",
        mime="application/json"
      )
      render_assessment_tabs(st.session_state.assessment_result)
    else:
      st.info("Complete User profile in the left sidebar")

if __name__ == "__main__":
    main()