"""
LangGraph Workflow Orchestration Module

TODO: Import required modules:
  - StateGraph, END from langgraph.graph: Graph construction
  - FitnessAssessmentState, get_initial_state from state: State management
  - All 7 node functions from nodes module

TODO: Implement build_fitness_assessment_graph(client=None):
  - Create StateGraph with FitnessAssessmentState
  - Add 7 nodes (form_parser, input_normalizer, fitness_scorer, injury_assessor,
                 workout_planner, nutrition_advisor, recovery_lifestyle_optimizer)
  - Wrap each node call in lambda to pass client parameter
  - Set entry_point to "form_parser"
  - Add edges in sequence:
    * form_parser → input_normalizer
    * input_normalizer → fitness_scorer (parallel)
    * input_normalizer → injury_assessor (parallel)
    * fitness_scorer → workout_planner (merge)
    * injury_assessor → workout_planner (merge)
    * workout_planner → nutrition_advisor
    * nutrition_advisor → recovery_lifestyle_optimizer
    * recovery_lifestyle_optimizer → END
  - Compile and return graph

TODO: Implement get_workflow_structure():
  - Return dict with workflow metadata:
    * name: 'Fitness Assessment Workflow'
    * description: Workflow purpose
    * nodes[]: List of node dicts with name, type (validation/llm/ml), description
    * edges[]: List of (from, to) tuples for workflow connections
    * entry_point, exit_point: Start and end nodes
"""

from langgraph.graph import StateGraph, END
from typing import Dict, Any
from state import FitnessAssessmentState, get_initial_state
# Import all 7 node functions that form the pipeline stages
from nodes import (
  form_parser_node,
  input_normalizer_node,
  fitness_scorer_node,
  injury_assessor_node,
  workout_planner_node,
  nutrition_advisor_node,
  recovery_lifestyle_optimizer_node,
)

def build_fitness_assessment_graph(client=None):
    """
    Build 7-node fitness assessment workflow graph.

    Returns:
        Compiled StateGraph for workflow execution
    """
    
    # Create StateGraph with FitnessAssessmentState
    workflow = StateGraph(FitnessAssessmentState)

    # Add 7 nodes wrapping each function in a lambda to pass the client parameter
    workflow.add_node("form_parser", lambda state: form_parser_node(state, client=client))
    workflow.add_node("input_normalizer", lambda state: input_normalizer_node(state, client=client))
    workflow.add_node("fitness_scorer", lambda state: fitness_scorer_node(state, client=client))
    workflow.add_node("injury_assessor", lambda state: injury_assessor_node(state, client=client))
    workflow.add_node("workout_planner", lambda state: workout_planner_node(state, client=client))
    workflow.add_node("nutrition_advisor", lambda state: nutrition_advisor_node(state, client=client))
    workflow.add_node("recovery_lifestyle_optimizer", lambda state: recovery_lifestyle_optimizer_node(state, client=client))

    # Set entry point
    workflow.set_entry_point("form_parser")

    # Add standard sequential edge
    workflow.add_edge("form_parser", "input_normalizer")

    # Parallel fan out from input_normalizer to fitness_scorer and injury_assessor
    workflow.add_edge("input_normalizer", "fitness_scorer")
    workflow.add_edge("input_normalizer", "injury_assessor")

    # Merge step: both parallel branches coverage on the workout planner
    workflow.add_edge("fitness_scorer", "workout_planner")
    workflow.add_edge("injury_assessor", "workout_planner")

    # Downstream sequence handling nutrition and lifestyle
    workflow.add_edge("workout_planner", "nutrition_advisor")
    workflow.add_edge("nutrition_advisor", "recovery_lifestyle_optimizer")
    workflow.add_edge("recovery_lifestyle_optimizer", END)

    # Compile and return graph
    return workflow.compile()

"""
TODO: Implement get_workflow_structure()
"""
def get_workflow_structure() -> Dict[str, Any]:
    """Return dict with workflow metadata: name, nodes[], edges[], entry/exit points."""
    return {
      "name": "Fitness Assessment Workflow",
      "description": "Orchestrates a comprehensive fitness assessment path extracting form fields, evaluating athletic/injury baselines and synthesizing workout, meal and recovery routines.",
      "entry_point": "form_parser",
      "exit_point": "END",
      # Each node entry describes its name, processing type, and purpose
      "nodes": [
        {
          "name": "form_parser",
          "type": "validation",
          "description": "Parses and validates incoming questionnaire structure",
        },
        {
          "name": "input_normalizer",
          "type": "validation",
          "description": "Standardizes metric values and cleans user inputs",
        },
        {
          "name": "fitness_scorer",
          "type": "ml",
          "description": "Runs classification algorithms to determine current fitness levels",
        },
        {
        
          "name": "injury_assessor",
          "type": "ml",
          "description": "Estimates contextual musculoskeletal injury hazard values",
        },
        {
        
          "name": "workout_planner",
          "type": "llm",
          "description": "Generates tailored training splits and target progressions",
        },
        {
        
          "name": "nutrition_advisor",
          "type": "llm",
          "description": "Compiles macro-nutrient breakdowns and hydration benchmarks",
        },
        {
        
          "name": "recovery_lifestyle_optimizer",
          "type": "llm",
          "description": "Constructs sleep schedule rules and active recovery routines",
        },
      ],
      # Edges define the execution order; parallel branches merge at workout_planner
      "edges": [
        ("form_parser", "input_normalizer"),
        ("input_normalizer", "fitness_scorer"),
        ("input_normalizer", "injury_assessor"),
        ("fitness_scorer", "workout_planner"),
        ("injury_assessor", "workout_planner"),
        ("workout_planner", "nutrition_advisor"),
        ("nutrition_advisor", "recovery_lifestyle_optimizer"),
        ("recovery_lifestyle_optimizer", "END"),
      ],
    }
