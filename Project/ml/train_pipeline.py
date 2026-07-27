"""Master orchestrator for ML training pipeline."""

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to sys.path to ensure imports work in all scenarios
# (script execution, module import, and pytest)
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try relative imports first (when imported as module), fall back to absolute
try:
    from .data_cleaning.clean_fitness_data import clean_fitness_data
    from .data_cleaning.clean_injury_data import clean_injury_data
    from .train_model.train_fitness_level import train_fitness_level_model
    from .train_model.train_injury_risk import train_injury_risk_model
    from .evaluation.evaluate_models import evaluate_fitness_level_model, evaluate_injury_risk_model
except ImportError:
    # Fall back to absolute imports when run as a script
    from ml.data_cleaning.clean_fitness_data import clean_fitness_data
    from ml.data_cleaning.clean_injury_data import clean_injury_data
    from ml.train_model.train_fitness_level import train_fitness_level_model
    from ml.train_model.train_injury_risk import train_injury_risk_model
    from ml.evaluation.evaluate_models import evaluate_fitness_level_model, evaluate_injury_risk_model


def run_training_pipeline(project_root: str = None) -> dict:
    """
    TODO: Implement run_training_pipeline function - Master orchestrator for complete ML training:

    Steps:
    1. Resolve project_root (default to current working directory)
    2. Define data paths: training_dataset/{fitness_level,injury_risk}_training.csv, evaluation_dataset/{fitness_level,injury_risk}_evaluation.csv, processed/{*}_cleaned.csv
    3. Define model output paths: ml/models/{*}_model.pkl, {*}_scaler.pkl, {*}_encoder.pkl
    4. Create ml/models directory if not exists
    5. Call clean_fitness_data(fitness_train_raw, fitness_train_cleaned)
    6. Call clean_injury_data(injury_train_raw, injury_train_cleaned)
    7. Call train_fitness_level_model(fitness_train_cleaned, model/scaler/encoder paths)
    8. Call train_injury_risk_model(injury_train_cleaned, model/scaler/encoder paths)
    9. Call evaluate_fitness_level_model(fitness_eval_raw, model paths)
    10. Call evaluate_injury_risk_model(injury_eval_raw, model paths)
    11. Collect results: fitness_training, injury_training, fitness_evaluation, injury_evaluation
    12. Add pipeline_status: 'success' or 'failed' with error message
    13. Return: {fitness_training, injury_training, fitness_evaluation, injury_evaluation, pipeline_status}
    """
    
    # 1. Resolve project_root (default to current working directory)
    if project_root is None:
        root = Path.cwd()
    else:
        root = Path(project_root)
    
    # 2. Define data paths under Project > data > [individual folders]
    data_dir = root / "data"
    
    fitness_train_raw = str(data_dir / "training_dataset" / "fitness_level_training.csv")
    injury_train_raw = str(data_dir / "training_dataset" / "injury_risk_training.csv")

    fitness_eval_raw = str(data_dir / "evaluation_dataset" / "fitness_level_evaluation.csv")
    injury_eval_raw = str(data_dir / "evaluation_dataset" / "injury_risk_evaluation.csv")

    fitness_train_cleaned = str(data_dir / "processed" / "fitness_level_training_cleaned.csv")
    injury_train_cleaned = str(data_dir / "processed" / "injury_risk_training_cleaned.csv")

    # 3. Define model output paths
    models_dir = root / "ml" / "models"

    fitness_model_path = str(models_dir / "fitness_level_model.pkl")
    fitness_scaler_path = str(models_dir / "fitness_level_scaler.pkl")
    fitness_encoder_path = str(models_dir / "fitness_level_encoder.pkl")

    injury_model_path = str(models_dir / "injury_risk_model.pkl")
    injury_scaler_path = str(models_dir / "injury_risk_scaler.pkl")
    injury_encoder_path = str(models_dir / "injury_risk_encoder.pkl")

    # 11. Initialize results payload shell for collecting results
    pipeline_results = {
        "fitness_training": None,
        "injury_training": None,
        "fitness_evaluation": None,
        "injury_evaluation": None,
        "pipeline_status": "failed",
    }

    try:
        # 4. Create ml/models directory if not exists
        models_dir.mkdir(parents=True, exist_ok=True)
        Path(fitness_train_cleaned).parent.mkdir(parents=True, exist_ok=True)

        # 5. Clean fitness data
        clean_fitness_data(input_path=fitness_train_raw, output_path=fitness_train_cleaned)

        # 6. Clean injury data
        clean_injury_data(input_path=injury_train_raw, output_path=injury_train_cleaned)

        # 7. Train fitness level model
        fitness_training_res = train_fitness_level_model(
            training_data_path=fitness_train_cleaned,
            model_output_path=fitness_model_path,
            scaler_output_path=fitness_scaler_path,
            encoder_output_path=fitness_encoder_path
        )
        pipeline_results["fitness_training"] = fitness_training_res

        # 8. Train injury risk model
        injury_training_res = train_injury_risk_model(
            training_data_path=injury_train_cleaned,
            model_output_path=injury_model_path,
            scaler_output_path=injury_scaler_path,
            encoder_output_path=injury_encoder_path
        )
        pipeline_results["injury_training"] = injury_training_res

        # 9. Evaluate fitness level model
        fitness_eval_res = evaluate_fitness_level_model(
            eval_data_path=fitness_eval_raw,
            model_path=fitness_model_path,
            scaler_path=fitness_scaler_path,
            encoder_path=fitness_encoder_path
        )
        pipeline_results["fitness_evaluation"] = fitness_eval_res

        # 10. Evaluate injury risk model
        injury_eval_res = evaluate_injury_risk_model(
            eval_data_path=injury_eval_raw,
            model_path=injury_model_path,
            scaler_path=injury_scaler_path,
            encoder_path=injury_encoder_path
        )
        pipeline_results["injury_evaluation"] = injury_eval_res

        # 12. Mark pipeline status as success
        pipeline_results["pipeline_status"] = "success"
    
    except Exception as e:
        # Catch error gracefully and inject error message into status
        pipeline_results["pipeline_status"] = f"failed: {str(e)}"
    
    # 13. Return compiled tracking summary
    return pipeline_results

def load_ml_artifacts(
    model_dir: str,
    artifact_filenames: Dict[str, str],
    agent_name: str = "ML",
) -> Dict[str, Optional[Any]]:
    """Load an ML agent's model/scaler/encoder pickles, training them on demand if missing.

    Args:
        model_dir: directory holding the pickle files.
        artifact_filenames: mapping of result key -> pickle filename, e.g.
            {"model": "fitness_level_model.pkl", "scaler": "...", "encoder": "..."}.
        agent_name: human-readable name used in log messages.

    Returns:
        Mapping of the same keys to the loaded objects. On any failure every value
        is None so callers can fall back to a "model not loaded" path.
    """
    artifacts: Dict[str, Optional[Any]] = dict.fromkeys(artifact_filenames)

    try:
        target_filenames = list(artifact_filenames.values())

        # Check whether all required pickle files are present
        files_exist = all(
            os.path.isfile(os.path.join(model_dir, f)) for f in target_filenames
        )

        # Trigger training pipeline if any model file is missing
        if not files_exist:
            print(f"ML model files missing in {model_dir}. Running training pipeline...")
            root_path = Path(__file__).parent.parent.resolve()
            pipeline_res = run_training_pipeline(project_root=str(root_path))
            if pipeline_res.get("pipeline_status", "").startswith("failed"):
                raise RuntimeError(
                    f"Pipeline Build fell apart with error: {pipeline_res.get('pipeline_status')}"
                )

        for key, filename in artifact_filenames.items():
            with open(os.path.join(model_dir, filename), "rb") as source_filestream:
                artifacts[key] = pickle.load(source_filestream)

        print(f"{agent_name} models loaded successfully.")
    except RuntimeError as e:
        print(f"Critical Initilization Failure: {e}")
    except (pickle.PickleError, EOFError, AttributeError) as e:
        print(f"Error:Corrupted or Invalid picke files found in {model_dir}.({e})")

    return artifacts


if __name__ == "__main__":
    # Run from AetherFit root directory
    project_root = str(Path(__file__).parent.parent)
    results = run_training_pipeline(project_root)
    sys.exit(0 if results['pipeline_status'] == 'success' else 1)
