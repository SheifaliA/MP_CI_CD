import sys
from pathlib import Path
import typing as t
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from vehicleinsurance_model import __version__ as _version
from vehicleinsurance_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

# Dynamically resolve file paths for module imports
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]

# Add parent directory to Python's module search path
sys.path.append(str(root))


## Pre-Pipeline Preparation
def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the dataset before running the pipeline.
    This function prints dataset information and performs preprocessing.
    """
    print(data_frame.info())

    # Uncomment the following code to drop unnecessary fields based on config settings
    # for field in config.model_config_.unused_fields:
    #     if field in data_frame.columns:
    #         data_frame.drop(labels=field, axis=1, inplace=True)    

    return data_frame


def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    """
    Loads the raw dataset from the specified file.
    """
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    Loads and prepares the dataset by applying pre-processing steps.
    """
    print(f"🔎 Loading dataset file: {file_name}")
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame=dataframe)
    return transformed


## Model Pipeline Functions
def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """
    Saves the trained pipeline to a file.
    The saved model is versioned to ensure reproducibility.
    Old pipelines are removed to maintain a clean state.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    # Remove outdated models before saving a new one
    remove_old_pipelines(files_to_keep=[save_file_name])

    # Save the trained pipeline using joblib
    joblib.dump(pipeline_to_persist, save_path)
    print("✅ Model/pipeline saved successfully.")


def load_pipeline(*, file_name: str) -> Pipeline:
    """
    Loads a persisted pipeline for inference.
    """
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Removes old model pipelines to maintain a clean environment.
    Ensures a one-to-one mapping between package version and model version.
    """
    do_not_delete = files_to_keep + ["__init__.py"]

    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()