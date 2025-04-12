import sys
from pathlib import Path
from typing import Dict, List
from pydantic import BaseModel
from strictyaml import YAML, load
import vehicleinsurance_model

# Dynamically resolve file paths for module imports
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]

# Add parent directory to Python's module search path
sys.path.append(str(root))

# Define project directories for configuration, datasets, and trained models
PACKAGE_ROOT = Path(vehicleinsurance_model.__file__).resolve().parent  # Root of package
ROOT = PACKAGE_ROOT.parent  # Parent directory of the package
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"  # Path to config file
DATASET_DIR = PACKAGE_ROOT / "datasets"  # Directory containing datasets
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"  # Directory for trained models


class AppConfig(BaseModel):
    """
    Application-level configuration.
    Defines core settings related to pipeline and training data.
    """
    package_name: str
    training_data_file: str
    pipeline_name: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    Configuration relevant to model training and feature engineering.
    Includes feature names, target variable, hyperparameters, and preprocessing mappings.
    """
    target: str  # Target variable for model training
    features: List[str]  # List of feature column names
    unused_fields: List[str]  # Fields excluded from model training

    # Define individual variable names used in dataset preprocessing
    id_var: str
    Gender_var: str
    Age_var: str
    Driving_License_var: str
    Region_Code_var: str
    Previously_Insured_var: str
    Vehicle_Age_var: str
    Vehicle_Damage_var: str
    Annual_Premium_var: str
    Policy_Sales_Channel_var: str
    Vintage_var: str

    Gender_mappings: Dict[str, int]  # Mapping dictionary for Gender column

    # Model training hyperparameters
    TEST_SIZE: float  # Test dataset size percentage
    RANDOM_STATE: int  # Random seed for reproducibility
    N_ESTIMATORS: int  # Number of estimators for the model
    MAX_DEPTH: int  # Maximum depth of trees
    MIN_SAMPLES_SPLIT: int  # Minimum number of samples required to split a node
    MIN_SAMPLES_LEAF: int  # Minimum number of samples required in a leaf node
    CRITERION: str  # Splitting criterion (e.g., 'gini' or 'entropy')


class Config(BaseModel):
    """Master configuration object that holds app-level and model-related configurations."""
    app_config_: AppConfig
    model_config_: ModelConfig


def find_config_file() -> Path:
    """
    Locate the configuration file.
    Raises an exception if the file is not found.
    """
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH

    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """
    Parse YAML containing the package configuration.
    """
    if not cfg_path:
        cfg_path = find_config_file()  # Locate config file dynamically

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())  # Load and parse YAML config
            return parsed_config

    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """
    Run validation on configuration values and ensure correct structure.
    """
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()  # Fetch and parse YAML config

    # Create validated configuration objects
    _config = Config(
        app_config_=AppConfig(**parsed_config.data),
        model_config_=ModelConfig(**parsed_config.data),
    )

    return _config


# Initialize and validate the configuration
config = create_and_validate_config()