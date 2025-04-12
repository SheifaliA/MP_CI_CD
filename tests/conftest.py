import sys
from pathlib import Path

# Dynamically determine the paths for resolving module imports
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]

# Add the parent directory to Python's module path for importing project modules
sys.path.append(str(root))

import pytest
from sklearn.model_selection import train_test_split
import yaml
# from vehicleinsurance_model.config.core import create_and_validate_config
from vehicleinsurance_model.processing.data_manager import load_dataset
import importlib 
import pytest
import importlib
import yaml
from pathlib import Path
from vehicleinsurance_model.config import core as config_module
import shutil

# Define the path to the configuration file
CONFIG_FILE_PATH = Path("vehicleinsurance_model/config.yml")


@pytest.fixture(scope="session", autouse=True)
def reload_config():
    """Fixture to reload the configuration module before running tests."""

    # Remove cache directories to ensure fresh configurations
    shutil.rmtree("vehicleinsurance_model/__pycache__", ignore_errors=True)
    shutil.rmtree(".pytest_cache", ignore_errors=True)

    # Validate YAML formatting to prevent misconfiguration issues
    try:
        with open(CONFIG_FILE_PATH, "r") as file:
            yaml.safe_load(file)  # Ensure the YAML file is valid
    except yaml.YAMLError as e:
        pytest.exit(f"YAML validation failed! Error: {e}")  # Exit tests if config is invalid

    # Reload the configuration module to apply the latest changes
    importlib.reload(config_module)

    # Load and validate the configuration object
    config = config_module.create_and_validate_config()
    print(f"✔ Config reloaded: {config.app_config_.training_data_file}")

    return config


@pytest.fixture
def sample_input_data(reload_config):
    """Fixture to provide sample input data for tests."""

    # Load dataset using the dynamically retrieved config file path
    data = load_dataset(file_name=reload_config.app_config_.training_data_file)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data[reload_config.model_config_.features],  # Predictor variables
        data[reload_config.model_config_.target],    # Target variable
        test_size=reload_config.model_config_.TEST_SIZE,  # Defined test size
        random_state=reload_config.model_config_.RANDOM_STATE,  # Ensures consistent results across runs
    )

    return X_test, y_test  # Return test dataset for evaluation