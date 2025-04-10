import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
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
CONFIG_FILE_PATH = Path("vehicleinsurance_model/config.yml")

@pytest.fixture(scope="session", autouse=True)
def reload_config():    
    # Remove Python cache to prevent outdated config usage    
    shutil.rmtree("vehicleinsurance_model/__pycache__", ignore_errors=True)
    shutil.rmtree(".pytest_cache", ignore_errors=True)
    # Validate YAML formatting before tests
    try:
        with open(CONFIG_FILE_PATH, "r") as file:
            yaml.safe_load(file)
    except yaml.YAMLError as e:
        pytest.exit(f"YAML validation failed! Error: {e}")
    # Force a fresh reload of the config module
    importlib.reload(config_module)
    # Load updated config object
    config = config_module.create_and_validate_config()
    print(f"✔ Config reloaded: {config.app_config_.training_data_file}")

    return config

# @pytest.fixture
# def dataset_path(reload_config):
#     """Fixture to retrieve the dataset path dynamically from config."""
#     return reload_config.app_config_.training_data_file
    
@pytest.fixture
def sample_input_data(reload_config):
    data = load_dataset(file_name = reload_config.app_config_.training_data_file)
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        
        data[reload_config.model_config_.features],     # predictors
        data[reload_config.model_config_.target],       # target
        test_size = reload_config.model_config_.TEST_SIZE,
        random_state=reload_config.model_config_.RANDOM_STATE,   # set the random seed here for reproducibility
    )

    return X_test, y_test
