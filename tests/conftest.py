import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
from sklearn.model_selection import train_test_split
import yaml
from vehicleinsurance_model.config.core import create_and_validate_config
from vehicleinsurance_model.processing.data_manager import load_dataset
# import importlib
    
@pytest.fixture
def sample_input_data():
    # config = config_dataset_path() 
    config = create_and_validate_config()
    data = load_dataset(file_name = config.app_config_.training_data_file)
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        
        data[config.model_config_.features],     # predictors
        data[config.model_config_.target],       # target
        test_size = config.model_config_.TEST_SIZE,
        random_state=config.model_config_.RANDOM_STATE,   # set the random seed here for reproducibility
    )

    return X_test, y_test
