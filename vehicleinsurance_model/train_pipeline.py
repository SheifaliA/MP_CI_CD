import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score


from vehicleinsurance_model.config.core import config
from vehicleinsurance_model.pipeline import vehicleinsurance_pipe
from vehicleinsurance_model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name = config.app_config_.training_data_file)
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        
        data[config.model_config_.features],     # predictors
        data[config.model_config_.target],       # target
        test_size = config.model_config_.TEST_SIZE,
        random_state=config.model_config_.RANDOM_STATE,   # set the random seed here for reproducibility
    )

    # Pipeline fitting
    vehicleinsurance_pipe.fit(X_train, y_train)
    y_pred = vehicleinsurance_pipe.predict(X_test)

    # Calculate the score/error
    print("Accuracy score:", round(accuracy_score(y_test, y_pred), 2))
    print("Precision score:", precision_score(y_test, y_pred))

    # persist trained model
    save_pipeline(pipeline_to_persist = vehicleinsurance_pipe)
    
if __name__ == "__main__":
    run_training()