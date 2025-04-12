import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from vehicleinsurance_model.config.core import config
from vehicleinsurance_model.pipeline import vehicleinsurance_pipe
from vehicleinsurance_model.processing.data_manager import load_dataset, save_pipeline

# Dynamically resolve file paths for module imports
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]

# Add parent directory to Python's module search path
sys.path.append(str(root))


def run_training() -> None:
    """
    Function to train the model.
    - Loads dataset.
    - Splits data into training and testing sets.
    - Fits the training data to the pipeline.
    - Evaluates model performance on test data.
    - Saves the trained model pipeline for future use.
    """

    # Step 1: Load the training dataset
    data = load_dataset(file_name=config.app_config_.training_data_file)

    # Step 2: Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config_.features],  # Predictor variables
        data[config.model_config_.target],  # Target variable
        test_size=config.model_config_.TEST_SIZE,  # Test set proportion
        random_state=config.model_config_.RANDOM_STATE,  # Random seed for reproducibility
    )

    # Step 3: Fit the pipeline with training data
    vehicleinsurance_pipe.fit(X_train, y_train)

    # Step 4: Generate predictions on test data
    y_pred = vehicleinsurance_pipe.predict(X_test)

    # Step 5: Evaluate model performance
    print("Accuracy score:", round(accuracy_score(y_test, y_pred), 2))  # Overall classification accuracy
    print("Precision score:", precision_score(y_test, y_pred))  # Precision for positive predictions

    # Step 6: Save the trained model pipeline
    save_pipeline(pipeline_to_persist=vehicleinsurance_pipe)


# Run the training function when script is executed directly
if __name__ == "__main__":
    run_training()