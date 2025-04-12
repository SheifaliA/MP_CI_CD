import sys
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
from vehicleinsurance_model import __version__ as _version
from vehicleinsurance_model.config.core import config
from vehicleinsurance_model.processing.data_manager import load_pipeline, pre_pipeline_preparation
from vehicleinsurance_model.processing.validation import validate_inputs

# Dynamically resolve file paths for module imports
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]

# Add parent directory to Python's module search path
sys.path.append(str(root))


# Load the trained model pipeline from saved file
pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
vehicleinsurance_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """
    Make a prediction using the trained model pipeline.

    Steps:
    1. Validate input data to ensure proper format and values.
    2. Preprocess data to match the expected model features.
    3. Use the trained model pipeline to generate predictions.
    4. Return predictions along with version and any errors encountered.
    """

    # Convert input data into a Pandas DataFrame and validate it
    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))

    # Ensure validated data is in the correct feature order before prediction
    validated_data = validated_data.reindex(columns=config.model_config_.features)

    # Initialize result structure
    results = {"predictions": None, "version": _version, "errors": errors}

    # Proceed with prediction only if there are no validation errors
    if not errors:
        predictions = vehicleinsurance_pipe.predict(validated_data)
        results = {
            "predictions": np.floor(predictions),  # Apply flooring to ensure integer output
            "version": _version,
            "errors": errors
        }
        print(results)  # Debugging output

    return results


if __name__ == "__main__":
    """
    Example usage: Running the script directly will perform a test prediction.
    """

    # Sample input data for testing the prediction function
    data_in = {
            "Gender": ["Male"], 
            "Age": [44], 
            "Driving_License":[1],
            "Region_Code": [28.0], 
            "Previously_Insured":[0],
            "Vehicle_Age": ["> 2 Years"],
            "Vehicle_Damage": ["Yes"],
            "Annual_Premium": [40454.0],
            "Policy_Sales_Channel": [26.0],
            "Vintage": [217],	
            "id":[1]
        }

    # Run a test prediction with the sample input data
    make_prediction(input_data=data_in)