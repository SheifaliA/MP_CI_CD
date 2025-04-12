import sys
from pathlib import Path
# Dynamically resolve file paths for module imports
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]

# Add parent directory to Python's module search path
sys.path.append(str(root))
from typing import List, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError
from vehicleinsurance_model.config.core import config
from vehicleinsurance_model.processing.data_manager import pre_pipeline_preparation

def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Validates model inputs by checking for unprocessable values.
    - Applies preprocessing steps before validation.
    - Uses Pydantic to enforce proper data formats.
    - Returns cleaned data and errors if any exist.
    """

    # Apply preprocessing steps
    pre_processed = pre_pipeline_preparation(data_frame=input_df)

    # Select features from the configuration for validation
    validated_data = pre_processed[config.model_config_.features].copy()
    errors = None

    try:
        # Replace NaN values with None before validation
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()  # Capture validation errors

    return validated_data, errors


class DataInputSchema(BaseModel):
    """
    Schema for validating individual input records.
    Defines expected types for each feature used in the model.
    """

    id: Optional[int]
    Gender: Optional[str]
    Age: Optional[int]
    Driving_License: Optional[int]
    Region_Code: Optional[int]
    Vehicle_Age: Optional[str]
    Previously_Insured: Optional[int]
    Vehicle_Damage: Optional[str]
    Annual_Premium: Optional[float]
    Policy_Sales_Channel: Optional[float]
    Vintage: Optional[int]


class MultipleDataInputs(BaseModel):
    """
    Wrapper class for handling multiple input records at once.
    Ensures consistency in batch processing.
    """
    inputs: List[DataInputSchema]