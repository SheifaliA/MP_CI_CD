import sys
from pathlib import Path
import json
from typing import Any
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from vehicleinsurance_model import __version__ as model_version
from vehicleinsurance_model.predict import make_prediction
from app import __version__, schemas
from app.config import settings

# Dynamically resolve file paths to support imports
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]

# Add parent directory to Python's module search path
sys.path.append(str(root))

# Create an API router instance for managing endpoints
api_router = APIRouter()

@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    API health check endpoint.
    Returns basic service metadata such as the API and model version.
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, 
        api_version=__version__, 
        model_version=model_version
    )
    return health.dict()


# Example request payload for API documentation
example_input = {
    "inputs": [
        {
            "Gender": "Male", 
            "Age": 44, 
            "Driving_License": 1,
            "Region_Code": 28.0, 
            "Previously_Insured": 0,
            "Vehicle_Age": "> 2 Years",
            "Vehicle_Damage": "Yes",
            "Annual_Premium": 40454.0,
            "Policy_Sales_Channel": 26.0,
            "Vintage": 217,	
            "id": 1,
        }
    ]
}


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs = Body(..., example=example_input)) -> Any:
    """
    Prediction endpoint for vehicle insurance model.
    Accepts input data in JSON format, processes it, and returns predictions.
    """

    # Convert input JSON data into a Pandas DataFrame
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    # Replace NaN values with None for proper handling
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    # Handle errors returned by the prediction process
    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results