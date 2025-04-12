import sys
from typing import List
from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application configuration settings using Pydantic's BaseSettings.
    """

    # API base path for versioning
    API_V1_STR: str = "/api/v1"

    # Allowed Cross-Origin Resource Sharing (CORS) origins
    # This enables secure API access from frontend applications.
    # BACKEND_CORS_ORIGINS is a list of allowed origins.
    # Example usage: http://localhost,http://localhost:4200,http://localhost:3000
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # Frontend running on localhost:3000
        "http://localhost:8000",  # Backend API running on localhost:8000
        "https://localhost:3000",  # Secure version of frontend origin
        "https://localhost:8000",  # Secure version of backend origin
    ]

    # Project metadata
    PROJECT_NAME: str = "Vehicle Insurance Prediction API"

    class Config:
        """Pydantic configuration settings."""
        case_sensitive = True  # Enforce case-sensitive environment variable parsing

# Instantiate the settings object
settings = Settings()