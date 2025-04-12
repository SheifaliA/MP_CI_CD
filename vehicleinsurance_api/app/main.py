import sys
from pathlib import Path
from typing import Any
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.api import api_router
from app.config import settings

# Dynamically resolve file paths for module imports
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]

# Add parent directory to Python's module search path
sys.path.append(str(root))

# Initialize the FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,  # API title from config settings
    openapi_url=f"{settings.API_V1_STR}/openapi.json"  # Define OpenAPI documentation URL
)

# Create an API router for handling root-level requests
root_router = APIRouter()


@root_router.get("/")
def index(request: Request) -> Any:
    """
    Root endpoint that returns a simple HTML response.
    Serves as a welcome page with a link to API documentation.
    """
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)


# Include API routers for handling different routes
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(root_router)

# Configure CORS settings for security and frontend compatibility
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],  # Allows defined origins
        allow_credentials=True,  # Enables cross-origin credential sharing
        allow_methods=["*"],  # Allows all HTTP methods
        allow_headers=["*"],  # Allows all headers
    )


if __name__ == "__main__":
    import uvicorn
    # Start the FastAPI application using Uvicorn, allowing connections from any host
    uvicorn.run(app, host="0.0.0.0", port=8001)  

    # Notes:
    # - localhost resolves to 127.0.0.1
    # - host="0.0.0.0" allows access from all external hosts