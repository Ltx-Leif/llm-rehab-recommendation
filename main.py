# main.py
from fastapi import FastAPI
# Use the specific router objects imported
from api.routes import preprocess_router, diagnosis_router
from core.config import settings
from utils import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="LLM Rehab Recommendation API",
    description="API for generating rehabilitation recommendations using multimodal patient data and LLMs.",
    version="0.2.0", # Increment version for Phase 2
)

# Include API routers
app.include_router(preprocess_router)
app.include_router(diagnosis_router) # Add the new diagnosis router

@app.get("/", tags=["Health Check"], summary="API Health Check")
async def read_root():
    logger.info("Health check endpoint '/' accessed.")
    return {"message": "LLM Rehab Recommendation API is running."}

# ... (Optional startup/shutdown events) ...

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server directly from main.py")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) # Added reload=True for development