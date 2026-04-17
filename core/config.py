# core/config.py
import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
# Assuming utils.py is at the root level relative to core/
import sys
PROJECT_ROOT_ABS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT_ABS not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_ABS)
from utils import get_logger # Now this should work

# Determine project root relative to this file
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load .env file from the project root
dotenv_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(dotenv_path=dotenv_path, override=True)

_logger = get_logger(__name__)
_logger.info(f"Loading environment variables from: {dotenv_path}")
if not os.path.exists(dotenv_path):
    _logger.warning(f".env file not found at {dotenv_path}. Relying on system environment variables.")


class Settings(BaseSettings):
    # ---------- Base Text LLM (Example: DeepSeek via SiliconFlow) ----------
    
    API_KEY: Optional[str] = os.getenv("API_KEY")
    BASE_URL: str = os.getenv("BASE_URL")
    # Specify the model name for text generation (DeepSeek)
    MODEL_NAME: str = os.getenv("MODEL_NAME")

    # ---------- Image Analysis LLM (HuatuoGPT-Vision) ----------
    # Path to the local HuatuoGPT-Vision model directory
    # This directory should contain the model weights, tokenizer, and the 'cli.py' script along with 'llava' subdir.
    HUATUOGPT_VISION_MODEL_PATH: str = os.getenv("HUATUOGPT_VISION_MODEL_PATH", os.path.join(PROJECT_ROOT, "huatuoGPT-Vision"))

    # ---------- Vector DB (ChromaDB) ----------
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", os.path.join(PROJECT_ROOT, "chroma_db_persist"))
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "rehab_knowledge_base")
    
    # ---------- API Settings ----------
    API_VERSION: str = "v1"
    API_PREFIX: str = f"/api/{API_VERSION}"
    APP_NAME: str = "LLM Rehab Recommendation API"

    # ---------- Logging ----------
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # ---------- Development Settings ----------
    RELOAD_UVICORN: bool = os.getenv("RELOAD_UVICORN", "True").lower() == "true"
    
    # ---------- Resource Paths (ensure they are absolute or correctly relative) ----------
    KB_DOCS_DIR: str = os.getenv("KB_DOCS_DIR", os.path.join(PROJECT_ROOT, "external", "kb_md_doc"))


    class Config:
        # Specify the .env file path relative to the project root if needed
        env_file = dotenv_path # Use the determined path
        env_file_encoding = "utf-8"
        extra = "ignore"

# Instantiate settings after defining the class
settings = Settings()

# --- Log Confirmation ---
_logger.info(f"Text LLM Model (SiliconFlow compatible): {settings.MODEL_NAME}")
_logger.info(f"HuatuoGPT-Vision Model Path: {settings.HUATUOGPT_VISION_MODEL_PATH}")


# Vector DB Configuration Logging
_logger.info(f"ChromaDB Persist Dir: {settings.CHROMA_PERSIST_DIR}")
_logger.info(f"Embedding Model Name: {settings.EMBEDDING_MODEL_NAME}")
_logger.info(f"Chroma Collection Name: {settings.CHROMA_COLLECTION_NAME}")

# Create ChromaDB directory if it doesn't exist
if not os.path.exists(settings.CHROMA_PERSIST_DIR):
    try:
        os.makedirs(settings.CHROMA_PERSIST_DIR)
        _logger.info(f"Created ChromaDB persistence directory: {settings.CHROMA_PERSIST_DIR}")
    except Exception as e:
        _logger.error(f"Failed to create ChromaDB persistence directory {settings.CHROMA_PERSIST_DIR}: {e}")

if not os.path.exists(settings.HUATUOGPT_VISION_MODEL_PATH) or not os.path.isdir(settings.HUATUOGPT_VISION_MODEL_PATH):
    _logger.error(f"HUATUOGPT_VISION_MODEL_PATH '{settings.HUATUOGPT_VISION_MODEL_PATH}' does not exist or is not a directory. Image analysis will likely fail.")
else:
    # Check for cli.py in the model path as a quick sanity check
    cli_py_path = os.path.join(settings.HUATUOGPT_VISION_MODEL_PATH, "cli.py")
    if not os.path.exists(cli_py_path):
        _logger.warning(f"cli.py not found in HUATUOGPT_VISION_MODEL_PATH: {cli_py_path}. Ensure the path is correct and contains the necessary scripts.")