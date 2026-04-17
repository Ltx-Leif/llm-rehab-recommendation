# external/image_analyzer.py
import os
import sys
import asyncio
import torch # 导入 torch 以检查 CUDA 可用性
from typing import Optional, Any, List # 确保 List 被导入
import time

# Determine project root dynamically to make path resolution more robust
# Assuming this file (image_analyzer.py) is in PROJECT_ROOT/external/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the HuatuoGPT-Vision directory, which contains cli.py and the llava subdirectory
# This path is now taken from settings
from core.config import settings
HUATUOGPT_VISION_DIR = settings.HUATUOGPT_VISION_MODEL_PATH

# Add HUATUOGPT_VISION_DIR to sys.path so 'from cli import HuatuoChatbot' works
# and cli.py can find its 'llava' module.
# This is crucial for the import system to locate the 'cli' module within huatuoGPT-Vision.
if HUATUOGPT_VISION_DIR not in sys.path:
    sys.path.insert(0, HUATUOGPT_VISION_DIR)

from models import ImageReport   # 从您的项目模型中导入 ImageReport
from utils import get_logger, ExternalToolError # 从您的项目工具中导入

logger = get_logger(__name__)

# Try to import HuatuoChatbot after potentially modifying sys.path
try:
    import importlib.util
    cli_path = os.path.join(HUATUOGPT_VISION_DIR, 'cli.py')
    spec = importlib.util.spec_from_file_location("cli", cli_path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    HuatuoChatbot = cli.HuatuoChatbot
except ImportError as e:
    logger.error(
        f"Failed to import HuatuoChatbot from cli.py, likely due to incorrect HUATUOGPT_VISION_MODEL_PATH "
        f"or missing huatuoGPT-Vision directory/files. HUATUOGPT_VISION_DIR was set to: {HUATUOGPT_VISION_DIR}. Error: {e}",
        exc_info=True
    )
    # Define a placeholder if import fails, to prevent app crash on startup,
    # but it will fail at runtime if used.
    class HuatuoChatbot: # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("HuatuoChatbot could not be imported. Check logs and HUATUOGPT_VISION_MODEL_PATH.")
        def inference(self, *args, **kwargs):
            raise NotImplementedError("HuatuoChatbot is not available.")


_huatuo_chatbot_instance: Optional[HuatuoChatbot] = None
_initialization_lock = asyncio.Lock() # Async lock for thread-safe initialization
_initialization_error: Optional[str] = None


async def get_huatuo_chatbot_instance() -> HuatuoChatbot:
    """
    Asynchronously gets or initializes the HuatuoChatbot instance.
    Ensures that the model is loaded only once.
    Raises ExternalToolError if initialization fails.
    """
    global _huatuo_chatbot_instance
    global _initialization_error

    if _huatuo_chatbot_instance:
        return _huatuo_chatbot_instance
    
    if _initialization_error: # If a previous attempt failed
        raise ExternalToolError(tool_name="HuatuoGPT-Vision Initialization", message=_initialization_error)

    async with _initialization_lock:
        # Double-check after acquiring the lock
        if _huatuo_chatbot_instance:
            return _huatuo_chatbot_instance
        if _initialization_error:
             raise ExternalToolError(tool_name="HuatuoGPT-Vision Initialization", message=_initialization_error)

        try:
            logger.info("Attempting to initialize HuatuoGPT-Vision model...")
            start_time = time.time()

            model_dir = settings.HUATUOGPT_VISION_MODEL_PATH
            if not os.path.isdir(model_dir):
                _initialization_error = f"HuatuoGPT-Vision model path does not exist or is not a directory: {model_dir}"
                logger.error(_initialization_error)
                raise ExternalToolError(tool_name="HuatuoGPT-Vision Initialization", message=_initialization_error)

            # Determine device: use CUDA if available, otherwise CPU (and log a warning)
            if torch.cuda.is_available():
                device = "cuda" # Or specify a specific CUDA device like "cuda:0"
                logger.info(f"CUDA is available. Using device: {device} for HuatuoGPT-Vision.")
            else:
                device = "cpu"
                logger.warning("CUDA not available. HuatuoGPT-Vision will run on CPU, which will be very slow.")
            
            # The HuatuoChatbot class from cli.py takes model_dir and device
            _huatuo_chatbot_instance = HuatuoChatbot(model_dir=model_dir, device=device)
            
            end_time = time.time()
            logger.info(f"HuatuoGPT-Vision model initialized successfully in {end_time - start_time:.2f} seconds.")
            return _huatuo_chatbot_instance

        except ImportError as ie: # Catch if cli.HuatuoChatbot itself has internal import errors
            _initialization_error = f"Failed to initialize HuatuoGPT-Vision due to an import error within its modules: {str(ie)}"
            logger.error(_initialization_error, exc_info=True)
            raise ExternalToolError(tool_name="HuatuoGPT-Vision Initialization", message=_initialization_error) from ie
        except Exception as e:
            _initialization_error = f"An unexpected error occurred during HuatuoGPT-Vision model initialization: {type(e).__name__} - {str(e)}"
            logger.error(_initialization_error, exc_info=True)
            # Store the error to prevent re-attempts if it's a persistent issue
            raise ExternalToolError(tool_name="HuatuoGPT-Vision Initialization", message=_initialization_error) from e


async def analyze_image_with_huatuo(image_path: str, prompt: str, request_id: str) -> ImageReport:
    """
    Analyzes a single image using the HuatuoGPT-Vision model.
    """
    log_prefix = f"[Request ID: {request_id}, Image: {os.path.basename(image_path)}] "
    logger.info(f"{log_prefix}Starting HuatuoGPT-Vision analysis.")
    
    report = ImageReport(image_ref=image_path)

    if not os.path.exists(image_path):
        report.error = f"Image file not found at path: {image_path}"
        logger.error(f"{log_prefix}{report.error}")
        return report

    try:
        bot = await get_huatuo_chatbot_instance() # Get or initialize the bot
        
        logger.debug(f"{log_prefix}HuatuoGPT-Vision prompt: '{prompt}'")
        
        # The cli.HuatuoChatbot.inference method expects a list of image paths
        # It seems the cli.py's inference method is synchronous.
        # To avoid blocking the FastAPI event loop, run it in a thread pool.
        loop = asyncio.get_event_loop()
        start_inference_time = time.time()
        try:
            analysis_result_text = await loop.run_in_executor(
                None,  # Uses the default ThreadPoolExecutor
                bot.inference, # The synchronous function to call
                prompt,        # query argument
                [image_path]   # image_paths argument (as a list)
            )
        except Exception as inference_e: # Catch errors specifically from bot.inference
            logger.error(f"{log_prefix}HuatuoGPT-Vision inference failed: {inference_e}", exc_info=True)
            raise ExternalToolError(
                tool_name="HuatuoGPT-Vision Inference",
                message=f"模型推理过程中出错: {str(inference_e)}"
            ) from inference_e
        
        inference_duration = time.time() - start_inference_time
        logger.info(f"{log_prefix}HuatuoGPT-Vision inference completed in {inference_duration:.2f} seconds.")

        if analysis_result_text and isinstance(analysis_result_text, str):
            report.analysis_text = analysis_result_text.strip()
            # Log only a part of the result to avoid flooding logs
            logger.debug(f"{log_prefix}Received analysis (first 200 chars): {report.analysis_text[:200]}...")
        else:
            report.error = "HuatuoGPT-Vision returned an empty or invalid response."
            logger.warning(f"{log_prefix}{report.error}")
            
    except ExternalToolError as ete: # Catch errors from get_huatuo_chatbot_instance or inference
         logger.error(f"{log_prefix}HuatuoGPT-Vision analysis process failed: {ete.message}")
         report.error = ete.message # ete.message already includes tool_name context
    except Exception as e:
        # Catch any other unexpected errors
        logger.exception(f"{log_prefix}Unexpected error during HuatuoGPT-Vision image analysis: {e}")
        report.error = f"Unexpected analysis error: {type(e).__name__} - {str(e)}"
    
    return report

class ImageAnalyzerClient:
    """
    Client for image analysis services.
    This class will be instantiated by PreprocessingService.
    """
    async def analyze_image(self, image_path: str, prompt: str, request_id: str) -> ImageReport:
        """
        Main method to analyze an image.
        It will call the HuatuoGPT-Vision specific analysis function.
        """
        # For now, directly calls the Huatuo-specific function.
        # This could be expanded if multiple image analysis backends were supported.
        return await analyze_image_with_huatuo(image_path, prompt, request_id)

async def clear_huatuo_chatbot_instance():
    """
    Utility function to clear the global chatbot instance, mainly for testing or controlled re-initialization.
    """
    global _huatuo_chatbot_instance
    global _initialization_error
    async with _initialization_lock:
        if _huatuo_chatbot_instance:
            logger.info("Clearing HuatuoGPT-Vision chatbot instance.")
            # If the chatbot instance has a specific close/release method, call it here.
            # For now, just setting to None.
            # e.g., if hasattr(_huatuo_chatbot_instance, 'close'): _huatuo_chatbot_instance.close()
            _huatuo_chatbot_instance = None
        _initialization_error = None # Reset any stored initialization error
    logger.info("HuatuoGPT-Vision chatbot instance cleared.")