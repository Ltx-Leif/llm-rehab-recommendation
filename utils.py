# utils.py
import logging
import sys

# --- Logging Setup ---
# Configure basic logging format and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Log to console
        # You can add logging.FileHandler('app.log') here to log to a file
    ]
)

# Get a logger instance for use in other modules
def get_logger(name: str):
    """Gets a logger instance."""
    return logging.getLogger(name)

# --- Custom Exceptions ---
class ExternalToolError(Exception):
    """Custom exception for errors during external tool/API calls."""
    def __init__(self, message="Error interacting with external tool", tool_name="Unknown"):
        self.tool_name = tool_name
        self.message = f"[{tool_name} Error] {message}"
        super().__init__(self.message)

# Example usage (optional, for testing):
if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("Logging setup complete.")
    try:
        raise ExternalToolError("API connection failed", tool_name="LLM Client")
    except ExternalToolError as e:
        logger.error(f"Caught custom exception: {e}")