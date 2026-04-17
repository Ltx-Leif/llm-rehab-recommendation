# external/llm_client.py
import sys
import os
import asyncio
import json
from typing import List, Dict, Any, Optional, Union

# Add the project root directory to sys.path
# This allows imports from sibling directories like 'core' and 'utils'
# when running this script directly.
# __file__ is the path to the current script: /path/to/project/external/llm_client.py
# os.path.dirname(__file__) is the directory of the current script: /path/to/project/external
# os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) is the project root: /path/to/project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now, the following imports should work
import openai
from openai import AsyncOpenAI # Explicit import for clarity
from core.config import settings
# Assuming 'utils' is also a sibling directory to 'external' or a module in the project root
from utils import get_logger, ExternalToolError


logger = get_logger(__name__)

# Configure the OpenAI client for SiliconFlow (or potentially Gemini if needed for text)
# Ensure API key and base URL are correctly loaded from settings
# Determine which client to initialize based on what's needed
# For now, assuming DeepSeek/SiliconFlow is still used for diagnosis text generation
deepseek_client: Optional[AsyncOpenAI] = None # Initialize with None
try:
    # Check if SiliconFlow keys are provided for the DeepSeek client
    if settings.SILICONFLOW_API_KEY and settings.SILICONFLOW_BASE_URL:
        deepseek_client = AsyncOpenAI(
            api_key=settings.SILICONFLOW_API_KEY,
            base_url=settings.SILICONFLOW_BASE_URL,
            timeout=120.0,
        )
        logger.info(f"DeepSeek Client initialized for base_url: {settings.SILICONFLOW_BASE_URL}")
    else:
        # deepseek_client remains None
        logger.warning("SiliconFlow API Key or Base URL not configured. DeepSeek LLM calls will fail.")

    # Potentially initialize Gemini client here too if needed for text tasks
    # gemini_client = ...

except Exception as e:
    logger.exception(f"Failed to initialize LLM client(s): {e}")
    deepseek_client = None # Ensure client is None if init fails

async def call_deepseek_llm(
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    model_name: str = settings.DEEPSEEK_MODEL_NAME, # Default to DeepSeek model from settings
    temperature: float = 0.7,
    max_tokens: int = 2048,
    json_mode: bool = False
) -> str:
    """
    Calls the configured DeepSeek LLM via SiliconFlow's API asynchronously.
    Handles prompt or message input. Attempts to handle JSON mode request gracefully.
    """
    # Use the initialized deepseek_client
    client_to_use = deepseek_client

    if not client_to_use:
        logger.error("DeepSeek LLM client is not initialized (check API Key/Base URL). Cannot make API call.")
        raise ExternalToolError("DeepSeek LLM client not initialized.", tool_name="LLM Client")

    if not messages and prompt:
        messages = [{"role": "user", "content": prompt}]
    elif not messages and not prompt:
        raise ValueError("Either 'prompt' or 'messages' must be provided.")
    elif not messages: # This condition implies messages is an empty list or None after the above checks
        raise ValueError("Messages list is empty or invalid.")


    logger.info(f"Calling DeepSeek LLM (model: {model_name}, temp: {temperature}, max_tokens: {max_tokens}, json_mode: {json_mode})...")
    # Simplified logging for prompt text
    log_prompt_text = ""
    if messages: # Ensure messages is not None
        last_user_message = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if last_user_message:
            content = last_user_message.get("content")
            if isinstance(content, str):
                log_prompt_text = content[:200] # Log less
            elif isinstance(content, list): # Handle multimodal content list
                text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                log_prompt_text = (" ".join(text_parts))[:200] # Log less

    logger.debug(f"LLM Prompt Text (first 200 chars): {log_prompt_text}...")

    request_params = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if json_mode:
        # Try adding the response_format parameter
        # Note: Compatibility depends on the specific API endpoint (SiliconFlow) and the openai library version
        try:
            # This check might not be perfectly accurate for non-OpenAI endpoints
            # A simpler check could be based on openai library version if known
            if hasattr(client_to_use.chat.completions, 'create'): # and 'response_format' in client_to_use.chat.completions.create.__code__.co_varnames:
                # The co_varnames check can be fragile. Let's assume if the method exists,
                # we can try sending the parameter and let the API endpoint handle it.
                # SiliconFlow might support this syntax.
                request_params["response_format"] = {"type": "json_object"}
                logger.info("Requesting JSON output mode from LLM.")
            else:
                logger.warning("Installed library version might not support 'response_format' or it's an older client. Sending standard request.")
        except Exception:
            logger.warning("Could not robustly check for 'response_format' support. Sending standard request for JSON mode.")


    try:
        response = await client_to_use.chat.completions.create(**request_params)

        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            logger.warning("LLM response structure invalid or content missing.")
            raise ExternalToolError("LLM returned empty or invalid response structure.", tool_name="LLM Client")

        content = response.choices[0].message.content
        logger.info("LLM call successful.")
        logger.debug(f"LLM Raw Response (first 500 chars): {content[:500]}...")

        # --- MODIFIED JSON HANDLING ---
        if json_mode:
            try:
                # Attempt to parse - this might fail even if JSON was requested
                json.loads(content)
                logger.info("LLM response successfully validated as JSON (as requested).")
                # Return the raw content which is expected to be JSON
            except json.JSONDecodeError as json_err:
                # Log a warning but DO NOT raise an error. Return the raw text instead.
                logger.warning(f"LLM was asked for JSON but returned invalid JSON: {json_err}. Returning raw text for fallback parsing.")
                logger.debug(f"Invalid JSON received: {content}")
                # The downstream function (_parse_llm_diagnosis_response) will handle parsing this raw text.
        # --- END MODIFIED JSON HANDLING ---

        return content.strip() # Return the raw content

    except openai.APIConnectionError as e:
        logger.error(f"LLM API Connection Error: {e}")
        raise ExternalToolError(f"Connection error: {e}", tool_name="LLM Client")
    except openai.RateLimitError as e:
        logger.error(f"LLM API Rate Limit Error: {e}")
        raise ExternalToolError(f"Rate limit exceeded: {e}", tool_name="LLM Client")
    except openai.APIStatusError as e:
        logger.error(f"LLM API Status Error: {e.status_code} - {e.response}")
        error_detail = str(e)
        try:
            # Try to get a more specific error message if the response is JSON
            if e.response and hasattr(e.response, 'json'):
                err_body = e.response.json()
                error_detail = err_body.get("error", {}).get("message", str(e))
        except json.JSONDecodeError: # If response body is not JSON
            pass
        except Exception: # Other potential errors during error parsing
            pass
        raise ExternalToolError(f"API status error ({e.status_code}): {error_detail}", tool_name="LLM Client")
    except openai.APITimeoutError as e:
        logger.error(f"LLM API Timeout Error: {e}")
        raise ExternalToolError(f"API request timed out: {e}", tool_name="LLM Client")
    except ValueError as e: # Catches ValueError from prompt/message validation
        logger.error(f"Value error during LLM call setup: {e}")
        raise ExternalToolError(f"Invalid input for LLM call: {e}", tool_name="LLM Client")
    except Exception as e:
        # Catch other unexpected errors during the API call itself
        logger.error(f"An unexpected error occurred during LLM call execution: {e}", exc_info=True)
        # Ensure a specific error type is raised for consistent handling downstream
        raise ExternalToolError(f"Unexpected LLM API error: {e}", tool_name="LLM Client")


# --- Example Usage (Keep as is) ---
async def main():
    # Ensure settings are available, especially for direct execution
    # This check is good, but relies on settings being importable first
    if not settings.SILICONFLOW_BASE_URL: # Added a check for base URL too
        print("Warning: SILICONFLOW_BASE_URL not found in settings. Tests may fail if not overridden.")
        # Optionally provide a default test URL if appropriate
        # settings.SILICONFLOW_BASE_URL = "YOUR_DEFAULT_TEST_BASE_URL"

    print("\n--- LLM Text Test (DeepSeek) ---")
    try:
        test_prompt = "你好，请介绍一下脑卒中康复。"
        response = await call_deepseek_llm(prompt=test_prompt, max_tokens=150)
        print(f"Response: {response}")
    except ExternalToolError as e:
        print(f"LLM Text Test Failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during text test: {e}")

    print("\n--- LLM JSON Mode Test (DeepSeek) ---")
    try:
        json_prompt = "列出三种常见的水果，并以JSON列表形式返回，例如： [\"水果1\", \"水果2\", \"水果3\"]"
        response_json = await call_deepseek_llm(prompt=json_prompt, json_mode=True, temperature=0.2, max_tokens=100)
        print("Raw Response (JSON requested):")
        print(response_json)
        # Validate if it's actually JSON
        try:
            parsed = json.loads(response_json)
            print("\nParsed JSON:")
            print(parsed)
        except json.JSONDecodeError:
            print("\nResponse was NOT valid JSON (but client returned raw text).") # Updated message
    except ExternalToolError as e:
        print(f"LLM JSON Test Failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during JSON test: {e}")

    print("\n-------------------------\n")


if __name__ == "__main__":
    # The sys.path modification above should allow 'from core.config import settings' to work.
    # The global 'deepseek_client' initialization happens after imports.

    # This block is for providing fallback API keys IF the settings didn't load them
    # AND the client failed to initialize globally.
    # Re-initialize client only if it's None AND we manage to set a placeholder key.
    if deepseek_client is None: # Check if global client initialization failed
        print("Global deepseek_client was not initialized. Attempting local initialization for main().")
        api_key_to_use = settings.SILICONFLOW_API_KEY # Could be None if not in .env
        base_url_to_use = settings.SILICONFLOW_BASE_URL # Could be None

        if not api_key_to_use:
            print("Warning: SILICONFLOW_API_KEY not found in settings, using placeholder for main() test.")
            api_key_to_use = "YOUR_PLACEHOLDER_KEY_FOR_TESTING_ONLY"
            # You might want to set this on the settings object if other parts of main() expect it
            # However, the primary use here is to initialize a local client for the test.
        
        if not base_url_to_use:
            print("Warning: SILICONFLOW_BASE_URL not found in settings, using placeholder for main() test.")
            # Example: "https://api.siliconflow.cn/v1" - replace with actual if you have a default test one
            base_url_to_use = "YOUR_PLACEHOLDER_BASE_URL_FOR_TESTING_ONLY"

        if api_key_to_use and base_url_to_use and api_key_to_use != "YOUR_PLACEHOLDER_KEY_FOR_TESTING_ONLY":
            try:
                # Try to re-initialize the global client if keys are now available
                # or initialize a local one for the test.
                # For simplicity, let's re-assign the global one for the scope of this main execution.
                deepseek_client = AsyncOpenAI(
                    api_key=api_key_to_use,
                    base_url=base_url_to_use,
                    timeout=120.0,
                )
                logger.info(f"DeepSeek Client re-initialized for main() with base_url: {base_url_to_use}")
            except Exception as e:
                logger.error(f"Failed to re-initialize deepseek_client in __main__: {e}")
                deepseek_client = None # Ensure it's None if re-init fails
        else:
            logger.warning("Cannot initialize DeepSeek client for main() due to missing API key or base URL (or using placeholders).")
            deepseek_client = None # Explicitly set to None if placeholders are used and you don't want to proceed


    if deepseek_client:
        asyncio.run(main())
    else:
        print("Cannot run main() example: DeepSeek client not initialized (API Key or Base URL likely missing or placeholders used).")
        print("Please ensure core.config.settings can load SILICONFLOW_API_KEY and SILICONFLOW_BASE_URL,")
        print("or that they are correctly set if you are running this script directly for testing.")