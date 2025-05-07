import logging
import os
import inspect
from datetime import datetime
import traceback
import functools

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/nexora_flow_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)

# Create a logger instance
flow_logger = logging.getLogger("nexora_flow")

def log_entry_exit(func):
    """Decorator to print function entry and exit"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        caller = inspect.currentframe().f_back
        filename = caller.f_code.co_filename.split("/")[-1] if caller else "unknown"
        lineno = caller.f_lineno if caller else 0
        
        print(f"ENTER: {func.__name__} in {filename}")
        
        try:
            result = func(*args, **kwargs)
            print(f"EXIT: {func.__name__} in {filename}")
            return result
        except Exception as e:
            print(f"ERROR in {func.__name__}: {str(e)}")
            # Uncomment for debugging
            # print(traceback.format_exc())
            raise
            
    return wrapper

def log_flow(message, level="info"):
    """
    Previously: Log a flow message with caller information
    Now: Print a flow message with basic information
    """
    # Simple implementation that just uses print
    if level.lower() == "warning":
        print(f"WARNING: {message}")
    elif level.lower() == "error":
        print(f"ERROR: {message}")
    elif level.lower() == "debug":
        print(f"DEBUG: {message}")
    else:
        print(f"{message}")

def log_api_call(api_name, params=None):
    """
    Previously: Log an API call with parameters
    Now: Print an API call with parameters
    """
    param_str = f" with params: {params}" if params else ""
    print(f"API CALL: {api_name}{param_str}")

def log_db_operation(operation, collection, query=None):
    """
    Previously: Log a database operation
    Now: Print a database operation
    """
    query_str = f" with query: {query}" if query else ""
    print(f"DB {operation}: {collection}{query_str}")

def log_state_update(state_key, value=None):
    """
    Previously: Log a state update
    Now: Print a state update
    """
    value_str = f": {value}" if value is not None else ""
    print(f"STATE UPDATE: {state_key}{value_str}")

def log_error(error_message, exception=None):
    """
    Previously: Log an error with stack trace
    Now: Print an error message with optional exception info
    """
    print(f"ERROR: {error_message}")
    
    if exception:
        print(f"Exception: {str(exception)}")
        # If detailed debugging is needed, uncomment this line
        # print(traceback.format_exc())

"""
#-----------------------------------------#
# HOW TO USE THE FUNCTIONAL LOGGER        #
#-----------------------------------------#

The functional logger is designed to track code execution flow, API calls, database operations,
and errors throughout the StudioNexora application. Use it to help debug issues in production.

1. To log code flow (general events):
   ```python
   from utils.functional_logger import log_flow
   
   # Basic info logging
   log_flow("Starting image generation process")
   
   # Warning level logging
   log_flow("No character profiles found, using defaults", level="warning")
   
   # Error level logging (but without exception)
   log_flow("Failed to generate image, using placeholder", level="error")
   ```

2. To log entry and exit from functions (automatic tracing):
   ```python
   from utils.functional_logger import log_entry_exit
   
   @log_entry_exit
   def my_important_function(param1, param2):
       # Function code here
       return result
   ```

3. To log API calls:
   ```python
   from utils.functional_logger import log_api_call
   
   # Before making an external API call
   log_api_call("OpenAI ChatCompletion", {"model": "gpt-4", "temperature": 0.7})
   response = openai_client.chat.completions.create(...)
   ```

4. To log database operations:
   ```python
   from utils.functional_logger import log_db_operation
   
   # Before database operations
   log_db_operation("find_one", "story_projects", {"session_id": session_id})
   project = story_projects_collection.find_one({"session_id": session_id})
   ```

5. To log state updates:
   ```python
   from utils.functional_logger import log_state_update
   
   # When updating important state variables
   log_state_update("character_profiles", f"{len(profiles)} profiles")
   state.character_profiles = profiles
   ```

6. To log errors with exception details:
   ```python
   from utils.functional_logger import log_error
   
   try:
       # Some code that might fail
   except Exception as e:
       log_error("Failed to process image", e)
       # Handle the error...
   ```

View logs in the logs/nexora_flow_YYYYMMDD.log file or in the console output.
""" 