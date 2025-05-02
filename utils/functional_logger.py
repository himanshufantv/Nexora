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
    """Decorator to log function entry and exit"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        caller = inspect.currentframe().f_back
        filename = caller.f_code.co_filename.split("/")[-1] if caller else "unknown"
        lineno = caller.f_lineno if caller else 0
        
        flow_logger.info(f"➡️ ENTER: {func.__name__} in {filename}:{lineno}")
        
        try:
            result = func(*args, **kwargs)
            flow_logger.info(f"⬅️ EXIT: {func.__name__} in {filename}")
            return result
        except Exception as e:
            flow_logger.error(f"❌ ERROR in {func.__name__}: {str(e)}")
            flow_logger.debug(traceback.format_exc())
            raise
            
    return wrapper

def log_flow(message, level="info"):
    """Log a flow message with caller information"""
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename.split("/")[-1]
    lineno = frame.f_lineno
    caller = frame.f_code.co_name
    
    log_message = f"🔄 FLOW: {message} in {filename}:{lineno} ({caller})"
    
    if level.lower() == "debug":
        flow_logger.debug(log_message)
    elif level.lower() == "warning":
        flow_logger.warning(log_message)
    elif level.lower() == "error":
        flow_logger.error(log_message)
    else:
        flow_logger.info(log_message)

def log_api_call(api_name, params=None):
    """Log an API call with parameters"""
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename.split("/")[-1]
    lineno = frame.f_lineno
    
    param_str = f" with params: {params}" if params else ""
    flow_logger.info(f"🔌 API: Calling {api_name}{param_str} in {filename}:{lineno}")

def log_db_operation(operation, collection, query=None):
    """Log a database operation"""
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename.split("/")[-1]
    lineno = frame.f_lineno
    
    query_str = f" with query: {query}" if query else ""
    flow_logger.info(f"💾 DB: {operation} on {collection}{query_str} in {filename}:{lineno}")

def log_state_update(state_key, value=None):
    """Log a state update"""
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename.split("/")[-1]
    lineno = frame.f_lineno
    
    value_str = f": {value}" if value is not None else ""
    flow_logger.info(f"🔄 STATE: Updated {state_key}{value_str} in {filename}:{lineno}")

def log_error(error_message, exception=None):
    """Log an error with stack trace"""
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename.split("/")[-1]
    lineno = frame.f_lineno
    
    flow_logger.error(f"❌ ERROR: {error_message} in {filename}:{lineno}")
    
    if exception:
        flow_logger.debug(traceback.format_exc())
        
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