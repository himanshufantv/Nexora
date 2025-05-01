# StudioNexora Logging System

This document describes the logging system implemented in StudioNexora to help with debugging and tracking code flow.

## Overview

The logging system is designed to provide detailed visibility into:

1. **Code Execution Flow**: Track the path through the application
2. **API Calls**: Log external service calls (OpenAI, Replicate, etc.)
3. **Database Operations**: Track MongoDB operations
4. **State Changes**: Monitor StoryState updates
5. **Errors**: Detailed error logging with stacktraces

## Log File Location

Logs are written to:
- `logs/nexora_flow_YYYYMMDD.log` (date-stamped file)
- Console output (for immediate visibility)

## How to Use the Logger

### 1. Import the Logger Functions

```python
from utils.functional_logger import (
    log_flow, 
    log_api_call, 
    log_db_operation, 
    log_state_update, 
    log_error, 
    log_entry_exit
)
```

### 2. Log Code Flow

Use `log_flow()` to track general execution flow:

```python
log_flow("Starting image generation process")
log_flow("No character profiles found, using defaults", level="warning")
log_flow("Failed to generate image, using placeholder", level="error")
```

### 3. Log Function Entry/Exit

Use the `@log_entry_exit` decorator to automatically log function entry and exit:

```python
@log_entry_exit
def generate_scene_image(prompt, character_profiles):
    # Function code here
    return image_url
```

### 4. Log API Calls

Use `log_api_call()` before making external API requests:

```python
log_api_call("OpenAI ChatCompletion", {"model": "gpt-4", "temperature": 0.7})
response = openai_client.chat.completions.create(...)
```

### 5. Log Database Operations

Use `log_db_operation()` before database calls:

```python
log_db_operation("find_one", "story_projects", {"session_id": session_id})
project = story_projects_collection.find_one({"session_id": session_id})
```

### 6. Log State Updates

Use `log_state_update()` when modifying important state:

```python
log_state_update("character_profiles", f"{len(profiles)} profiles")
state.character_profiles = profiles
```

### 7. Log Errors

Use `log_error()` to log errors with exception details:

```python
try:
    # Code that might fail
except Exception as e:
    log_error("Failed to process image", e)
    # Error handling code
```

## Log Format

Logs follow this format for easy searching and filtering:

- **Code Flow**: `🔄 FLOW: [message] in [filename]:[line] ([function])`
- **API Calls**: `🔌 API: Calling [api_name] with params: [params] in [filename]:[line]`
- **DB Operations**: `💾 DB: [operation] on [collection] with query: [query] in [filename]:[line]`
- **Function Entry**: `➡️ ENTER: [function_name] in [filename]:[line]`
- **Function Exit**: `⬅️ EXIT: [function_name] in [filename]`
- **Errors**: `❌ ERROR: [message] in [filename]:[line]`

## Debugging Tips

1. **Search by Module**: Search for a specific file to see all logs from that module
   ```
   grep "nexora_engine.py" logs/nexora_flow_20240601.log
   ```

2. **Search by Action Type**: Search for all API calls or database operations
   ```
   grep "API:" logs/nexora_flow_20240601.log
   grep "DB:" logs/nexora_flow_20240601.log
   ```

3. **Search by Error**: Find all errors
   ```
   grep "ERROR:" logs/nexora_flow_20240601.log
   ```

4. **Trace Function Flow**: Follow the entry/exit pattern of a specific function
   ```
   grep "generate_flux_image" logs/nexora_flow_20240601.log
   ```

## Adding New Loggers

The logging system is extensible. If you need to add a new log type:

1. Add a new function to `utils/functional_logger.py`
2. Follow the pattern of existing logging functions
3. Import and use it in your code 