# utils/logger.py
import json
import os
from datetime import datetime

def log_agent_output(agent_name: str, state):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_folder = f"logs/{agent_name}"
    os.makedirs(log_folder, exist_ok=True)
    filepath = f"{log_folder}/{timestamp}.json"

    try:
        # Only dump serializable fields
        safe_data = {}
        for key, value in state.dict().items():
            try:
                json.dumps(value)  # check if value is serializable
                safe_data[key] = value
            except (TypeError, OverflowError):
                safe_data[key] = str(value)  # fallback to string if not serializable

        with open(filepath, "w") as f:
            json.dump(safe_data, f, indent=2)
        
        print(f"📝 Logged {agent_name} output → {filepath}")
    except Exception as e:
        print(f"❌ Failed to log {agent_name} output: {e}")
