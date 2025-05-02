# utils/chat_logger.py
import os, json
from datetime import datetime

def start_new_session_log() -> str:
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"logs/session_{timestamp}.json"
    with open(path, "w") as f:
        json.dump([], f)
    return path

def log_chat_turn(path: str, user_msg: str, agent_name: str, agent_output: str):
    # Fallback in case output is None
    safe_output = agent_output.strip() if isinstance(agent_output, str) else str(agent_output)

    entry = {
        "user": user_msg,
        "agent_type": agent_name,
        "agent_response": safe_output,
        "timestamp": datetime.now().isoformat()
    }

    with open(path, "r+") as f:
        data = json.load(f)
        data.append(entry)
        f.seek(0)
        json.dump(data, f, indent=2)

