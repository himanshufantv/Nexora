# utils/parser.py
import json

def safe_parse_json_string(json_str: str) -> dict:
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            return json.loads(json_str.replace("\n", "").replace("'", '"'))
        except Exception:
            return {}
