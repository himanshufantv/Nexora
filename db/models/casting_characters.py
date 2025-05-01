# db/models/casting_characters.py

from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["StudioNexora"]  # ✅ Make sure this matches your correct database name
casting_characters_collection = db["casting_characters"]

def get_candidate_characters():
    """Fetch all candidate characters from database."""
    characters = list(casting_characters_collection.find({}))
    return characters

def check_compatibility(char1: dict, char2: dict) -> bool:
    """Check if two characters are compatible based on 'compatible_with' field."""
    if not char1 or not char2:
        return False
    
    char1_compat = char1.get("compatible_with", {})
    char2_compat = char2.get("compatible_with", {})

    # Check if each other's _id exists in compatible list
    return str(char2.get("_id")) in char1_compat and str(char1.get("_id")) in char2_compat
