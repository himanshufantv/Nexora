# utils/db.py

import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_connection():
    try:
        client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"), serverSelectionTimeoutMS=1000)
        # Test the connection
        client.server_info()
        db = client["StudioNexora"]
        return db
    except Exception as e:
        print(f"⚠️ MongoDB connection error: {e}")
        print("Continuing without database connectivity...")
        return None

def get_chat_sessions_collection():
    db = get_db_connection()
    if db:
        return db["chat_sessions"]
    return None

def get_story_projects_collection():
    db = get_db_connection()
    if db:
        return db["story_projects"]
    return None

def get_writer_profiles_collection():
    db = get_db_connection()
    if db:
        return db["writer_profiles"]
    return None

def get_db():
    """Return the database object."""
    return get_db_connection()

def get_collections():
    """Return the collections we use."""
    return get_story_projects_collection(), get_chat_sessions_collection()
